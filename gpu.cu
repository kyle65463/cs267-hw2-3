#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

typedef struct {
    int nx, ny;
    double cell_size;
    int* bin_counts;
    int* bin_starts;
    int* sorted_indices;
    int* particle_bins;
} gpu_grid_t;

gpu_grid_t d_grid;
int num_bins;
int blks;

__device__ int get_bin_index(double x, double y, int nx, int ny, double cell_size) {
    int bin_x = int(x / cell_size);
    int bin_y = int(y / cell_size);
    
    bin_x = max(0, min(nx-1, bin_x));
    bin_y = max(0, min(ny-1, bin_y));
    
    return bin_y * nx + bin_x;
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    
    if (r2 > cutoff * cutoff)
        return;
    
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_bins_kernel(particle_t* particles, int num_parts, 
                                   int* particle_bins, int nx, int ny, double cell_size) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parts) return;
    
    particle_t* p = &particles[pid];
    particle_bins[pid] = get_bin_index(p->x, p->y, nx, ny, cell_size);
}

__global__ void count_particles_per_bin_kernel(int* particle_bins, int num_parts, int* bin_counts) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parts) return;
    
    int bin_id = particle_bins[pid];
    atomicAdd(&bin_counts[bin_id], 1);
}

__global__ void sort_particles_kernel(particle_t* particles, int num_parts, 
                                     int* particle_bins, int* bin_starts, int* sorted_indices) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parts) return;
    
    int bin_id = particle_bins[pid];
    int insert_pos = atomicAdd(&bin_starts[bin_id], 1);
    sorted_indices[insert_pos] = pid;
}

__global__ void reset_bin_starts_kernel(int* bin_starts, int* prefix_sum, int num_bins) {
    int bin_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_id >= num_bins) return;
    
    bin_starts[bin_id] = prefix_sum[bin_id];
}

__global__ void compute_forces_grid_kernel(particle_t* particles, int num_parts, 
                                        int* sorted_indices, int* bin_counts, 
                                        int* bin_starts, int nx, int ny, double cell_size) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parts) return;
    
    particle_t* p = &particles[pid];
    p->ax = p->ay = 0;
    
    // Calculate particle's bin
    int bin_x = int(p->x / cell_size);
    int bin_y = int(p->y / cell_size);
    
    // Clamp to grid bounds
    bin_x = max(0, min(nx-1, bin_x));
    bin_y = max(0, min(ny-1, bin_y));
    
    for (int dy = -1; dy <= 1; dy++) {
        int neighbor_y = bin_y + dy;
        if (neighbor_y < 0 || neighbor_y >= ny) continue;
        
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_x = bin_x + dx;
            if (neighbor_x < 0 || neighbor_x >= nx) continue;
            
            int neighbor_bin = neighbor_y * nx + neighbor_x;
            int start_idx = bin_starts[neighbor_bin];
            int end_idx = start_idx + bin_counts[neighbor_bin];
            
            // Check all particles in this neighbor bin
            for (int j = start_idx; j < end_idx; j++) {
                int neighbor_idx = sorted_indices[j];
                if (neighbor_idx == pid) continue; // Skip self
                apply_force_gpu(*p, particles[neighbor_idx]);
            }
        }
    }
}

// Kernel to move particles
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_parts) return;
    
    particle_t* p = &particles[pid];
    
    // Velocity Verlet integration
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Bounce from walls
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Calculate grid dimensions
    d_grid.cell_size = cutoff * 2.0;
    d_grid.nx = (int)(size / d_grid.cell_size) + 1;
    d_grid.ny = (int)(size / d_grid.cell_size) + 1;
    num_bins = d_grid.nx * d_grid.ny;
    
    // Calculate blocks for kernel launches
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    
    // Allocate device memory
    cudaMalloc(&d_grid.bin_counts, num_bins * sizeof(int));
    cudaMalloc(&d_grid.bin_starts, num_bins * sizeof(int));
    cudaMalloc(&d_grid.sorted_indices, num_parts * sizeof(int));
    cudaMalloc(&d_grid.particle_bins, num_parts * sizeof(int));
    
    // Initialize device memory
    cudaMemset(d_grid.bin_counts, 0, num_bins * sizeof(int));
    cudaMemset(d_grid.bin_starts, 0, num_bins * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset bin counts
    cudaMemset(d_grid.bin_counts, 0, num_bins * sizeof(int));
    
    compute_bins_kernel<<<blks, NUM_THREADS>>>(parts, num_parts, 
                                             d_grid.particle_bins, d_grid.nx, d_grid.ny, d_grid.cell_size);
    
    count_particles_per_bin_kernel<<<blks, NUM_THREADS>>>(d_grid.particle_bins, num_parts, d_grid.bin_counts);
    
    thrust::device_ptr<int> dev_counts(d_grid.bin_counts);
    thrust::device_ptr<int> dev_starts(d_grid.bin_starts);
    thrust::exclusive_scan(dev_counts, dev_counts + num_bins, dev_starts);
    
    // Make a copy for tracking insertion positions during sorting
    int* temp_bin_starts;
    cudaMalloc(&temp_bin_starts, num_bins * sizeof(int));
    cudaMemcpy(temp_bin_starts, d_grid.bin_starts, num_bins * sizeof(int), cudaMemcpyDeviceToDevice);
    
    sort_particles_kernel<<<blks, NUM_THREADS>>>(parts, num_parts, 
                                               d_grid.particle_bins, temp_bin_starts, d_grid.sorted_indices);
    
    // Free temporary memory
    cudaFree(temp_bin_starts);
    
    compute_forces_grid_kernel<<<blks, NUM_THREADS>>>(parts, num_parts, 
                                                   d_grid.sorted_indices, d_grid.bin_counts, 
                                                   d_grid.bin_starts, d_grid.nx, d_grid.ny, d_grid.cell_size);
    
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}