#include "common.h"
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

int h_nx, h_ny, h_n_cells;
double h_cell_size;

// Device arrays
int *d_cell_count;       // Number of particles per cell
int *d_cell_offset;      // Starting index of each cell in sorted array
int *d_cell_pos;         // Position counter for sorting
particle_t *d_sorted_particles; // Sorted particle array
int *d_original_indices; // Maps sorted index to original index

// Clamp function to ensure indices stay within bounds
__device__ int clamp(int val, int min_val, int max_val) {
    return min(max(val, min_val), max_val);
}

// Compute cell ID with clamping for safety
__device__ int get_cell_id(double x, double y, double cell_size, int nx, int ny) {
    int cx = clamp((int)(x / cell_size), 0, nx - 1);
    int cy = clamp((int)(y / cell_size), 0, ny - 1);
    return cy * nx + cx;
}

// Kernel to count particles per cell
__global__ void compute_cell_count(particle_t *particles, int num_parts, double cell_size, int nx, int ny, int *cell_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t p = particles[tid];
    int cell_id = get_cell_id(p.x, p.y, cell_size, nx, ny);
    atomicAdd(&cell_count[cell_id], 1);
}

// Kernel to build sorted particle array based on cell IDs
__global__ void build_sorted_particles(particle_t *particles, int num_parts, double cell_size, int nx, int ny, int *cell_offset, int *cell_pos, particle_t *sorted_particles, int *original_indices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t p = particles[tid];
    int cell_id = get_cell_id(p.x, p.y, cell_size, nx, ny);
    int local_pos = atomicAdd(&cell_pos[cell_id], 1);
    int idx = cell_offset[cell_id] + local_pos;
    sorted_particles[idx] = p;
    original_indices[idx] = tid;
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor, double &ax, double &ay) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    ax += coef * dx;
    ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t *sorted_particles, int *original_indices, int num_parts, double cell_size, int nx, int ny, int *cell_offset, int *cell_count, particle_t *particles) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t p = sorted_particles[tid];
    int original_idx = original_indices[tid];
    int cx = clamp((int)(p.x / cell_size), 0, nx - 1);
    int cy = clamp((int)(p.y / cell_size), 0, ny - 1);
    double ax = 0.0, ay = 0.0;

    // Check 3x3 neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        int neighbor_cy = cy + dy;
        if (neighbor_cy < 0 || neighbor_cy >= ny) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_cx = cx + dx;
            if (neighbor_cx < 0 || neighbor_cx >= nx) continue;
            int cell_id = neighbor_cy * nx + neighbor_cx;
            int start = cell_offset[cell_id];
            int end = start + cell_count[cell_id];
            for (int j = start; j < end; j++) {
                particle_t neighbor = sorted_particles[j];
                apply_force_gpu(p, neighbor, ax, ay);
            }
        }
    }
    particles[original_idx].ax = ax;
    particles[original_idx].ay = ay;
}

// Move particles using Velocity Verlet integration and bounce off walls
__global__ void move_gpu(particle_t *particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t *p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

void init_simulation(particle_t *parts, int num_parts, double size) {
    h_cell_size = cutoff * 2.5;
    h_nx = (int)ceil(size / h_cell_size);
    h_ny = (int)ceil(size / h_cell_size);
    h_n_cells = h_nx * h_ny;

    cudaMalloc(&d_cell_count, h_n_cells * sizeof(int));
    cudaMalloc(&d_cell_offset, h_n_cells * sizeof(int));
    cudaMalloc(&d_cell_pos, h_n_cells * sizeof(int));
    cudaMalloc(&d_sorted_particles, num_parts * sizeof(particle_t));
    cudaMalloc(&d_original_indices, num_parts * sizeof(int));
}

void simulate_one_step(particle_t *parts, int num_parts, double size) {
    int blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Reset cell counts
    cudaMemset(d_cell_count, 0, h_n_cells * sizeof(int));
    compute_cell_count<<<blocks, NUM_THREADS>>>(parts, num_parts, h_cell_size, h_nx, h_ny, d_cell_count);

    // Compute offsets with prefix sum using Thrust
    thrust::device_ptr<int> cell_count_ptr(d_cell_count);
    thrust::device_ptr<int> cell_offset_ptr(d_cell_offset);
    thrust::exclusive_scan(cell_count_ptr, cell_count_ptr + h_n_cells, cell_offset_ptr);

    // Build sorted particles array
    cudaMemset(d_cell_pos, 0, h_n_cells * sizeof(int));
    build_sorted_particles<<<blocks, NUM_THREADS>>>(parts, num_parts, h_cell_size, h_nx, h_ny, d_cell_offset, d_cell_pos, d_sorted_particles, d_original_indices);

    // Compute forces
    compute_forces_gpu<<<blocks, NUM_THREADS>>>(d_sorted_particles, d_original_indices, num_parts, h_cell_size, h_nx, h_ny, d_cell_offset, d_cell_count, parts);

    // Move particles
    move_gpu<<<blocks, NUM_THREADS>>>(parts, num_parts, size);
}