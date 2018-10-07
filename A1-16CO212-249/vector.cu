#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// CUDA kernel. Each thread takes care of one element of sum
__global__ void vector_add(double *a, double *b, double *sum, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        sum[id] = a[id] + b[id];
}
 
int main()
{
    // Size of vectors
    int n = 100000;
 
    // Host input vectors
    double *host_p;
    double *host_q;
    //Host output vector
    double *host_sum;
 
    // Device input vectors
    double *device_p;
    double *device_q;
    //Device output vector
    double *device_sum;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    host_p = (double*)malloc(bytes);
    host_q = (double*)malloc(bytes);
    host_sum = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&device_p, bytes);
    cudaMalloc(&device_q, bytes);
    cudaMalloc(&device_sum, bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        host_p[i] =((float)rand()/(float)RAND_MAX)*100;
        host_q[i] =((float)rand()/(float)RAND_MAX)*100;
    }
 
    // Copy host vectors to device
    cudaMemcpy( device_p, host_p, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( device_q, host_q, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    vector_add<<<gridSize, blockSize>>>(device_p, device_q, device_sum, n);
 
    // Copy array back to host
    cudaMemcpy( host_sum, device_sum, bytes, cudaMemcpyDeviceToHost );
 
    

    for(i=0; i<n; i++)
        printf("%f+%f=%f\n",host_p[i],host_q[i],host_sum[i]);
 
    // Release device memory
    cudaFree(device_p);
    cudaFree(device_q);
    cudaFree(device_sum);
 
    // Release host memory
    free(host_p);
    free(host_q);
    free(host_sum);
 
    return 0;
}
