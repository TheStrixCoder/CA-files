//16CO212 16CO249
//Computer Architecture Lab Assignment 0
//Question 2

//Array generated in the program is 1^2,2^2,3^2,...(N)^2 and here N=500 and Value of N is changable.

#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define BLOCK_SIZE 1024
#define N 500

__global__ void AddArray(float *A, float* ans)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + tid;

 for(unsigned int s = blockDim.x / 2; s>0; s >>=1)
 {
  if(tid < s)
  {
    A[i] += A[i + s];
  }
  __syncthreads();

 }
  if(tid == 0)                      //Returns the sum of the array elements
  {
    atomicAdd(ans, A[i]);
  }
}

int main()
{
  float* A;
  float* ans;
  float* d_A;
  float* d_ans;

  A  = (float *) malloc(N * sizeof(float));
  int i;

   for(i=0; i<N; i++)               
  {
    A[i] = (float)(i*2);
  }

  int blocks;

  blocks = ceil(N/1024.00);
  
  ans = (float*) malloc(sizeof(float));
  
  *ans = 0;
  //Allocating device_memory 

  cudaMalloc((void **)&d_A,  N * sizeof(float));
  
  cudaMalloc((void **)&d_ans, sizeof(float));
  
  //Copying the memory from host to device
  cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ans, ans, sizeof(float), cudaMemcpyHostToDevice);
  

  //invoke kernel
  
  AddArray<<<blocks, 1024>>>(d_A, d_ans);

  cudaMemcpy(ans, d_ans, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Sum of generated array= %f\n", *ans);

 //Freeing the memory
  cudaFree(d_A);
 
  cudaFree(d_ans);
 
  free(A);
  free(ans);
   
   return 0;
}