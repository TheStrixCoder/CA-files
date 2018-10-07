#include<stdio.h>
#include<cuda.h>
#define BLOCK_SIZE 16

// CUDA code to add matrix. It linearizes the 2D matrix and adds them on different threads.
__global__ static void AddMatrix(float *dev_buf1, float *dev_buf2, float *dev_buf_s, size_t pitch, int row_size, int col_size)

{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int index = pitch/sizeof(float);
	if(tidx<row_size && tidy<col_size)
	{
		dev_buf_s[tidx * index  + tidy] = dev_buf1[tidx * index + tidy] + dev_buf2[tidx * index + tidy];
	}
}


//Print Matrix
void printMatrix(float *lin_matrix, int row_size, int col_size)

{
	  for(int idxM = 0; idxM < row_size; idxM++)
	{
		for(int idxN = 0; idxN < col_size; idxN++)
		{
			printf("%f  ",lin_matrix[(idxM * col_size) + idxN]);
		}
		printf("\n");
	}

	printf("\n");
}

int main()

{
	  int row_size=100,col_size=100;

	  //Allocation of memory
	  float *host_mat1 = (float*)malloc(row_size * col_size * sizeof(float));
	  float *host_mat2 = (float*)malloc(row_size * col_size * sizeof(float));
	  float *host_sum = (float*)malloc(row_size * col_size * sizeof(float));

	  //Fill matrix with random numbers
	  for(int j=0;j<(row_size*col_size);j++)
	  {
			host_mat1[j]=((float)rand()/(float)RAND_MAX)*10000;
			host_mat2[j]=((float)rand()/(float)RAND_MAX)*10000;
	  }

	  //Print input matrixs
	  printf("==================Matrix 1===========================\n");
	  printMatrix(host_mat1, row_size, col_size);

	  printf("===========================Matrix 2==========================\n");
	  printMatrix(host_mat2, row_size, col_size);

	  //CUDA allocation on device
	  float *dev_mat1, *dev_mat2, *dev_mat_sum;
	  size_t dev_mat_p;
	  cudaMallocPitch((void**)&dev_mat1,&dev_mat_p,col_size*sizeof(float),row_size);
	  cudaMallocPitch((void**)&dev_mat2,&dev_mat_p,col_size*sizeof(float),row_size);
	  cudaMallocPitch((void**)&dev_mat_sum,&dev_mat_p,col_size*sizeof(float),row_size);

	  //Copy data to device
	  cudaMemcpy2D(dev_mat1,dev_mat_p,host_mat1,col_size * sizeof(float), col_size * sizeof(float), row_size, cudaMemcpyHostToDevice);
	  cudaMemcpy2D(dev_mat2,dev_mat_p,host_mat2,col_size * sizeof(float), col_size * sizeof(float), row_size, cudaMemcpyHostToDevice);

	  //Threads and Block sizes
	  dim3 blocks(1,1,1);
	  dim3 threads_per_block(BLOCK_SIZE,BLOCK_SIZE,1);
	  blocks.x=((row_size/BLOCK_SIZE) + (((row_size)%BLOCK_SIZE)==0?0:1));
	  blocks.y=((col_size/BLOCK_SIZE) + (((col_size)%BLOCK_SIZE)==0?0:1));

	  //Function call to add
	  AddMatrix<<<blocks, threads_per_block>>>(dev_mat1, dev_mat2, dev_mat_sum, dev_mat_p, row_size,col_size);

	  cudaThreadSynchronize();

	  //Copy back result matrix to host
	  cudaMemcpy2D(host_sum, col_size * sizeof(float),dev_mat_sum, dev_mat_p, col_size * sizeof(float), row_size, cudaMemcpyDeviceToHost);

	  //Free CUDA device memory
	  cudaFree(dev_mat1);
	  cudaFree(dev_mat2);
	  cudaFree(dev_mat_sum);

	  
	  printf("=================Matrix Sum=========================\n");
	  printMatrix(host_sum, row_size, col_size);
}
