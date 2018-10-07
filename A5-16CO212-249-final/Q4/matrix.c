/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <stdio.h>
#include <omp.h>
#define SIZE 1000

int matrix_1[SIZE][SIZE],matrix_2[SIZE][SIZE],result_sequence[SIZE][SIZE],result_parallel[SIZE][SIZE];

void seq_matrix_mult(int matrix_1[][SIZE],int matrix_2[][SIZE])
{
	int i,j,k;

	for(i=0;i<SIZE;++i)
	{
		for(j=0;j<SIZE;++j)
		{
			result_sequence[i][j]=0;
			for(k=0;k<SIZE;++k)
			{
				result_sequence[i][j] = result_sequence[i][j] + matrix_1[i][k]*matrix_2[k][j];
			}
		}
	}
}

void parellel_mat_mult(int matrix_1[][SIZE],int matrix_2[][SIZE],int NUM_THREADS)
{

	int n_threads;
	omp_set_num_threads(NUM_THREADS);
	int i,j,k;
	#pragma omp parallel for private(i)
		for(i=0;i<SIZE;i++)
		{
			#pragma omp parallel for private(j)
			for(j=0;j<SIZE;++j)
			{
				int t_sum = 0;
				#pragma omp parallel for reduction(+:t_sum) private(k)
				for(k=0;k<SIZE;++k)
				{
					t_sum = t_sum + matrix_1[i][k]*matrix_2[k][j];
				}

				#pragma omp critical
				{
					result_parallel[i][j] = t_sum;
				}
			}
		}
}

int check_equality_matrix()
{
	for(int i=0;i<SIZE;++i)
		for(int j=0;j<SIZE;++j)
			if(result_sequence[i][j] != result_parallel[i][j])
				return 0;
	return 1;
}
int main()
{
	int i,j;
	for(i=0;i<SIZE;++i)
		for(j=0;j<SIZE;++j)
			matrix_1[i][j]=matrix_2[i][j] = 1;

	double time_sequence = omp_get_wtime(),time_parallel;
	seq_matrix_mult(matrix_1,matrix_2);
	time_sequence = omp_get_wtime() - time_sequence;
	int NUM_THREADS=2;
	while(NUM_THREADS<=20)
	{
		time_parallel = omp_get_wtime();
		parellel_mat_mult(matrix_1,matrix_2,NUM_THREADS);
		time_parallel = omp_get_wtime() - time_parallel;
		if(check_equality_matrix())
			printf("Speed up obtained: %lf \t Threads: %d\n",time_parallel/time_sequence,NUM_THREADS); 
		else
			printf("Mismatch result\n");
		
		NUM_THREADS++;
	}
}	