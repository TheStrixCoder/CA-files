/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <omp.h>
#include <stdio.h>
#define SIZE 65536

void seq_daxpy(double X[],double Y[],double a)
{
	int i;
	for(i=0;i<SIZE;++i)
		X[i] = a*X[i] + Y[i];
}

void parallel_daxpy(double X[],double Y[],double a,int NUM_THREADS)
{
	int n_threads=0;
	omp_set_num_threads(NUM_THREADS);
	int i;
	#pragma omp parallel for private(i)
		for(i=0;i<SIZE;i++)
		{
			X[i] = a*X[i] + Y[i];
		}

}
int main()
{
	double X[SIZE],Y[SIZE],a=1;
	int i;
	//Filling the arrays with some random values
	for(i=0;i<SIZE;++i)
		X[i] = Y[i] = i+1;

	//Serial Execution
	double time_par,time_seq;
	time_seq = omp_get_wtime();
	seq_daxpy(X,Y,a);
	time_seq = omp_get_wtime() - time_seq;
	printf("Time taken for serial function : %lf ms\n", time_seq*100);

	//Parallel Execution
	int NUM_THREADS = 2;
	while(NUM_THREADS<=20)
	{
		time_par = omp_get_wtime();
		parallel_daxpy(X,Y,a,NUM_THREADS);
		time_par = omp_get_wtime() - time_par;
		printf("Speed up : %lf  Threads : %d\n",time_par/time_seq,NUM_THREADS);
		
		NUM_THREADS++;
	}
}
