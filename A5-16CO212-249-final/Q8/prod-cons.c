/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <omp.h>
#include <stdio.h>
#define SIZE 100000
int flag = 0;
void random_populate(int N,double A[])
{
	for(int i=0;i<N;++i)
		A[i] = 1;
	printf("Random producer populated data\n");
	#pragma omp flush
	flag = 1;

	#pragma omp flush(flag)
}

double array_sum(int N,double A[])
{
	double sum = 0.0;
	int p_flag;
	while(1)
	{
		p_flag = 0;
		#pragma omp flush(flag)
		p_flag = flag;

		if(p_flag)
			break;
	}

	#pragma omp flush
	for(int i=0;i<N;++i)
		sum = sum + A[i];
	
	printf("Consumer calculated Array sum\n" );
	return sum;
}

double sequential_producer_consumer()
{
	double A[SIZE];
	random_populate(SIZE,A);
	double sum = array_sum(SIZE,A);
	return sum;
}

double parallel_producer_consumer()
{
	double A[SIZE];
	double  sum = 0.0;
	omp_set_num_threads(2);
	#pragma omp parallel sections
	{
		#pragma omp section
			random_populate(SIZE,A);

		#pragma omp section
			sum = array_sum(SIZE,A);
	}

	return sum;
}

int main()
{
	double time_seq,time_par,sum=0.0;

	//Sequential Producer-Consumer
	time_seq = omp_get_wtime();
	sum = sequential_producer_consumer();
	time_seq = omp_get_wtime() - time_seq;
	printf("Time: %lf seconds, Sequential sum: %lf \n",time_seq,sum);

	//Parallel Producer-Consumer
	time_par = omp_get_wtime();
	sum = parallel_producer_consumer();
	time_par = omp_get_wtime() - time_par;
	printf("Time: %lf seconds, Parallel sum: %lf \n",time_par,sum);
	printf("Speed up: %lf\n", time_par/time_seq);
}