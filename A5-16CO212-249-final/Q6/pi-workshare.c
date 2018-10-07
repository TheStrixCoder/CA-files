/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <stdio.h>
#include <omp.h>

double pi_sequential(long n_steps)
{
	double sum=0.0,x;
	int i;
	for(i=0;i<n_steps;++i)
	{
		x = (i + 0.5)/n_steps;
		sum = sum + 4.0/(1.0 + x*x);
	}
	return sum/n_steps;
}
double pi_parallel(long n_steps,int num_threads)
{
	double sum=0.0,x;
	int i;
	omp_set_num_threads(num_threads);
	#pragma omp parallel for reduction(+:sum) private(x)
		for(i=0;i<n_steps;++i)
		{
			x = (i + 0.5)/n_steps;
			sum = sum + 4.0/(1.0 + x*x);
		}
	
	return sum/n_steps;
}
int main()
{
	int i;
	static long n_steps = 10000000;
	double time_sequence,time_parallel,pi;
	//Sequential Pi Calculation
	time_sequence = omp_get_wtime();
	pi = pi_sequential(n_steps);
	time_sequence = omp_get_wtime() - time_sequence;
	printf("Value of Pi by Sequential Calculation  : %lf\n",pi );
	printf("Time taken: %lf ms\n",time_sequence*100);

	//Parallel Pi Calculation
	printf("Parallel Calculation initiated:\n");
	int num_threads =2;
	while(num_threads<=20)
	{
		time_parallel = omp_get_wtime();
		pi = pi_parallel(n_steps,num_threads);
		time_parallel = omp_get_wtime()-time_parallel;	
		printf("Pi value:  %lf \t Speedup obtained: %lf \t Threads nos: %d\n", pi,time_parallel/time_sequence,num_threads);		
		num_threads++;
	}
}	