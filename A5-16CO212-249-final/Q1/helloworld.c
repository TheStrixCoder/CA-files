/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <stdio.h>
#include <omp.h>
int main()
{
	#pragma omp parallel
	{
		printf("Hello World\n");
	}
	return 0;
}
