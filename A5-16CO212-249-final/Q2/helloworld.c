/*Bidyadhar Mohanty(16CO212)
Soham Patil(16CO249)*/
#include <stdio.h>
#include <omp.h>
void print(int threadIDfromMain)
{
	printf("Hello World:(%d)\n",threadIDfromMain );
}
int main()
{
	#pragma omp parallel
	{
		int ID_main=omp_get_thread_num();
		print(ID_main);
	}
	return 0;
}
