
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "find.h"
#define SIZE_OF_LONG_INT 64

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"


void __global__ find(unsigned long long  int* x, unsigned long long  int* new_x, unsigned int N)
{
	unsigned int n = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
	int pos, sh, p;
	int NNN;
	cuPrintf("in find\n");
	return;
}


__global__ void addKernel()
{
   
    cuPrintf("add kernel\n");
}



int main()
{
	unsigned long long * x, * new_x, n = 1;

	cudaMalloc(&x, sizeof(unsigned long long));
	cudaMalloc(&new_x, sizeof(unsigned long long));
    cudaPrintfInit();

    addKernel << <1, 10 >> > ();
	find << <1, 1 >> > (x,new_x,n);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    return 0;
}