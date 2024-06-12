
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "find.h"
#define SIZE_OF_LONG_INT 64

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"


//редукция большого массива 64-разрядных целых к массиву размером в 64 раза меньше,
//где каждому целому числу изначального массива соответствует 1 бит, ненулевой если
//в соответсвующем элементе исходного массива был хотя бы один ненулевлй бит
__device__ unsigned long long int get_bit_position(unsigned long long  int x, int n)
{
	int pos, sh;               // n - позиция вектора x в большом векторе правой части
	unsigned long long p;
#ifdef bbbb
	unsigned long long p0;  // n_minor - номер 64-битной последовательности в маленьком векторе слева
	char str[500];
#endif

	// sh - position of 1 in the 64 bit sequence meaning that the corresponding
	// element of the long int array "x" has some non-zero bit

// позиция первого ненулевого бита в 64-разрядном целом числе
	pos = __ffsll(x);
	cuPrintf("__ffsll %d \n",pos);
	return pos;


	//printf("get_bit_position pos %d x %llu n %d \n",pos,x,n);

	//элементы массива нумеруются с нуля, биты с единицы
//	sh = position_in_64bit_word(n + 1, SIZE_OF_LONG_INT);
	//флаг наличия в векторе x хотя бы одного ненулевого бита
//	set_given_bit_to_position(&p, pos && 1, sh);
	//if(n >= 32)
#ifdef bbbb
	 //    long_to_binary(p0,str0);
	long_to_binary(p, str);
	printf("get_bit_position x %25llu n %3d sh %2d pos %3d pos && 1 %3d p0 %25llu p %25llu %s \n", x, n, sh, pos, pos && 1, p0, p, str);
#endif
	//возвращаем часть элемента левого массива, сооотвествующую одному элементу правого массива
	// (часть, потому что весь 64-разрядный элемент левого, укороченного массива должен содержать информацию о )
	// 64-х соседних элементах правого массива
	return p;
}

//возвращает элемент массива, если нет выход за границу
__device__ unsigned long long int get_array(unsigned long long  int* x, int n, int size)
{
	//   if(n >= 32) printf("n %d size %d (n < size) %d reurn %llu \n",n,size,(n < size),((n < size) ? x[n] : 0));
	return ((n < size) ? x[n] : 0);
}


void __global__ find(unsigned long long  int* x, unsigned long long  int* new_x, unsigned int N)
{
	unsigned int n = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
	int pos, sh, p;
	int NNN;
	cuPrintf("in find %lx \n",*x);
	NNN = blockDim.x;
	tmp[threadIdx.x] = get_bit_position(get_array(x, n, N), n);
	cuPrintf("tmp[threadIdx.x] %lx \n", tmp[threadIdx.x]);
	return;
}


__global__ void addKernel()
{
   
    cuPrintf("add kernel\n");
}



int main()
{
	unsigned long long * x, * new_x, n = 1,h_v = 61680;

	cudaMalloc(&x, sizeof(unsigned long long));
	cudaMemcpy(x, &h_v, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	cudaMalloc(&new_x, sizeof(unsigned long long));
    cudaPrintfInit();

    addKernel << <1, 10 >> > ();
	find << <1, 1 >> > (x,new_x,n);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    return 0;
}