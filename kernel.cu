
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "find.h"
#define SIZE_OF_LONG_INT 64

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

// обрезка незначащих элементов у последнего слова
//__host__ 
__device__  int position_in_64bit_word(int num, int div)
{
	int res = num % div, t;

	if (num == 0) return 0;

	t = (res > 0) ? (num % div) : div;
	cuPrintf("64bit_word shift num %d div %d res %d t %d\n ",num,div,res,t);

	return  t;
}

__host__ __device__ void long_to_binary(unsigned long long  int x, char* b, unsigned int leng)
{
	//static char b[500];
	int s, lz;
	char bit;
	//b[SIZE_OF_LONG_INT] = '\0';
   // printf("\n %25llu \n",x);
	unsigned long long int z;

	s = SIZE_OF_LONG_INT - 1;//leng-1;
	z = 1;
	z <<= s;
	for (; z > 0; z >>= 1)
	{
		//	printf("z %llu log %d\n",z,(int)(log(z)/log(2.0)));
		   // strcat(b, ((x & z) == z) ? "1" : "0");
		lz = (int)(log((double)z) / log(2.0));
		bit = (((x & z) == z) ? '1' : '0');
		b[lz] = bit;
		//printf("%10llu %s \n",z,b);
	}
	b[s + 1] = 0;
	//    puts("long_to_binary");
	//   puts(b);
	  /*  for(int i = 0;i < s/2;i++)
		{
			char tmp;

			tmp = b[i];
			b[i] = b[s - i];
			b[s - i] = tmp;
		}*/

	int term = (leng < SIZE_OF_LONG_INT) ? leng : SIZE_OF_LONG_INT;
	b[term] = 0;

	//  return b;
}

//__host__ 
__device__ void set_given_bit_to_position(unsigned long long * x, int bit, int pos)
{
	unsigned long long int one = bit;
	cuPrintf("set_given_bit_to_position x %lx bit %d pos %d\n",*x,bit,pos);

	if (pos == 0) return;

	*x = one << (pos - 1);
	cuPrintf("set_given_bit_to_position x %lx bit %d pos %d\n", *x, bit, pos);
}



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
	//cuPrintf("__ffsll %d \n",pos);
	//return pos;


	cuPrintf("get_bit_position pos %d x %lx n %d \n",pos,x,n);

	//элементы массива нумеруются с нуля, биты с единицы
	//позиция единицы(или не единицы), говорящей о наличии единицы в этом слове 
	//в итоговом 64-битном слове, представляющем собой сводку по всему массиву
	sh = position_in_64bit_word(n + 1, SIZE_OF_LONG_INT);
	cuPrintf("sh %d\n",sh);
	//флаг наличия в векторе x хотя бы одного ненулевого бита
	set_given_bit_to_position(&p, pos && 1, sh);
	cuPrintf("p %lx pos %d pos &&1 %d,sh %d\n",
		      p,    pos,   pos &&1,   sh);
	//if(n >= 32)
#ifdef bbbb
	 //    long_to_binary(p0,str0);
	long_to_binary(p, str);
	//cuPrintf("get_bit_position x %25llu n %3d sh %2d pos %3d pos && 1 %3d p0 %25llu p %25llu %s \n", 
		                       x,       n,    sh,    pos,    pos && 1, p0, p, str);
#endif
	//возвращаем часть элемента левого массива, сооотвествующую одному элементу правого массива
	// (часть, потому что весь 64-разрядный элемент левого, укороченного массива должен содержать информацию о )
	// 64-х соседних элементах правого массива
	cuPrintf("get_bit_position returns %d \n",p);
	return p;
}

//возвращает элемент массива, если нет выход за границу
__device__ unsigned long long get_array(unsigned long long  int* x, int n, int size)
{
	cuPrintf("n %d size %d (n < size) %d return %lx \n",n,size,(n < size),((n < size) ? x[n] : 0));
	cuPrintf("get_array 0 %lx 1 %lx 2 %lx \n", x[0], x[1], x[2]);
	return ((n < size) ? x[n] : 0);
}


void __global__ find(unsigned long long* x, unsigned long long  int* new_x, unsigned int N)
{
	unsigned int n = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
	unsigned long long gbp;
	int pos, sh, p;
	int NNN;
	cuPrintf("in find %lx n %u get_array %lx \n",*x,n, get_array(x, n, N));
	NNN = blockDim.x;
	//cuPrintf("get_bit_position");
	gbp = get_bit_position(get_array(x, n, N), n);
	cuPrintf("get_bit_position in find result  %lx \n",gbp);
	cuPrintf("tmp before % tmp %lx %lx %lx %lx \n",
		tmp[0], tmp[1], tmp[2], tmp[3]);

	tmp[threadIdx.x] = gbp;
	cuPrintf("tmp at end %lx tmp %lx %lx %lx %lx \n", tmp[threadIdx.x],
		                   tmp[0],tmp[1],tmp[2],tmp[3]);
	//return;
	pos = __ffsll(x[n]);

	cuPrintf("pos in find %d \n", pos);

	//     num[n] = pos;

	return;
	unsigned int n_minor;
	n_minor = n / SIZE_OF_LONG_INT; // n_minor это позиция 64-битной последовательности в векторе результата, в левом массиве

	sh = n % SIZE_OF_LONG_INT; //номер бита в отдельном элементе 64-битной послеждовательности
	p = (pos && 1) << sh;
	cuPrintf(  "threadIdx.x %d n %d n_minor %d size %d pos %d sh %d p %d pf %d new_xb %llu pos \n", threadIdx.x,
		n, n_minor,
		SIZE_OF_LONG_INT,
		pos,
		sh,
		p, get_bit_position(x[n], n),
		new_x[n_minor] 
	           ); 
	//,num[n]);
}


__global__ void addKernel()
{
   
    cuPrintf("add kernel\n");
}



int main()
{
	unsigned long long* x, * new_x, n = 4,

		h_v[] = { 0,0xABCDABCDABCD0000, 0x0F08000800080008, 61568 };

	cudaMalloc(&x, n*sizeof(unsigned long long));
	cudaMemcpy(x, &h_v, n*sizeof(unsigned long long), cudaMemcpyHostToDevice);

	cudaMalloc(&new_x, sizeof(unsigned long long));
    cudaPrintfInit();

    //addKernel << <1, 10 >> > ();
	find << <1, 4 >> > (x,new_x,n);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    return 0;
}