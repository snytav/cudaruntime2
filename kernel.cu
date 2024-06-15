
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned long long int* LongPointer;

//#include "find.h"
#define SIZE_OF_LONG_INT 64

#include <stdio.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#define LEVELS 10
#define OPT_THREADS 128
#define OPT_REDUCE

//actual length of a column (in bits)
//M*LENGTH1=2^25 <-ОГРАНИЧЕНИЕ ПО ПАМЯТИ
// M>20 ТОЛЬКО ЧАСТЬ ТАБЛИЦЫ, ОСТАЛЬНОЕ ПЕРЕБЕРАЕТСЯ ДОБАВЛЕНИЕМ СТРОКИ
#define LENGTH1 (4*64)


//#define fff
//#define QQQ
const int NN2 = (LENGTH1 - 1) / 64 + 1;
int threads2 = (OPT_THREADS < NN2) ? OPT_THREADS : NN2;
int blocks2 = (NN2 - 1) / threads2 + 1;

#define N1 (((LENGTH1 % SIZE_OF_LONG_INT) ==0) ?  (LENGTH1/SIZE_OF_LONG_INT): (LENGTH1/SIZE_OF_LONG_INT+1))


//    unsigned long long  int h_v[N],
unsigned long long int* h_new_v;
LongPointer  d_vfrst[LEVELS], d_vnumb[LEVELS];
char** tb;


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
	
	tmp[threadIdx.x] = gbp;
	cuPrintf("tmp at end %lx tmp %lx %lx %lx %lx \n", tmp[threadIdx.x],
		                   tmp[0],tmp[1],tmp[2],tmp[3]);
	//return;
	pos = __ffsll(x[n]);

	cuPrintf("pos in find %d \n", pos);

	//     num[n] = pos;

	//return;
	unsigned int n_minor;
	n_minor = n / SIZE_OF_LONG_INT; // n_minor это позиция 64-битной последовательности в векторе результата, в левом массиве

	sh = n % SIZE_OF_LONG_INT; //номер бита в отдельном элементе 64-битной послеждовательности
	p = (pos && 1) << sh;
	cuPrintf(  "threadIdx.x %d n %d n_minor %d size %d pos %d sh %d p %d pf %d new_xb %lx pos \n", threadIdx.x,
		n, n_minor,
		SIZE_OF_LONG_INT,
		pos,
		sh,
		p, get_bit_position(x[n], n),
		new_x[n_minor] 
	           ); 
	//,num[n]);
}

__global__ void copy_block(unsigned long long int* dv, unsigned long long int* dv0)
{
	__syncthreads();
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < NN2)dv[tid] = dv0[tid];
}


__global__ void first_backward(LongPointer* d_v, int* d_first_non_zero, int level)
{
	int f[LEVELS];
	unsigned long long int* dvl, u;
	char lprt[100];

	//	printf("inverse level %d \n",level);


	f[level + 1] = 1;
	while (level >= 0)
	{
		dvl = d_v[level];
		int index1 = f[level + 1] - 1;// + (f[level+1]-1)*SIZE_OF_LONG_INT;
		u = dvl[index1];
#ifdef fff
		long_to_binary(u, lprt, LENGTH1);
		printf("element number %d at level %d %llu %s (numbers in array from 0, positions in bit sequence from 1)\n",
			index, level + 1, u, lprt);
#endif
		f[level] = __ffsll(u) + index1 * SIZE_OF_LONG_INT;
#ifdef fff
		printf("level %d u %llu %s f[level] %d\n", level, u, lprt, f[level]);
#endif
		//        if(level == 0)return;
				//printf("level %d f %d f[+1] %d\n",level,f[level],f[level+1]);
		level--;
	}
	*d_first_non_zero = f[0];// + (f[1]-1)*SIZE_OF_LONG_INT;
	if (*d_first_non_zero > LENGTH1) *d_first_non_zero = 0;
#ifdef ffff
	printf("d_first_non_zero %d  pointer= %p\n", *d_first_non_zero, d_first_non_zero);
#endif
}

void reduce_array(unsigned long long  int* d_v1, unsigned long long  int* d_v, unsigned int size, unsigned int level, unsigned int N)
{
	//	char s1[1000],s2[1000];
	unsigned long long int h_new_v[N1], h_v[N1];
	cudaError_t err1;//,err0;

	cudaError_t err = cudaGetLastError();
	//	printf("errors at enter reduce_array %d\n",err);

	unsigned int blocks, threads = (size < SIZE_OF_LONG_INT) ? size : SIZE_OF_LONG_INT;

	cudaError_t err0 = cudaMemcpy(h_v, d_v, sizeof(unsigned long long  int) * size, cudaMemcpyDeviceToHost);
	//	printf("size %d err %d %s  %p\n",size,err0,cudaGetErrorString(err0),d_v);

	//		printf("size1 %d \n",size);

	blocks = (int)ceil(((double)size) / threads);
	printf("reduce_array#####  size %d blocks %d threads %d h_v[0] %llx h_v[1] %llx d\n",
		size, blocks, threads, h_v[0], h_v[1]);

	//cudaPrintfInit();
//	test <<< 1, 1 >>> ();
	//find<<<blocks,threads>>>(d_v,d_v1,size);
	//cudaPrintfDisplay(stdout, true);
//	cudaPrintfEnd();

	//cudaDeviceSynchronize();

	err1 = cudaGetLastError();

	if (err1 != cudaSuccess)
	{
		//#ifdef frst
		printf("kernel error %d %s size %d\n", err1, cudaGetErrorString(err1), size);
		//#endif
		exit(0);
	}
#ifdef frst
	err0 = cudaMemcpy(h_new_v, d_v1, sizeof(unsigned long long  int) * size, cudaMemcpyDeviceToHost);
	if (err0 != cudaSuccess)
	{

		printf("D2H error0 %d %s\n", err0, cudaGetErrorString(err0));
		exit(0);
	}
	err1 = cudaMemcpy(h_v, d_v, sizeof(unsigned long long  int) * size, cudaMemcpyDeviceToHost);
	//printf("h_new0 %ul\n",h_new_v[0]);
	//err = cudaMemcpy(res,d_res,sizeof(int)*size,cudaMemcpyDeviceToHost);
		//printf("h_new0 %ul\n",h_new_v[0]);
	//err1 = cudaMemcpy(h_v,d_v,sizeof(unsigned long long  int)*size,cudaMemcpyDeviceToHost);


	printf("D2H error %d %s\n", err1, cudaGetErrorString(err1));
	FILE* f_res;
	char fname[100];

	sprintf(fname, "result%02d.dat", level);
	if ((f_res = fopen(fname, "wt")) == NULL) return 0;
	for (int i = 0; i < size; i++)
	{
		long_to_binary(h_v[i], s1);
		long_to_binary(h_new_v[i], s2);
		//printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
		fprintf(f_res, "i %3d %s,%25llu result_vector %d init %s \n", i, s2, h_new_v[i], get_position_bit(h_new_v, i), s1);
		// printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
	}
	fclose(f_res);
#endif
}



int first(unsigned long long int* dv0, int size, int* d_first_non_zero, unsigned int N)
{
	static int frst = 1;
	static LongPointer* dev_d_v;
	int big_n = size, level = 0, n = 1;


	cudaError_t err = cudaGetLastError(), err_m, err_c;
#ifdef QQQ
	char str[100];
	print_device_bit_row("first0", dv0, big_n * SIZE_OF_LONG_INT, 0, N);
#endif
	//    cudaMemcpy(d_v[0],dv0,N*sizeof(unsigned long long  int),cudaMemcpyDeviceToDevice); //must be!!!
	copy_block << <1, N >> > (d_vfrst[0], dv0);
	cudaDeviceSynchronize();
#ifdef QQQ
	print_device_bit_column("first1", dv0, big_n * SIZE_OF_LONG_INT, N);
	printf("errors at enter first %d\n", err);
	printf("START n %3d big_n %3d level %d \n ", n, big_n, level);
#endif
	for (big_n = size; big_n > 1; big_n = (int)ceil((double)big_n / (double)SIZE_OF_LONG_INT))
	{
		n = (int)ceil((double)big_n / (double)SIZE_OF_LONG_INT);
#ifdef QQQ
		printf("n %3d big_n %3d level %d \n ", n, big_n, level);
		cudaError_t err = cudaGetLastError();
		printf("errors before reduce %d\n", err);

		sprintf(str, "level%02d", level);
		print_device_bit_column(str, dv[level], big_n * SIZE_OF_LONG_INT, N);
#endif
		reduce_array(d_vfrst[level + 1], d_vfrst[level], big_n, level, N);
#ifdef QQQ
		sprintf(str, "level%02d_result", level);
		print_device_bit_column(str, dv[level + 1], big_n, N);
		err = cudaGetLastError();
		printf("errors at after reduce %d\n", err);
#endif
		level++;

	}
	// printf("FND: level=%i \t",level);
	if (frst == 1)
	{
		err_m = cudaMalloc(&dev_d_v, sizeof(LongPointer) * LEVELS);
		err_c = cudaMemcpy(dev_d_v, d_vfrst, sizeof(LongPointer) * LEVELS, cudaMemcpyHostToDevice);
		frst = 0;
	}

#ifdef ffff
	printf("malloc %d copy %d\n", err_m, err_c);
#endif
	err = cudaGetLastError();
	//	   	printf("errors at before inverse %d %s\n",err,cudaGetErrorString(err));
									//TODO: make a device copy of the d_v array and set it as 1st parameter of first_backward
	//        puts("INVERSE");
	first_backward << <1, 1 >> > (dev_d_v, d_first_non_zero, level);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	//	    	       	    	printf("errors at after inverse %d %s\n",err,cudaGetErrorString(err));
	//	    while(level >= 0)
	//	    {
	//	    	int h_first_non_zero;
	//	    	cudaMemcpy(&h_first_non_zero,d_first_non_zero,sizeof(int),cudaMemcpyDeviceToHost);
	//	    	printf("n %3d level %d first non-zero %5d \n",n,level,h_first_non_zero);
	//	        first_non_zero<<<1,1>>>(d_v[level],d_first_non_zero,n,d_first_non_zero);
	//
	//
	//	        cudaMemcpy(&h_first_non_zero,d_first_non_zero,sizeof(int),cudaMemcpyDeviceToHost);
	//	        printf("n %3d level %d first non-zero %5d \n",n,level,h_first_non_zero);
	//
	//	        n *= SIZE_OF_LONG_INT;
	//	        level--;
	//	    }

	return 0;
}



int main()
{
	unsigned long long* x, * new_x, n = 4,h_new_x,

		h_v[] = { 0,0xABCDABCDABCD0000, 0x0F08000800080008, 61568 };

	cudaMalloc(&x, n*sizeof(unsigned long long));
	cudaMemcpy(x, &h_v, n*sizeof(unsigned long long), cudaMemcpyHostToDevice);

	cudaMalloc(&new_x, sizeof(unsigned long long));
    cudaPrintfInit();

    //addKernel << <1, 10 >> > ();
	find << <1, 4 >> > (x,new_x,n);
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
	
	cudaMemcpy(&h_new_x, new_x, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("find result %\lx \n",h_new_x);
    return 0;
}