
//#include "find.h"
#include "param.h"
//#include "star_kernel.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>     /* strcat */



#define LEVELS 10
#define OPT_THREADS 128
#define OPT_REDUCE 16
//#define fff
//#define QQQ
        const int NN2=(LENGTH1-1)/64+1;
        int threads2=(OPT_THREADS<NN2)?OPT_THREADS:NN2;
		int blocks2=(NN2-1)/threads2+1;

//    unsigned long long  int h_v[N],
     unsigned long long int *h_new_v;
    LongPointer  d_vfrst[LEVELS],d_vnumb[LEVELS];
    char ** tb;

//    int i,sh,*d_res,res[N],r_step;


__host__ __device__  int position_in_64bit_word(int num,int div)
{
	int res = num%div,t;

	if(num == 0) return 0;

	t = (res > 0) ? (num%div) : div;
	//printf("shift num %d div %d res %d t %d\n ",num,div,res,t);

	return  t;
}

__host__ __device__  int get_64bit_word(int num,int div)
{
	int res = num%div,t;

	if(num == 0) return 0;

	t = (res > 0) ? num/div : num/div-1;
	//printf("get64 num %d div %d res %d t %d\n ",num,div,res,t);

	return  t;
}
__global__ void set_kernel(unsigned long long int *dst,unsigned long long int *src)
{
#ifdef ssss
	printf("set_kernel %u %llu %llu \n",blockIdx.x,dst[blockIdx.x],src[blockIdx.x]);
#endif
	int ind=threadIdx.x + blockIdx.x*blockDim.x;
    dst[ind] = src[ind];
// printf("Get_Col[%i] %llu %llu \n ",ind,dst[ind],src[ind]);

#ifdef ssss
	printf("set_kernel %u %llu %llu \n",blockIdx.x,dst[blockIdx.x],src[blockIdx.x]);
#endif
}
__host__ __device__ void or_given_bit_to_position(unsigned long long int *x,int bit,int pos)
{
	unsigned long long int one = bit;

	if(pos == 0) return;

	*x |= one << (pos-1);
}

__host__ __device__ void set_given_bit_to_position(unsigned long long int *x,int bit,int pos)
{
	unsigned long long int one = bit;

	if(pos == 0) return;

	*x = one << (pos-1);
}



__host__ __device__ void assign_given_bit_to_position(unsigned long long int *x,int bit,int pos,int op)
{
	unsigned long long int one = 1,bit_pos,zero = 0,res,p;

	if(pos == 0) return;
	p = *x;

	bit_pos = one << (pos-1);

/*	if(pos == 3)
		{
		char b[100],nb[100],bp[100],ress[100];
		int d;
			long_to_binary(p,b,64);
			long_to_binary(bit_pos,bp,64);
			long_to_binary(~bit_pos,nb,64);
			res = bit_pos;
			res |= zero;
			d = p;
			long_to_binary(res,ress,64);
			printf("bit %d bit_pos %s not-bit_pos %s x %d %llu %s res %s \n",bit,bp,nb,d,p,b,ress);
		}
*/
	*x =    (op == OR)  * ( *x | ((one=bit) << (pos-1)) )
		  + (op == SET) * ( (bit == 1) ? (*x | bit_pos) : (*x & (~bit_pos)) );

}

__host__ __device__ void set_bit_to_position(unsigned long long int *x,int pos)
{
	set_given_bit_to_position(x,1,pos);//,SET);
}


__host__ __device__ void long_to_binary(unsigned long long  int x,char *b, unsigned int leng)
{
    //static char b[500];
    int s,lz;
    char bit;
    //b[SIZE_OF_LONG_INT] = '\0';
   // printf("\n %25llu \n",x);
    unsigned long long int z;

    s =  SIZE_OF_LONG_INT-1;//leng-1;
    z = 1;
    z <<= s;
    for (; z > 0; z >>= 1)
    {
    //	printf("z %llu log %d\n",z,(int)(log(z)/log(2.0)));
       // strcat(b, ((x & z) == z) ? "1" : "0");
    	lz = (int)(log((double)z)/log(2.0));
    	bit  = (((x & z) == z) ? '1' : '0');
    	b[lz] = bit;
        //printf("%10llu %s \n",z,b);
    }
    b[s+1] = 0;
//    puts("long_to_binary");
//   puts(b);
  /*  for(int i = 0;i < s/2;i++)
    {
    	char tmp;

    	tmp = b[i];
    	b[i] = b[s - i];
    	b[s - i] = tmp;
    }*/

    int term = (leng <  SIZE_OF_LONG_INT) ? leng :SIZE_OF_LONG_INT;
    b[term] = 0;

  //  return b;
}
__host__ __device__ void long_to_binary1(unsigned long long  int x,char *b, unsigned int leng)
{
    //static char b[500];
    int s,lz;
    char bit;
    //b[SIZE_OF_LONG_INT] = '\0';
   // printf("\n %25llu \n",x);
    unsigned long long int z;

    s = leng-1;
    z = 1;
    z <<= s;
    lz=0;
    for (; z > 0; z >>= 1)
    {
    //	printf("z %llu log %d\n",z,(int)(log(z)/log(2.0)));
       // strcat(b, ((x & z) == z) ? "1" : "0");
 //   lz=(int)(log((double)z)/log(2.0));
    	bit  = (((x & z) == z) ? '1' : '0');
    	b[lz] = bit;
    	lz++;
        //printf("%10llu %s \n",z,b);
    }
    b[leng] = 0;

    int term = (leng <  SIZE_OF_LONG_INT) ? leng :SIZE_OF_LONG_INT;
    b[term] = 0;

  //  return b;
}

//редукция большого массива 64-разрядных целых к массиву размером в 64 раза меньше,
//где каждому целому числу изначального массива соответствует 1 бит, ненулевой если
//в соответсвующем элементе исходного массива был хотя бы один ненулевлй бит
__device__ unsigned long long int get_bit_position(unsigned long long  int x,int n)
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


	     //printf("get_bit_position pos %d x %llu n %d \n",pos,x,n);

	     //элементы массива нумеруются с нуля, биты с единицы
	     sh = position_in_64bit_word(n+1,SIZE_OF_LONG_INT);
	     //флаг наличия в векторе x хотя бы одного ненулевого бита
	     set_given_bit_to_position(&p,pos && 1,sh);
	     //if(n >= 32)
#ifdef bbbb
     //    long_to_binary(p0,str0);
         long_to_binary(p,str);
	     printf("get_bit_position x %25llu n %3d sh %2d pos %3d pos && 1 %3d p0 %25llu p %25llu %s \n",x,n,sh,pos,pos && 1,p0,p,str);
#endif
	     //возвращаем часть элемента левого массива, сооотвествующую одному элементу правого массива
	     // (часть, потому что весь 64-разрядный элемент левого, укороченного массива должен содержать информацию о )
	     // 64-х соседних элементах правого массива
	     return p;
}

//записывает элемент "левого" массива, если нет выхода за границу, и если номер нити кратен 64
__device__ int write_bit(int thr_n,int n,int lhs_size,unsigned long long  int *x)
{
	int permit = thr_n%SIZE_OF_LONG_INT == 0;
	//printf("thrn %d write_bit n %d permit %d returnx %d \n",thr_n,n,permit,(n < lhs_size));
	return (permit ? ((n < lhs_size) ? x[n] : 0 ) :1);
}

//возвращает элемент массива, если нет выход за границу
__device__ unsigned long long int get_array(unsigned long long  int *x,int n,int size)
{
	 //   if(n >= 32) printf("n %d size %d (n < size) %d reurn %llu \n",n,size,(n < size),((n < size) ? x[n] : 0));
        return ((n < size) ? x[n] : 0);
}



//редукция "большого" массива x, до массива new_x, меньшего по размеру в 64 раза,
//где каждому целому числу изначального массива соответствует 1 бит, ненулевой если
//в соответсвующем элементе исходного массива был хотя бы один ненулевой бит

void __global__ find(unsigned long long  int *x,unsigned long long  int *new_x, unsigned int N)
{
	 unsigned int n = threadIdx.x + blockIdx.x*blockDim.x;
	 __shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
	 int pos,sh,p;
	 int NNN;
	 cuPrintf("in find\n");
	 return;
      NNN=blockDim.x;
	 tmp[threadIdx.x] = get_bit_position(get_array(x,n,N),n);


     pos = __ffsll(x[n]);

     printf("pos %d \n",pos);

//     num[n] = pos;

     return;
   	 unsigned int n_minor;
    n_minor = n/SIZE_OF_LONG_INT; // n_minor это позиция 64-битной последовательности в векторе результата, в левом массиве

     sh = n%SIZE_OF_LONG_INT; //номер бита в отдельном элементе 64-битной послеждовательности
     p =  (pos && 1) << sh;
	 printf("threadIdx.x %d n %d n_minor %d size %d pos %d sh %d p %d pf %d new_xb %llu pos \n", threadIdx.x,
		 n, n_minor,
		 SIZE_OF_LONG_INT,
		 pos,
		 sh,
		 p, get_bit_position(x[n], n),
		 new_x[n_minor]);//,num[n]);

//TODO: change "n" for work with further parts of array, n= tthreadIdx.x*Size_long
	      n= threadIdx.x*SIZE_OF_LONG_INT;

     if(threadIdx.x <= N/SIZE_OF_LONG_INT+1)
     {
         for(int i = 0; i < N;i++)
         {
        	 printf("array i %d %llu direct-x %llu \n",i,get_array(x,i,N),x[i]);
         }
         // в каждый элемент нового, укороченного массива пишут 64 нити, каждая из которых обрабатывает 64
    	 // 64-разрядных числа, начиная с n
	 __syncthreads();
	 if (threadIdx.x==0)
      new_x[blockIdx.x] =  get_array(tmp,0,NNN)
    	    	    	|  get_array(tmp,1,NNN)
    	    	    	|  get_array(tmp,2,NNN)
    	    	    	|  get_array(tmp,3,NNN)
    	    	        |  get_array(tmp,4,NNN)
    	    	        |  get_array(tmp,5,NNN)
    	    	        |  get_array(tmp,6,NNN)
    			        |  get_array(tmp,7,NNN)
    	         		|  get_array(tmp,8,NNN)
    			        |  get_array(tmp,9,NNN)
    	                |  get_array(tmp,10,NNN)
    	                |  get_array(tmp,11,NNN)
    	                |  get_array(tmp,12,NNN)
	                    |  get_array(tmp,13,NNN)
	                    |  get_array(tmp,14,NNN)
	                    |  get_array(tmp,15,NNN)
                        |  get_array(tmp,16,NNN)
                        |  get_array(tmp,17,NNN)
                        |  get_array(tmp,18,NNN)
            			|  get_array(tmp,19,NNN)
			            |  get_array(tmp,20,NNN)
		             	|  get_array(tmp,21,NNN)
                        |  get_array(tmp,22,NNN)
                        |  get_array(tmp,23,NNN)
                        |  get_array(tmp,24,NNN)
     	                |  get_array(tmp,25,NNN)
	                    |  get_array(tmp,26,NNN)
	                    |  get_array(tmp,27,NNN)
                        |  get_array(tmp,28,NNN)
                        |  get_array(tmp,29,NNN)
                        |  get_array(tmp,30,NNN)
	                    |  get_array(tmp,31,NNN)
		                |  get_array(tmp,32,NNN)
		                |  get_array(tmp,33,NNN)
                        |  get_array(tmp,34,NNN)
                        |  get_array(tmp,35,NNN)
                        |  get_array(tmp,36,NNN)
     	                |  get_array(tmp,37,NNN)
	                    |  get_array(tmp,38,NNN)
	                    |  get_array(tmp,39,NNN)
                        |  get_array(tmp,40,NNN)
                        |  get_array(tmp,41,NNN)
                        |  get_array(tmp,42,NNN)
			            |  get_array(tmp,43,NNN)
	            		|  get_array(tmp,44,NNN)
	             		|  get_array(tmp,45,NNN)
                        |  get_array(tmp,46,NNN)
                        |  get_array(tmp,47,NNN)
                        |  get_array(tmp,48,NNN)
      	                |  get_array(tmp,49,NNN)
	                    |  get_array(tmp,50,NNN)
	                    |  get_array(tmp,51,NNN)
                        |  get_array(tmp,52,NNN)
                        |  get_array(tmp,53,NNN)
                        |  get_array(tmp,54,NNN)
		                |  get_array(tmp,55,NNN)
	         	        |  get_array(tmp,56,NNN)
		                |  get_array(tmp,57,NNN)
                        |  get_array(tmp,58,NNN)
                        |  get_array(tmp,59,NNN)
                        |  get_array(tmp,60,NNN)
                        |  get_array(tmp,61,NNN)
                        |  get_array(tmp,62,NNN)
                        |  get_array(tmp,63,NNN);

//         		           get_bit_position(get_array(x,n+1,N),n+1) |
//		                   get_bit_position(get_array(x,n+2,N),n+2) |
//		                   get_bit_position(get_array(x,n+3,N),n+3) |
//		                   get_bit_position(get_array(x,n+4,N),n+4) |
//		                   get_bit_position(get_array(x,n+5,N),n+5) |
//		                   get_bit_position(get_array(x,n+6,N),n+6) |
//		                   get_bit_position(get_array(x,n+7,N),n+7) |
//		                   get_bit_position(get_array(x,n+8,N),n+8) |
//		                   get_bit_position(get_array(x,n+9,N),n+9) |
//		                   get_bit_position(get_array(x,n+10,N),n+10) |
//		                   get_bit_position(get_array(x,n+11,N),n+11) |
//		                   get_bit_position(get_array(x,n+12,N),n+12) |
//		                   get_bit_position(get_array(x,n+13,N),n+13) |
//		                   get_bit_position(get_array(x,n+14,N),n+14) |
//		                   get_bit_position(get_array(x,n+15,N),n+15) |
//		                   get_bit_position(get_array(x,n+16,N),n+16) |
//		                   get_bit_position(get_array(x,n+17,N),n+17) |
//		                   get_bit_position(get_array(x,n+18,N),n+18) |
//		                   get_bit_position(get_array(x,n+19,N),n+19) |
//		                   get_bit_position(get_array(x,n+20,N),n+20) |
//		                   get_bit_position(get_array(x,n+21,N),n+21) |
//		                   get_bit_position(get_array(x,n+22,N),n+22) |
//		                   get_bit_position(get_array(x,n+23,N),n+23) |
//		                   get_bit_position(get_array(x,n+24,N),n+24) |
//		                   get_bit_position(get_array(x,n+25,N),n+25) |
//		                   get_bit_position(get_array(x,n+26,N),n+26) |
//		                   get_bit_position(get_array(x,n+27,N),n+27) |
//		                   get_bit_position(get_array(x,n+28,N),n+28) |
//		                   get_bit_position(get_array(x,n+29,N),n+29) |
//		                   get_bit_position(get_array(x,n+30,N),n+30) |
//		                   get_bit_position(get_array(x,n+31,N),n+31) |
//		                   get_bit_position(get_array(x,n+32,N),n+32) |
//		                   get_bit_position(get_array(x,n+33,N),n+33) |
//		                   get_bit_position(get_array(x,n+34,N),n+34) |
//		                   get_bit_position(get_array(x,n+35,N),n+35) |
//		                   get_bit_position(get_array(x,n+36,N),n+36) |
//		                   get_bit_position(get_array(x,n+37,N),n+37) |
//		                   get_bit_position(get_array(x,n+38,N),n+38) |
//		                   get_bit_position(get_array(x,n+39,N),n+39) |
//		                   get_bit_position(get_array(x,n+40,N),n+40) |
//		                   get_bit_position(get_array(x,n+41,N),n+41) |
//		                   get_bit_position(get_array(x,n+42,N),n+42) |
//		                   get_bit_position(get_array(x,n+43,N),n+43) |
//		                   get_bit_position(get_array(x,n+44,N),n+44) |
//		                   get_bit_position(get_array(x,n+45,N),n+45) |
//		                   get_bit_position(get_array(x,n+46,N),n+46) |
//		                   get_bit_position(get_array(x,n+47,N),n+47) |
//		                   get_bit_position(get_array(x,n+48,N),n+48) |
//		                   get_bit_position(get_array(x,n+49,N),n+49) |
//		                   get_bit_position(get_array(x,n+50,N),n+50) |
//		                   get_bit_position(get_array(x,n+51,N),n+51) |
//		                   get_bit_position(get_array(x,n+52,N),n+52) |
//		                   get_bit_position(get_array(x,n+53,N),n+53) |
//		                   get_bit_position(get_array(x,n+54,N),n+54) |
//		                   get_bit_position(get_array(x,n+55,N),n+55) |
//		                   get_bit_position(get_array(x,n+56,N),n+56) |
//		                   get_bit_position(get_array(x,n+57,N),n+57) |
//		                   get_bit_position(get_array(x,n+58,N),n+58) |
//		                   get_bit_position(get_array(x,n+59,N),n+59) |
//		                   get_bit_position(get_array(x,n+60,N),n+60) |
//		                   get_bit_position(get_array(x,n+61,N),n+61) |
//		                   get_bit_position(get_array(x,n+62,N),n+62) |
//		                   get_bit_position(get_array(x,n+63,N),n+63) |
//		                   get_bit_position(get_array(x,n+64,N),n+64) |
//                           0
//    	                  ); // |
		                   // get_bit_position(x[n+4],n+4);

     if(threadIdx.x==0)    printf("new_xa %llu n_minor %d \n",new_x[blockIdx.x],blockIdx.x);
     }
}

__host__ __device__ int get_position_bit(unsigned long long int *h,int n)
{
	int num,sh;
	unsigned long long int p = 1;

	num = get_64bit_word(n,SIZE_OF_LONG_INT);
	sh =  position_in_64bit_word(n,SIZE_OF_LONG_INT);


	set_bit_to_position(&p,sh);
//	p = p << sh;
#ifdef bbb
	printf("get_positio_bit n %d num %d sh %d shifted p %llu \n",n,num,h[num],p);
#endif

	return (h[num] & p ) && 1;
}

__global__ void copy_block(unsigned long long int *dv,unsigned long long int *dv0)
{	    __syncthreads();
	int tid=threadIdx.x+ blockIdx.x*blockDim.x;
    if (tid<NN2)dv[tid]=dv0[tid];
}
__global__ void copy_block1(unsigned long long int *dv,unsigned long long int *dv0)
{   int k,tid=threadIdx.x + blockIdx.x*blockDim.x;
    unsigned long long int zero=1;
    if(tid<NN2)  dv[tid]=dv0[tid];
//    else dv[tid]=0;
    if (tid==(NN2-1)) // in the last element need to zero the tail
    {
    	/*zero=(1<<(num % SIZE_OF_LONG_INT)-1)-1;
	  zero=~zero;*/
    	k=(LENGTH1%SIZE_OF_LONG_INT);
    	zero=(zero<<k)-1;
//    	printf("k=%i  %llu \n",k, zero);
    	if (k!=0)
    		dv[tid]=dv0[tid]&zero;
    	else dv[tid]=dv0[tid];
    }


//	printf("numb_%i %i\n",tid,__popcll(dv0[tid]));
}
void reduce_array(unsigned long long  int *d_v1,unsigned long long  int*d_v,unsigned int size,unsigned int level, unsigned int N)
{
//	char s1[1000],s2[1000];
	unsigned long long int h_new_v[N1],h_v[N1];
	cudaError_t err1;//,err0;

	cudaError_t err = cudaGetLastError();
//	printf("errors at enter reduce_array %d\n",err);

	unsigned int blocks, threads = (size < SIZE_OF_LONG_INT) ? size : SIZE_OF_LONG_INT;

	cudaError_t err0 = cudaMemcpy(h_v,d_v,sizeof(unsigned long long  int)*size,cudaMemcpyDeviceToHost);
//	printf("size %d err %d %s  %p\n",size,err0,cudaGetErrorString(err0),d_v);

//		printf("size1 %d \n",size);

	blocks = (int)ceil( ((double)size)/threads);
    printf("reduce_array#####  size %d blocks %d threads %d h_v[0] %llx h_v[1] %llx d\n",
		                       size,   blocks,   threads,   h_v[0], h_v[1]);

	//cudaPrintfInit();
//	test <<< 1, 1 >>> ();
    //find<<<blocks,threads>>>(d_v,d_v1,size);
	//cudaPrintfDisplay(stdout, true);
//	cudaPrintfEnd();

    //cudaDeviceSynchronize();

    err1 = cudaGetLastError();

    if(err1 != cudaSuccess)
    	{
//#ifdef frst
    	printf("kernel error %d %s size %d\n",err1,cudaGetErrorString(err1),size);
//#endif
    		exit(0);
    	}
#ifdef frst
	err0 = cudaMemcpy(h_new_v,d_v1,sizeof(unsigned long long  int)*size,cudaMemcpyDeviceToHost);
	if(err0 != cudaSuccess)
	{

		printf("D2H error0 %d %s\n",err0,cudaGetErrorString(err0));
		exit(0);
	}
	err1 = cudaMemcpy(h_v,d_v,sizeof(unsigned long long  int)*size,cudaMemcpyDeviceToHost);
	//printf("h_new0 %ul\n",h_new_v[0]);
	//err = cudaMemcpy(res,d_res,sizeof(int)*size,cudaMemcpyDeviceToHost);
        //printf("h_new0 %ul\n",h_new_v[0]);
	//err1 = cudaMemcpy(h_v,d_v,sizeof(unsigned long long  int)*size,cudaMemcpyDeviceToHost);


	printf("D2H error %d %s\n",err1,cudaGetErrorString(err1));
	FILE *f_res;
    char fname[100];

	sprintf(fname,"result%02d.dat",level);
	if( (f_res =fopen(fname,"wt")) == NULL ) return 0;
	for(int i = 0;i < size;i++)
	{
		   long_to_binary(h_v[i],s1);
		   long_to_binary(h_new_v[i],s2);
		   //printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
	       fprintf(f_res,"i %3d %s,%25llu result_vector %d init %s \n",i,s2,h_new_v[i],get_position_bit(h_new_v,i),s1);
	      // printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
	}
	fclose(f_res);
#endif
}

__global__ void first_non_zero(unsigned long long int *d_v,int *n,int size,int *d_first_non_zero)
{


	    if(*n == -1)
	    {
	    	*d_first_non_zero = 0;
	    	return;
	    }
//TODO: 1. needed to define position in the whole initial bit sequence, not only in one array element
//	    2. 0th bit of the second array element must be somehow 65  !!!!
//solution: make a "kosher" % function
//and a "kosher" set-to-position function
	    int nz = ((size == 1) ? __ffsll(d_v[0]) : (__ffsll(d_v[*n-1]) + (*n-1)*size) );
	    (*d_first_non_zero) = nz;
	    printf("first_non_zero n %d size %d nz %d  ffsll %d to-add %d\n",*n,size,nz,__ffsll(d_v[*n-1]),(*n-1)*size);
}

__global__ void first_backward(LongPointer *d_v,int *d_first_non_zero,int level)
{
	int f[LEVELS];
	unsigned long long int *dvl,u;
	char lprt[100];

//	printf("inverse level %d \n",level);


	f[level+1] = 1;
	while(level >= 0)
	{
		dvl = d_v[level];
		int index1 = f[level+1]-1;// + (f[level+1]-1)*SIZE_OF_LONG_INT;
        u = dvl[index1];
#ifdef fff
        long_to_binary(u,lprt,LENGTH1);
		printf("element number %d at level %d %llu %s (numbers in array from 0, positions in bit sequence from 1)\n",
				index,level+1,u,lprt);
#endif
	    f[level] = __ffsll(u) + index1*SIZE_OF_LONG_INT;
#ifdef fff
        printf("level %d u %llu %s f[level] %d\n",level,u,lprt,f[level]);
#endif
//        if(level == 0)return;
		//printf("level %d f %d f[+1] %d\n",level,f[level],f[level+1]);
		level--;
	}
	*d_first_non_zero = f[0];// + (f[1]-1)*SIZE_OF_LONG_INT;
	if (*d_first_non_zero>LENGTH1) *d_first_non_zero=0;
#ifdef ffff
	printf("d_first_non_zero %d  pointer= %p\n",*d_first_non_zero,d_first_non_zero);
#endif
}

int first(unsigned long long int *dv0,int size,int *d_first_non_zero, unsigned int N)
{
	    static int frst=1;
	    static LongPointer *dev_d_v;
	    int big_n = size,level = 0,n=1;


	    cudaError_t err = cudaGetLastError(),err_m,err_c;
#ifdef QQQ
	    char str[100];
	    print_device_bit_row("first0",dv0,big_n*SIZE_OF_LONG_INT,0,N);
#endif
	//    cudaMemcpy(d_v[0],dv0,N*sizeof(unsigned long long  int),cudaMemcpyDeviceToDevice); //must be!!!
	    copy_block<<<1,N>>>(d_vfrst[0],dv0);
	    cudaDeviceSynchronize();
#ifdef QQQ
        print_device_bit_column("first1",dv0,big_n*SIZE_OF_LONG_INT,N);
	    	printf("errors at enter first %d\n",err);
	    	printf("START n %3d big_n %3d level %d \n ",n,big_n,level);
#endif
	    for(big_n = size; big_n > 1; big_n  = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT))
	    {
	    	n = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT);
#ifdef QQQ
	    	printf("n %3d big_n %3d level %d \n ",n,big_n,level);
	    	cudaError_t err = cudaGetLastError();
	    	printf("errors before reduce %d\n",err);

	    	sprintf(str,"level%02d",level);
	    	print_device_bit_column(str,dv[level],big_n*SIZE_OF_LONG_INT,N);
#endif
	        reduce_array(d_vfrst[level+1],d_vfrst[level],big_n,level,N);
#ifdef QQQ
	    	sprintf(str,"level%02d_result",level);
	    	print_device_bit_column(str,dv[level+1],big_n,N);
	        err = cudaGetLastError();
	       	    	printf("errors at after reduce %d\n",err);
#endif
	        level++;

	    }
// printf("FND: level=%i \t",level);
	    if (frst==1)
	    {
	    	err_m = cudaMalloc(&dev_d_v,sizeof(LongPointer)*LEVELS);
	    	err_c = cudaMemcpy(dev_d_v,d_vfrst,sizeof(LongPointer)*LEVELS,cudaMemcpyHostToDevice);
	    	frst=0;
	    }

#ifdef ffff
        printf("malloc %d copy %d\n",err_m,err_c);
#endif
	    err = cudaGetLastError();
//	   	printf("errors at before inverse %d %s\n",err,cudaGetErrorString(err));
	   	    	       	    	//TODO: make a device copy of the d_v array and set it as 1st parameter of first_backward
//        puts("INVERSE");
	    first_backward<<<1,1>>>(dev_d_v,d_first_non_zero,level);
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
__global__ void some_backward(LongPointer d_v,int *d_first_non_zero)
{
		*d_first_non_zero= (d_v[0]!=0)?1:0;
}
int some(unsigned long long int *dv0,int size,int *d_first_non_zero,unsigned int N)
{
	    static int frst=1;
	    static LongPointer *dev_d_v;
	    int big_n = size,level = 0;//,n=1;
	    cudaError_t err = cudaGetLastError();
	    copy_block<<<1,N>>>(d_vfrst[0],dv0);

	    for(big_n = size; big_n > 1; big_n  = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT))
	    {
//	    	n = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT);
	        reduce_array(d_vfrst[level+1],d_vfrst[level],big_n,level,N);
	        level++;
	    }

	    if (frst==1)
	    {
	    	cudaMalloc(&dev_d_v,sizeof(LongPointer)*LEVELS);
	    	cudaMemcpy(dev_d_v,d_vfrst,sizeof(LongPointer)*LEVELS,cudaMemcpyHostToDevice);
	    	frst=0;
	    }

	    err = cudaGetLastError();
	    some_backward<<<1,1>>>(d_vfrst[level],d_first_non_zero);
	return 0;
}


const unsigned long long int m[6]={0x5555555555555555,
				  0x3333333333333333,
                0x0f0f0f0f0f0f0f0f,
				  0x00ff00ff00ff00ff,
	    		  0x0000ffff0000ffff,
				  0x00000000ffffffff};
unsigned long long int *d_m;

__device__ void numb_shift1(LongPointer d_v,unsigned long long int *m ,int red_numb,int N)
{ int i;
  int index=threadIdx.x + blockIdx.x*blockDim.x;
  unsigned long long int b1,b2;
   if (index<N)
   {  if(d_v[index]>0)
	   for(int j=0; j<3; j++)
	   {
	   i=1<<red_numb;
	   b1=d_v[index]&m[red_numb];
	   b2=(d_v[index]>>i)&m[red_numb];
	   d_v[index]=b1+b2;
	   red_numb++;}
   }
}

__device__ void numb_reduce(LongPointer dv, LongPointer dv1,int red_numb, int N)
{   //blockDim.x=min(OPT_THREADS,2^(2^red_num -1))
	__shared__ unsigned long long int cache[OPT_THREADS];
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int cacheIndex=threadIdx.x;
	int red_count=(red_numb==0)?OPT_REDUCE:min(OPT_THREADS,blockDim.x);
//	printf("red_numb=%i, red_count=%i\n", red_numb, red_count);
	int i=OPT_THREADS;

	if (N>1)
	{
	cache[cacheIndex]=get_array(dv,tid,N);
	while (red_count <= i) i >>= 1;// минимальная степень двойки>blockDim.x
//	if(tid==0) printf("\n red_count=%i, i=%i \n",red_count,i);
	__syncthreads();
	while (i!=0)
	{
//		if (cacheIndex<i)//
		if (cacheIndex%red_count<i)
		{
			cache[cacheIndex]+=get_array(cache,cacheIndex+i,blockDim.x);
//		printf(" %i+%i:%llu  ",cacheIndex,cacheIndex+i,get_array(cache,cacheIndex+i,blockDim.x));
		}
		__syncthreads();
		i/=2;
	}
//	if (cacheIndex==0)//
	if (cacheIndex%red_count==0)
		{
	dv1[tid/red_count]=cache[cacheIndex];
//		dv1[blockIdx.x]=cache[0];
//		printf("red_count=%i   dv1[%i:%i]=%llu   ",red_count,tid/red_count,cacheIndex,cache[cacheIndex]);
		};
	}
	else dv1[tid]=dv[tid];
}
__global__ void numb_shift(LongPointer d_v,LongPointer d_v1,unsigned long long int *d_m ,int red_numb,int N)
{  int i=3*red_numb;
    if(i<6)	numb_shift1(d_v,d_m,i,N);//3 шага
    __syncthreads();
	numb_reduce(d_v,d_v1,i,N); // свертка на 16:128 или копирование
}

__global__ void numb_backward(LongPointer d_v,int *d_first_non_zero)
{
		*d_first_non_zero=d_v[0];
//		if (d_v[0]>0)printf("numb: %i\n",*d_first_non_zero);
}
int number(unsigned long long int *dv0,int size, int *d_numb,unsigned int N)
{
    static int frst=1;
    static LongPointer *dev_d_v;
    int //big_n = size,
    		level = 0;//,n=1;
 //   int i;// , red_count;
    cudaError_t err = cudaGetLastError();
    err=cudaMemset(d_numb,0,sizeof(int));
 //   printf("cudaMemset %d , %s \n",err,cudaGetErrorString(err));
    unsigned int blocks, threads = (size < OPT_THREADS) ? size : OPT_THREADS;
    blocks = (int)ceil( ((double)size)/threads);
    copy_block1<<<blocks,threads>>>(d_vnumb[0],dv0);

//    for(level=0; level < 2;level++)
//    {
 //   	    threads = (size < OPT_THREADS) ? size : OPT_THREADS;
 //   	    blocks = (int)ceil( ((double)size)/threads);
 //       printf("size %d blocks %d threads %d \n",size,blocks, threads);
       numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//3 шага, свертка не более, чем на 16
       if (size>1) size=((size-1)/OPT_REDUCE)+1;
       level++;

                threads = (size < OPT_THREADS) ? size : OPT_THREADS;
          	    blocks = (int)ceil( ((double)size)/threads);
//              printf("size %d blocks %d threads %d \n",size,blocks, threads);
             numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//3 шага, свертка не более чем на 128
             if (size>1) size=((size-1)/OPT_THREADS)+1;
             level++;
//       printf("\n=========================== level=%i, red_count=%i, size=%i \n",level, red_count, big_n);
 //       printf("level=%i, size=%i, red_count=%i \n",level,big_n, red_count);
//    }

//Если необходимо еще сворачивать (N>1048576)
    while (size>1)
    { threads = (size < OPT_THREADS) ? size : OPT_THREADS;
      blocks = (int)ceil( ((double)size)/threads);
//    printf("size %d blocks %d threads %d \n",size,blocks, threads);
      numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//только свертка
      size=((size-1)/OPT_THREADS)+1;
      level++;
    }

    if (frst==1)
    {
    	cudaMalloc(&dev_d_v,sizeof(LongPointer)*LEVELS);
    	cudaMemcpy(dev_d_v,d_vfrst,sizeof(LongPointer)*LEVELS,cudaMemcpyHostToDevice);
    	frst=0;
    }

    err = cudaGetLastError();
    numb_backward<<<1,1>>>(d_vnumb[level],d_numb);
return 0;
}
__global__ void copy_block_plus(unsigned long long int *dv,unsigned long long int *dv0, LongPointer *d_tab, unsigned long long int *d_and, int j)
		{   int k,tid=threadIdx.x + blockIdx.x*blockDim.x;
		    unsigned long long int zero=1;

		    unsigned long long int *d_col;

		      d_col=d_tab[j-1];//d_tab[i];
		 //     _assign(d_res,d_col);
		 //     _and(d_res,d_and);
		      if(tid<NN2)  dv0[tid]=d_col[tid]&d_and[tid];

		    if(tid<NN2)  dv[tid]=dv0[tid];
		//    else dv[tid]=0;
		    if (tid==(NN2-1)) // in the last element need to zero the tail
		    {
		    	/*zero=(1<<(num % SIZE_OF_LONG_INT)-1)-1;
			  zero=~zero;*/
		    	k=(LENGTH1%SIZE_OF_LONG_INT);
		//    	printf("k=%i\n",k);
		    	zero=(zero<<k)-1;
		    	if (k!=0) dv0[tid]&=zero;
		    	dv[tid]=dv0[tid];
		    }
		}
__global__ void numb_backward_plus(LongPointer d_v,int *d_first_non_zero)
{
		*d_first_non_zero+=d_v[0];
//		if (d_v[0]>0)printf("numb:+%i= %i\n",d_v[0],*d_first_non_zero);
}
int number_plus(LongPointer *d_tab, unsigned long long int *d_and, int j,unsigned long long int *dv0, int size, int *d_numb,unsigned int N)
{
//    static int frst=1;
//    static LongPointer *dev_d_v;
    int big_n = size,level = 0;//,n=1;
    int i, red_count;
    cudaError_t err = cudaGetLastError();
////    err=cudaMemset(d_numb,0,sizeof(int));
 //   printf("cudaMemset %d , %s \n",err,cudaGetErrorString(err));
    unsigned int blocks, threads = (size < OPT_THREADS) ? size : OPT_THREADS;
    blocks = (int)ceil( ((double)size)/threads);
    copy_block_plus<<<blocks,threads>>>(d_vnumb[0],dv0, d_tab, d_and,j);

//    for(level=0; level < 2;level++)
//    {
 //   	    threads = (size < OPT_THREADS) ? size : OPT_THREADS;
 //   	    blocks = (int)ceil( ((double)size)/threads);
 //       printf("size %d blocks %d threads %d \n",size,blocks, threads);
       numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//3 шага, свертка не более, чем на 16
       if (size>1) size=((size-1)/OPT_REDUCE)+1;
       level++;

                threads = (size < OPT_THREADS) ? size : OPT_THREADS;
          	    blocks = (int)ceil( ((double)size)/threads);
//              printf("size %d blocks %d threads %d \n",size,blocks, threads);
             numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//3 шага, свертка не более чем на 128
             if (size>1) size=((size-1)/OPT_THREADS)+1;
             level++;
//       printf("\n=========================== level=%i, red_count=%i, size=%i \n",level, red_count, big_n);
 //       printf("level=%i, size=%i, red_count=%i \n",level,big_n, red_count);
//    }

//Если необходимо еще сворачивать (N>1048576)
    while (size>1)
    { threads = (size < OPT_THREADS) ? size : OPT_THREADS;
      blocks = (int)ceil( ((double)size)/threads);
//    printf("size %d blocks %d threads %d \n",size,blocks, threads);
      numb_shift<<<blocks,threads>>>(d_vnumb[level],d_vnumb[level+1],d_m,level,size);//только свертка
      size=((size-1)/OPT_THREADS)+1;
      level++;
    }

/*    if (frst==1)
    {
    	cudaMalloc(&dev_d_v,sizeof(LongPointer)*LEVELS);
    	cudaMemcpy(dev_d_v,d_vfrst,sizeof(LongPointer)*LEVELS,cudaMemcpyHostToDevice);
    	frst=0;
    }
*/
    err = cudaGetLastError();
    numb_backward_plus<<<1,1>>>(d_vnumb[level],d_numb);
return 0;
}

__host__ __device__ void assign_bit(unsigned long long int *h_v,int nz,int bit,int op)
{
   int ni;

   ni = get_64bit_word(nz,SIZE_OF_LONG_INT);
   int pos = position_in_64bit_word(nz,SIZE_OF_LONG_INT);

   assign_given_bit_to_position(&h_v[ni],bit,pos,op);
   //set_bit_to_position(&h_v[ni],pos);
 }

__host__ __device__ void set_bit(unsigned long long int *h_v,int nz)
{
   int ni;

   ni = get_64bit_word(nz,SIZE_OF_LONG_INT);
   int pos = position_in_64bit_word(nz,SIZE_OF_LONG_INT);

   set_bit_to_position(&h_v[ni],pos);
 }

void print_host_bit_column(char *label,unsigned long long *h_v,int length)
{
     FILE *f_ini;
     char s[1000];

     sprintf(s,"%s_bit.dat",label);
     if((f_ini =fopen(s,"wt"))== NULL) return;

     for(int i = 1;i <= length;i++)
     {
             fprintf(f_ini,"%10d %d \n",i,get_position_bit(h_v,i));
     }
     fclose(f_ini);
}

void print_device_bit_column(char *label,unsigned long long *d_v,int length,unsigned int N)
{
//	 static unsigned long long *h_v;
//	 static int flag_malloc=1;

/*	 if(flag_malloc==1)
	 {
	 h_v = (unsigned long long *)malloc(N*sizeof(unsigned long long));
	 flag_malloc=0;
	 }*/
     cudaMemcpy(h_new_v,d_v,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost);

     print_host_bit_column(label,h_new_v,length);

//     free(h_v);
}

 void print_device_bit_row(char *label,unsigned long long *d_v,int length,int row_flag,unsigned int N)
{
//	 unsigned long long *h_v;
//	static int flag_malloc=1;
	char s[N1][65];
	char bit_row[N1*65+1];
	FILE *f_ini;
	char fname[1000];

//	sprintf(fname,"%s.dat",label);
//	if((f_ini =fopen(fname,"wt"))== NULL) return;
//	fprintf(f_ini,"QQQQQQQ \n");
//	fclose(f_ini);



 //    if (flag_malloc==1)
 //    {
//	 h_v = (unsigned long long *)malloc(N*sizeof(unsigned long long));
//     flag_malloc=0;
 //    }
     cudaError_t err = cudaMemcpy(h_new_v,d_v,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
     strcpy(bit_row,"");

     for (int i = 0;i < N;i++)
     {

         long_to_binary(h_new_v[i],s[i],64);
//        puts(s[i]);
//         printf("%s %llu\n",s[i],h_v[i]);
//         fprintf(f_ini,"%s",s);

         strcat(bit_row,s[i]);
     }

 	 sprintf(fname,"%s.dat",label);
     bit_row[length] = 0;
     if(row_flag == 1)
     {
     	if((f_ini =fopen(fname,"wt"))== NULL) return;
        fprintf(f_ini,"%s\n",bit_row);

        fclose(f_ini);
     }
     else
     {
    	 if((f_ini =fopen(fname,"wt"))== NULL) return;
    	 for(int i = 0;i < length;i++)
    	 {
    		 fprintf(f_ini,"%5d %3c\n",i+1,bit_row[i]);
    	 }
    	 fclose(f_ini);
     }
     puts(label);
     puts(bit_row);
//     fclose(f_ini);
    // free(h_v);
}

void InitArrays()
{
	h_new_v=(unsigned long long  int *)malloc(sizeof(unsigned long long  int)*N1);
	tb=new char*[M];
    for(int i = 0;i < LEVELS;i++)
    {
	    unsigned long long *tmp;
        cudaError_t err = cudaMalloc(&tmp,sizeof(unsigned long long  int)*N1);
#ifdef ffff
        printf("cudaMalloc %d err %d %s %p \n",i,err,cudaGetErrorString(err),tmp);
#endif
        d_vfrst[i] = tmp;
        cudaMemset(d_vfrst[i],0,sizeof(unsigned long long  int)*N1);
    }

    for(int i = 0;i < LEVELS;i++)
    {
	    unsigned long long *tmp;
        cudaError_t err = cudaMalloc(&tmp,sizeof(unsigned long long  int)*N1);
#ifdef ffff
        printf("cudaMalloc %d err %d %s %p \n",i,err,cudaGetErrorString(err),tmp);
#endif
        d_vnumb[i] = tmp;
        cudaMemset(d_vnumb[i],0,sizeof(unsigned long long  int)*N1);
    }

	for(int i=0; i<M;i++) tb[i]=new char[LENGTH1+1];

//копируется константный массив для вычисления numb
	 cudaMalloc(&d_m,sizeof(unsigned long long  int)*6);
	 cudaError_t err_m= cudaMemcpy(d_m,m,6*sizeof(unsigned long long  int),cudaMemcpyHostToDevice);
	 printf("cudaMalloc err %d %s\n",err_m,cudaGetErrorString(err_m));
//    printf("m=[%llu,%llu,%llu,%llu,%llu,%llu] \n",m[0],m[1],m[2],m[3],m[4],m[5]);

}

//int main(void)
//{
//    unsigned long long  int h_v[N],h_new_v[N];
//    LongPointer  d_v[LEVELS];
//    int i,sh,*d_res,res[N],r_step;
//    char s1[1000],s2[1000],lprt[500];
//    FILE *f_ini,*f_res;
//    unsigned int blocks, threads = SIZE_OF_LONG_INT;
//    int *d_first_non_zero;
//    unsigned long long  int one = 1;
//
//    cudaMalloc(&d_first_non_zero,sizeof(int));
//
//    blocks = (int)ceil( ((double)N)/((double)SIZE_OF_LONG_INT));
//
//    srandom(time(NULL));
//    r_step = random()%10 +3;
//
//    printf("random step %d\n",r_step);
//
//    if((f_ini =fopen("init.dat","wt"))== NULL) return 0;
//
//    set_bit(h_v,POS_NON_ZERO);
//
//    for(i = 0;i < N;i++)
//    {
//    	    h_v[i] = 0;
//    }
//
//
//    print_host_bit_column("init",h_v,LENGTH);
//    set_bit(h_v,POS_NON_ZERO);
//    print_host_bit_column("init1",h_v,LENGTH);
//
//    for(i = 0;i < N;i++)
//        {
//        //	    h_v[i] = 0;
//        	  	long_to_binary(h_v[i],lprt);
//                fprintf(f_ini,"init %3d %25llu %s \n",i,h_v[i],lprt);
//        }
//    fclose(f_ini);
//   // exit(0);
//
//    for(int i = 0;i < LEVELS;i++)
//    {
//    	unsigned long long *tmp;
//        cudaError_t err = cudaMalloc(&tmp,sizeof(unsigned long long  int)*N);
//        printf("cudaMalloc %d err %d %s %p \n",i,err,cudaGetErrorString(err),tmp);
//        d_v[i] = tmp;
//        cudaMemset(d_v[i],0,sizeof(unsigned long long  int)*N);
//    }
//
////    cudaMalloc(&d_v,sizeof(unsigned long long  int)*N);
////
////    cudaMalloc(&d_res,sizeof(int)*N);
////
////    cudaMalloc(&d_v2,sizeof(unsigned long long  int)*N);
////    cudaMemset(d_v2,0,sizeof(unsigned long long  int)*N);
////
////    cudaMalloc(&d_v3,sizeof(unsigned long long  int)*N);
////    cudaMemset(d_v3,0,sizeof(unsigned long long  int)*N);
//
//    cudaError_t err1 = cudaMemcpy(d_v[0],h_v,sizeof(unsigned long long  int)*N,cudaMemcpyHostToDevice);
//    	printf("errors at after copy %d\n",err1);
//    //printf("H2D error %d %s\n",err1,cudaGetErrorString(err1));
//
////    cudaMemcpy(h_v1,d_v,sizeof(unsigned long long  int)*N,cudaMemcpyHostToDevice);
////    err1 = cudaGetLastError();
////        printf("debug copy error %d %s\n",err1,cudaGetErrorString(err1));
////TODO:
//
////
//
//
//    //  6. then up to 64x64
//    //  7. second stage
//    //  8. reverse
//    //  9. profile
//
//    //один блок,число нитей N/SIZE_OF_LONG_INT+1
//    //пишет только одна из 64
//
//    first(d_v,N,d_first_non_zero);
//
//    return 0;
//
////    int big_n = N,level = 0,n;
////    for(big_n = N; big_n > 1; big_n  = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT))
////    {
////    	n = (int)ceil((double)big_n/(double)SIZE_OF_LONG_INT);
////    	printf("n %3d big_n %3d level %d \n ",n,big_n,level);
////        reduce_array(d_v[level+1],d_v[level],big_n,level);
////
////        level++;
////
////    }
////
////
////    return 0;
//
////
////   // find<<<blocks,threads>>>(d_v,d_v1,d_res);
////
////    err1 = cudaGetLastError();
////    //printf("kernel error %d %s\n",err1,cudaGetErrorString(err1));
////
////	//cudaError_t err,err0 = cudaMemcpy(h_new_v,d_v1,sizeof(unsigned long long  int)*N,cudaMemcpyDeviceToHost);
////	//printf("h_new0 %ul\n",h_new_v[0]);
//////	err = cudaMemcpy(res,d_res,sizeof(int)*N,cudaMemcpyDeviceToHost);
////        //printf("h_new0 %ul\n",h_new_v[0]);
////
////
////	//printf("D2H error %d %s\n",err,cudaGetErrorString(err));
////	//printf("D2H error0 %d %s\n",err0,cudaGetErrorString(err0));
////
////	if( (f_res =fopen("result.dat","wt")) == NULL ) return 0;
////	for(i = 0;i < N;i++)
////	{
////		   long_to_binary(h_v[i],s1);
////		   long_to_binary(h_new_v[i],s2);
////		   //printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
////	       fprintf(f_res,"i %3d %s,%25llu result_vector %d init %s \n",i,s2,h_new_v[i],get_position_bit(h_new_v,i),s1);
////	      // printf("i %3d %s,%25llu res %d\n",i,s2,h_new_v[i],res[i]);
////	}
////	fclose(f_res);
//
//    return 0;
//
//}
