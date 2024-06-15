#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "param.h"
#include "slice.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/copy.h>

unsigned long long int h_v[N1];// for print mast be copied from d_v


int Slice::Init(unsigned int k)
{

	cudaError_t err1, err = cudaGetLastError();

//	printf("before all error %d , %s \n",err,cudaGetErrorString(err));
//	if (err!=0)exit(0);
	length=k;
	NN=(((k % SIZE_OF_LONG_INT) ==0)?(k/SIZE_OF_LONG_INT):(k/SIZE_OF_LONG_INT+1));
//    printf("slice.init %u ", NN);
#ifdef ssss
	int *d_i;
	printf("Slice init %d %s\n",err,cudaGetErrorString(err));
	err = cudaMalloc(&d_i,sizeof(int));

	d_first_non_zero =d_i;
	printf("Slice alloc error %d %s \n",err,cudaGetErrorString(err));
#endif
//	err1 = cudaGetLastError();
//	printf("before alloc error %d , %s \n",err1,cudaGetErrorString(err1));
    err = cudaMalloc(&d_v,NN*sizeof(unsigned long long int));
 //   printf("Slice alloc error %d , %s ,%p \n",err,cudaGetErrorString(err),d_v);
#ifdef ssss
    printf("Slice alloc error %d %s \n",err,cudaGetErrorString(err));
#endif
    cudaMemset(d_v,0,NN*sizeof(unsigned long long int));

//	exit(0);

    return err;
}
//void Print();
//void set_from_host_array (unsigned long long int *f_h_v);
//void set_from_device_array (unsigned long long int *f_h_v);

__global__ void set_long_values(unsigned long long int *d_v,unsigned long long int num)
{
	//char s[100];
//	printf("set \n");//long %llu \n",num);
//	return;
//	long_to_binary(num,s);
	d_v[blockIdx.x] = num;
//	printf("set long %s \n",s);
}

//заполнить единичками,
void Slice::SET()
{
	 unsigned long long int zero = 0;
	 zero = ~zero;
#ifdef ss
	 char s[100];
	 long_to_binary(zero,s);
	 printf("SET %s \n",s);


	 cudaError_t err = cudaGetLastError();
	 printf("error before set_lon_values %d \n",err);
	 cudaError_t err_c = cudaMemcpy(h_v,d_v,sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	 long_to_binary(h_v[0],s);
	 printf("h_v[0] %llu err %d %s\n",h_v[0],err_c,s);
	 print("q1",1);
#endif
     set_long_values<<<NN,1>>>(d_v,zero);
 //    printf("SET: %i->%llu \n",NN,zero);
#ifdef qq
     err_c = cudaMemcpy(h_v,d_v,sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	 long_to_binary(h_v[0],s);
	 printf("h_v[0] %llu err %d %s\n",h_v[0],err_c,s);


     print("q2",1);

	 err = cudaGetLastError();
	 printf("error after set_lon_values %d \n",err);
#endif
}

//заполнить нулями,
void Slice::CLR()
{
	 unsigned long long int zero = 0;
    set_long_values<<<NN,1>>>(d_v,zero);
}
__global__ void set_mask_values(unsigned long long int *d_v, int num)
{ unsigned long long int zero=1;
  int num_el=num>>6; // номер элемента, содержащий переход от 0 к 1;
  int el=num % SIZE_OF_LONG_INT;
//  printf("%i in %i \n", num,num_el);
  if (blockIdx.x==num_el)
   {
	  zero=(el==0)?0:(zero<<(el-1))-1;
	  zero=~zero;
   }
  else
  {
      zero=0;
      if (blockIdx.x>num_el)
      {
    	  zero=~zero;
      }
  }
   d_v[blockIdx.x]=zero;
}
void Slice::MASK(int i)
{
	set_mask_values<<<NN,1>>>(d_v,i);
}

__global__ void get_kernel(unsigned long long int *d_v,unsigned char *d_num,bool get,int n)
{
		int num = get_position_bit(d_v,n);
		*d_num = (unsigned char)num;
}

__global__ void put(unsigned long long int *d_v,unsigned char d_num,int n)
{
	assign_bit(d_v,n,d_num,SET);
}

//доступ к i-ой компоненте слайса, как на чтение, так и на запись,
unsigned char Slice::get(int i)
{
//	cudaError_t   err = cudaGetLastError();
//    printf("begin get %d, %d , %s \n",i,err,cudaGetErrorString(err));
    unsigned char n;
   static int flag=1;
   static unsigned char *d_n;
    if (flag==1)
    {
    cudaMalloc(&d_n,sizeof(unsigned char));
    flag=0;
    }
	get_kernel<<<1,1>>>(d_v,d_n,1,i);

	cudaMemcpy(&n,d_n,sizeof(unsigned char),cudaMemcpyDeviceToHost);
//	err = cudaGetLastError();
//	printf("end get %d, %d , %s \n",i,err,cudaGetErrorString(err));
//	if(err!=0)exit(0);
//   printf("get_ %d : %us \n", i,n);
	return (unsigned char)n;
}

void Slice::set(int i,unsigned char n)
{

     put<<<1,1>>>(d_v,n,i);
}

// - выдает номер старшей единичке в слайсе Y
int Slice::FND()
{
	int h_first_non_zero;
	static 	int *d_first_non_zero;
    static int flag_malloc=1;


   if (flag_malloc==1)
  {
	cudaMalloc(&d_first_non_zero,sizeof(int));
    flag_malloc=0;
   }
//    print_device_bit_row("FND",d_v,NN*SIZE_OF_LONG_INT,0,NN);

	first(d_v,NN,d_first_non_zero,NN);

	cudaMemcpy(&h_first_non_zero,d_first_non_zero,sizeof(int),cudaMemcpyDeviceToHost);
    if (h_first_non_zero>length)
    	h_first_non_zero=0;
//    printf("FND %i",h_first_non_zero);
	return h_first_non_zero;
}
int Slice::NUMB()
{

	int h_first_non_zero;
	static 	int *d_first_non_zero;
    static int flag_malloc=1;

    if (flag_malloc==1)
    {
	cudaMalloc(&d_first_non_zero,sizeof(int));
	 flag_malloc=0;
    }
//    print_device_bit_row("NUMB",d_v,NN*SIZE_OF_LONG_INT,0,NN);

	number(d_v,NN,d_first_non_zero,NN);

	cudaMemcpy(&h_first_non_zero,d_first_non_zero,sizeof(int),cudaMemcpyDeviceToHost);
//    printf("NUMB=%i \n",h_first_non_zero);
	return h_first_non_zero;
}
__global__ void numb_thrust(int *dev_vec, LongPointer d_v)
{ int k;
  unsigned long long int zero=1;
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if (tid<N1)
  {
	  dev_vec[tid]=__popcll(d_v[tid]);
//      printf("%i  ",dev_vec[tid]);
  }
//  else dev_vec[tid]=0;
  if (tid==(N1-1)) // in the last element need to zero the tail
  {
  	/*zero=(1<<(num % SIZE_OF_LONG_INT)-1)-1;
	  zero=~zero;*/
  	k=(LENGTH1%SIZE_OF_LONG_INT);
//    	printf("k=%i\n",k);
  	zero=(zero<<k)-1;
  	zero&=d_v[tid];
  	dev_vec[tid]=__popcll(zero);
  }
}
int Slice::NUMB1()
{
	 thrust::device_vector<int> d_a(N1);
	 int * dv_ptr = thrust::raw_pointer_cast(d_a.data());
	 numb_thrust<<<blocks1,threads1>>>(dv_ptr,d_v);
	int h_first_non_zero;
     h_first_non_zero=thrust::reduce(d_a.begin(),d_a.end());
 //   printf("NUMB1=%i \n",h_first_non_zero);
	return h_first_non_zero;
}
__global__ void fnd_thrust(int *dev_vec, LongPointer d_v)
{ int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if (tid<N1)
  {   int fnd_tid= (__ffsll(d_v[tid])!=0)?(__ffsll(d_v[tid])+tid*SIZE_OF_LONG_INT):(SIZE_OF_LONG_INT*N1+1);
	  dev_vec[tid]=fnd_tid;
//      printf("%i  ",dev_vec[tid]);
  }
}
int Slice::FND1()
{
	 thrust::device_vector<int> d_a(N1);
	 int * dv_ptr = thrust::raw_pointer_cast(d_a.data());
	 fnd_thrust<<<blocks1,threads1>>>(dv_ptr,d_v);
	int h_first_non_zero;
     h_first_non_zero=*(thrust::min_element(d_a.begin(),d_a.end()));
 //   printf("NUMB1=%i \n",h_first_non_zero);
     if (h_first_non_zero>length)
         	h_first_non_zero=0;
	return h_first_non_zero;
}
// - то же самое, но эту единичку заменяет на ноль
int Slice::STEP()
{
//	print_device_bit_row("STEP",d_v,NN*SIZE_OF_LONG_INT,0,NN);
	int f = FND();
	if (f>0)set(f,0);
//	print_device_bit_row("S_res",d_v,NN*SIZE_OF_LONG_INT,0,NN);
//	printf("vertex %i ",f);
	return f;
}

// - конвертирует слайс в строку (используется крайне редко).
/*void Slice::CONVERT()
{
//	word_flag = 1;
}*/

//Побитовые X and Y, not X, X or Y, X xor Y
__global__ void and_long_values(unsigned long long int *d_v,unsigned long long int *d_v1)
{
	d_v[blockIdx.x] &= d_v1[blockIdx.x];
}

__global__ void or_long_values(unsigned long long int *d_v,unsigned long long int *d_v1)
{
#ifdef ssss
	unsigned long long old,old1;
	char s_old[100],s_old1[100],res[100];
	old  = d_v[blockIdx.x];
	old1 = d_v1[blockIdx.x];

	long_to_binary(old,s_old);
	long_to_binary(old1,s_old1);
#endif
	d_v[blockIdx.x] |= d_v1[blockIdx.x];
#ifdef ssss
	long_to_binary(d_v[blockIdx.x],res);
	printf("blockIdx.x %u old %llu %s %llu %s %llu %s\n",blockIdx.x,old,s_old,old1,s_old1,d_v[blockIdx.x],res);
#endif

}

__global__ void xor_long_values(unsigned long long int *d_v,unsigned long long int *d_v1)
{
	d_v[blockIdx.x] ^= d_v1[blockIdx.x];
}

__global__ void not_long_values(unsigned long long int *d_v)
{
	d_v[blockIdx.x]=~d_v[blockIdx.x];
}

Slice Slice::operator & (const Slice & b)
{
	and_long_values<<<NN,1>>>(d_v,b.d_v);

	return *this;
}
void Slice::AND(const Slice *b)
{
	and_long_values<<<NN,1>>>(d_v,b->d_v);
}
void Slice::OR(const Slice *b)
{
	or_long_values<<<NN,1>>>(d_v,b->d_v);
}
Slice Slice::operator | (const Slice & b)
{

	or_long_values<<<NN,1>>>(d_v,b.d_v);

	return *this;
}
void Slice::XOR(const Slice * b)
{
	xor_long_values<<<NN,1>>>(d_v,b->d_v);
}
Slice Slice::operator ^ (const Slice & b)
{

	xor_long_values<<<NN,1>>>(d_v,b.d_v);

	return *this;
}

Slice Slice::operator ~()
{
	not_long_values<<<NN,1>>>(d_v);
	return *this;
}
// - true, если X ненулевой.
void Slice::NOT()
{
	not_long_values<<<NN,1>>>(d_v);
}
bool Slice::SOME()
{
	int f = FND();
//  printf("SOME %d \n", f);
	return (f > 0);
}
bool Slice::SOME1()
{
	 int h_if_zero;
	static  int *d_if_zero;
    static int flag_malloc=1;

    if (flag_malloc==1)
    {
	cudaMalloc(&d_if_zero,sizeof( int));
	  flag_malloc=0;
    }
//    print_device_bit_row("FND",d_v,NN*SIZE_OF_LONG_INT,0,NN);

	some(d_v,NN,d_if_zero,NN);

	cudaMemcpy(&h_if_zero,d_if_zero,sizeof( int),cudaMemcpyDeviceToHost);

   printf("SOME1 %d \n", h_if_zero);
	return (h_if_zero!=0);
}
bool Slice::ZERO()
{
	int f = FND();
//   printf("ZERO %d \n", f);
	return (f == 0);
}
unsigned long long int char_to_long(char *s)
{
	unsigned long long int u = 0,u1,t;
	double d;
//	char str[LENGTH1];
//    puts("char_to_long");
	int len = (LENGTH1 < SIZE_OF_LONG_INT ? LENGTH1: SIZE_OF_LONG_INT);

	for (int i = 0;i < len;i++)
	{
#ifdef ssss
		 printf("i %d\n",i);
#endif
		 d = pow(2.0,(double)i);
		 t = (unsigned long long int)ceil(d);
		 u1 = (s[i]-'0')*t;
         u += u1;
#ifdef ssss
         long_to_binary(u,str);
         printf("i %d d %40.25e t %25llu s[i] %c u1 %25llu u %25llu %s\n",i,d,t,s[i],u1,u,str);
#endif
	}
	return u;
}

Slice Slice::operator= (char *s)
{
	char num[SIZE_OF_LONG_INT+1];
#ifdef sssss
	puts(s);
#endif
	for(int i = 0;i < strlen(s);i += SIZE_OF_LONG_INT)
	{
	    strncpy(num,s+i,SIZE_OF_LONG_INT);
	    num[SIZE_OF_LONG_INT] = 0;
	    h_v[i/SIZE_OF_LONG_INT] = char_to_long(num);
#ifdef sssss
	    printf("i %d num %s %llu\n",i,num,h_v[i/SIZE_OF_LONG_INT]);
#endif
	}
	cudaMemcpy(d_v,h_v,NN*sizeof(unsigned long long int),cudaMemcpyHostToDevice);
//	puts("=s");
	return *this;
}



void Slice::assign(const Slice * s)
{
//	printf("before:= %llu %llu \n", d_v,s->d_v);
//	exit(0);
	if (NN!=s->NN)printf("sizes of slices are not equal %u %u %u %u \n",NN,s->NN,length,s->length);

    length=s->length;
    NN=s->NN;
	set_kernel<<<blocks1,threads1>>>(d_v,s->d_v);
//	printf("after:= %p %p \n", d_v,s->d_v);
}

Slice Slice::operator= (Slice *s)
{
//	printf("before:= %llu %llu \n", d_v,s->d_v);
//	exit(0);
	if (NN!=s->NN)printf("sizes of slices are not equal %u %u %u %u \n",NN,s->NN,length,s->length);

    length=s->length;
    NN=s->NN;
	set_kernel<<<NN,1>>>(d_v,s->d_v);
//	printf("after:= %p %p \n", d_v,s->d_v);
	return *this;
}

void Slice::convert_to_string(char *str)
{
    char s[SIZE_OF_LONG_INT+1];
//    printf("convert %d %d \n", NN, length);

    cudaMemcpy(h_v,d_v,NN*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
//	printf("convert: %d %llu \n",0,h_v[0]);
   strcpy(str,"");
   for (int i = 0;i < NN;i++)
   {
       long_to_binary(h_v[i],s,length);
//#ifdef ssss
 //      puts(s);
//#endif
//       sprintf(s,"%s",s);
       strcat(str,s);
   }
   str[length] = 0;
//#ifdef ssss
//   puts(str);
//#endif
}

void __global__ digit_kernel(unsigned long long *w, unsigned long long *dig)
{
	dig[0]=__brevll(w[0]);
}

unsigned long long int Slice::ToDigit()
{ unsigned long long high,low,*d_dig1,res=0;
	if (NN==1)
	{  cudaMalloc(&d_dig1,sizeof(unsigned long long));
		digit_kernel<<<1,1>>>(d_v,d_dig1);
		cudaMemcpy(&res,d_dig1,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
		res>>=(64-H1);
	}
/*	{
		cudaMemcpy(&d_dig,d_v,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
		low=1;
		high=1<<(H1-1);
	for (int i=0; i<H1;i++)
	{   if (d_dig&low)
		res+=high;
		low<<=1;
		high>>=1;
	}
	}*/
	return res;
}
unsigned long long int Slice::FromDigit(unsigned long long dig)
{	 unsigned long long high,low,*d_dig1,res=0;
		if (NN==1)
		{ dig<<=(64-H1);
			cudaMalloc(&d_dig1,sizeof(unsigned long long));
			cudaMemcpy(d_dig1,&dig,sizeof(unsigned long long),cudaMemcpyHostToDevice);
			digit_kernel<<<1,1>>>(d_dig1,d_v);
		}
}




void __device__ trim_(unsigned long long int *d_v, unsigned long long int *d_v_in,int i, int h)
// предполагается и для матрицы, и для слайса
//threadth - для номера столбца
//block - для номера элемента по меньшему слайсу
/*{
	int i=10;
	int h=15;
	unsigned long long int d_v, d_v_in= 4842603519;
	int num_el=i>>6;//номер элемента в большем слайсе
   int num_bit_first= i % SIZE_OF_LONG_INT; // номер бита в элементе, который станет первым в маленьком слайсе
   int num_bit_last = (i+h) % SIZE_OF_LONG_INT;
   unsigned long long int teal, head =d_v_in<<num_bit_first;

  	   teal = d_v_in>>(SIZE_OF_LONG_INT-num_bit_first);
	   d_v=head | teal;

   printf("head=%ull teal=%ull \n",head,teal);
}*/

{  int num_el=(i-1)>>6;//номер первого элемента в большем слайсе
   int num_el1=(h-1)>>6; // номер последнего элемента в маленьком
   int num_el2=(i+h-1)>>6;// номер последнего элемента в большом слайсе
   int num_bit_first= i % SIZE_OF_LONG_INT -1; // номер бита в элементе, который станет первым в маленьком слайсе
   int num_bit_last = h % SIZE_OF_LONG_INT;
   char prb[65];
//   printf("num_els %i and %i (%i) bits from %i to %i \n",num_el,num_el1,num_el2, num_bit_first, num_bit_last);
   unsigned long long int teal, head =d_v_in[blockIdx.x+num_el]>>(num_bit_first);
//   long_to_binary(head,prb,64);
//   	   printf("\n head:");printf(prb);
   if (blockIdx.x +num_el< num_el2)
   {
	   teal = d_v_in[blockIdx.x+1+num_el]<<(SIZE_OF_LONG_INT-num_bit_first);
//	   long_to_binary(teal,prb,64);
//	   printf("\n teal:");printf(prb);
	   d_v[blockIdx.x]=head | teal;

//	  	   printf("\n elem:");printf(prb);
   }
   if (blockIdx.x==num_el1) // обрезать последние биты от num_bit_last
   {
	   teal=1;
	   teal=(num_bit_last==0)? ~0:((teal<<num_bit_last)-1);
//	   long_to_binary(teal,prb,64);
//	   printf("\n teal_up (%i):",num_bit_last);printf(prb);
	   d_v[blockIdx.x]&=teal;
   }
//   printf("ind=%i d_v_in=%llu head=%llu teal=%llu d_v=%llu\n",blockIdx.x+num_el, d_v_in[blockIdx.x+num_el],head,teal,d_v[blockIdx.x]);
}

void __global__ trim_slice_kernal(unsigned long long int *d_v, unsigned long long int *d_v_in,int i, int h)
{ trim_(d_v,d_v_in,i,h);
//   printf("in trim_kernal");
}

void Slice::trim(int i, int h, Slice *s)
{
//	puts("in Slice.trim \n");
//	unsigned long long int *d_v
	unsigned long long int *d_v_in;
	d_v_in= s->get_device_pointer();
	int hh,n=this->NN;
	if (s->length>i+h)
	{hh=h;}
	else
	{hh=s->length-i+1;}

	trim_slice_kernal<<<n,1>>>(d_v,d_v_in,i,hh);
}

void __device__ shiftup(unsigned long long int *d_v, unsigned long long int *d_v_in,int i)
{
	int num_el=i>>6;//номер элемента в большем слайсе
//int num_el1=(i+h)>>6;
int num_bit_first= i % SIZE_OF_LONG_INT ; // номер бита в элементе, который станет первым в маленьком слайсе
//int num_bit_last = h % SIZE_OF_LONG_INT;
//printf("num_els %i (%i) bits from %i  \n",blockIdx.x +num_el,gridDim.x,num_bit_first);
unsigned long long int teal, head =d_v_in[blockIdx.x+num_el]>>(num_bit_first);
if (blockIdx.x +num_el<gridDim.x)//?????????
{
	   teal = (blockIdx.x+1+num_el<gridDim.x)?d_v_in[blockIdx.x+1+num_el]<<(SIZE_OF_LONG_INT-num_bit_first):0;
	   d_v[blockIdx.x]=head | teal;
}
else // обрезать последние биты от num_bit_last
{
//	   teal=(1<<num_bit_last) -1;
	   d_v[blockIdx.x]=0;//head & teal;
}
}

void __device__ shiftdown(unsigned long long int *d_v, unsigned long long int *d_v_in,int i)
{
	int num_el=i>>6;//номер элемента в большем слайсе
//int num_el1=(i+h)>>6;
int num_bit_first= i % SIZE_OF_LONG_INT ; // номер бита в элементе, который станет первым в маленьком слайсе
//int num_bit_last = h % SIZE_OF_LONG_INT;
//printf("num_els %i  bits from %i  \n",blockIdx.x -num_el,num_bit_first);
unsigned long long int teal,head;
if (blockIdx.x >num_el)//?????????
{      head =d_v_in[blockIdx.x-num_el]<<(num_bit_first);
	   teal =d_v_in[blockIdx.x-1-num_el]>>(SIZE_OF_LONG_INT-num_bit_first);
	   d_v[blockIdx.x]=head | teal;
	   printf("num_els %i (%i)%llu head=%llu (%i)%llu teal=%llu  \n",blockIdx.x,blockIdx.x-num_el,d_v_in[blockIdx.x-num_el],head,blockIdx.x-1-num_el,d_v_in[blockIdx.x-1-num_el],teal);
}
else // обрезать последние биты от num_bit_last
{
//	   teal=(1<<num_bit_last) -1;
	   d_v[blockIdx.x]=(blockIdx.x==num_el)? (d_v_in[0]<<(num_bit_first)):0;//head & teal;
	   printf("num_els %i  0 %i \n",blockIdx.x,num_el );
}

}

void __global__ shiftdown_kernel(unsigned long long int *d_v, unsigned long long int *d_v_in,int i)
{
	shiftdown(d_v,d_v_in,i);
}

void __global__ shiftup_kernel(unsigned long long int *d_v, unsigned long long int *d_v_in,int i)
{
	shiftup(d_v,d_v_in,i);
}
void Slice::shift_up(int i,Slice *s)
{
	unsigned long long int *d_v_in;
	d_v_in= s->get_device_pointer();
	int n=NN;
	shiftup_kernel<<<n,1>>>(d_v,d_v_in,i);
}
void Slice::shift_down(int i,Slice *s)
{
	unsigned long long int *d_v_in;
	d_v_in= s->get_device_pointer();
	int n=NN;
	shiftdown_kernel<<<n,1>>>(d_v,d_v_in,i);
}
