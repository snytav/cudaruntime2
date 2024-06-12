/*
 * find.h
 *
 *  Created on: Aug 31, 2015
 *      Author: snytav
 */



#ifndef FIND_H_
#define FIND_H_





#define SIZE_OF_LONG_INT 64
//length of a column in 64-bit words
#define N1 (((LENGTH1 % SIZE_OF_LONG_INT) ==0) ?  (LENGTH1/SIZE_OF_LONG_INT): (LENGTH1/SIZE_OF_LONG_INT+1))

typedef enum {SET,OR} operations;

#define POS_NON_ZERO 2
#define POS_NON_ZERO1 5492

typedef unsigned long long int *LongPointer;

void __global__ find(unsigned long long  int* x, unsigned long long  int* new_x, unsigned int N);


__host__ __device__ void set_bit(unsigned long long int *h_v,int nz);

void print_host_bit_column(char *label,unsigned long long *h_v,int length);

void print_device_bit_column(char *label,unsigned long long *d_v,int length,unsigned int N);

int first(unsigned long long int *dv0,int size,int *d_first_non_zero,unsigned int N);
int some(unsigned long long int *dv0,int size, int *d_if_zero,unsigned int N);
int number(unsigned long long int *dv0,int size, int *d_if_zero,unsigned int N);
int number_plus(LongPointer *d_tab, unsigned long long int *d_and, int i,unsigned long long int *dv0,int size, int *d_if_zero,unsigned int N);
void InitArrays();

__host__ __device__ void long_to_binary(unsigned long long  int x,char *b,unsigned int length);
__host__ __device__ void long_to_binary1(unsigned long long  int x,char *b,unsigned int length);
__host__ __device__ void set_given_bit_to_position(unsigned long long int *x,int bit,int pos,int op);
//__host__ __device__ void set_bit_to_position(unsigned long long int *x,int pos);
__host__ __device__ void assign_bit(unsigned long long int *h_v,int nz,int bit,int op);

__host__ __device__ int get_position_bit(unsigned long long int *h,int n);

//__host__ __device__
void print_device_bit_row(char *label,unsigned long long *d_v,int length,int row, unsigned int N);

__host__ __device__  int position_in_64bit_word(int num,int div);

__host__ __device__  int get_64bit_word(int num,int div);

__device__ unsigned long long int get_array(unsigned long long  int *x,int n,int size);

__global__ void set_kernel(unsigned long long int *dst,unsigned long long int *src);

#endif /* FIND_H_ */
