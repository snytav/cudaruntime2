/*
 *  param.h saves the parameters of variative table
 */

//actual length of a column (in bits)
//M*LENGTH1=2^25 <-ОГРАНИЧЕНИЕ ПО ПАМЯТИ
// M>20 ТОЛЬКО ЧАСТЬ ТАБЛИЦЫ, ОСТАЛЬНОЕ ПЕРЕБЕРАЕТСЯ ДОБАВЛЕНИЕМ СТРОКИ
#define LENGTH1 524288
//32768
//524288
//1048576
// это до2^20
//(65536)
//(163400)
//size of column

#define INFINITE ((1<<H1)-1)
//код бесконечности для матрицы весов
#define H1 21
//max value=31 because of int was used;
#define VER 1
// 100000
#define M H1
//(VER*H1)
// size of row
// number of vertex for
//        <left,right> M=log_2(VER),
//        for adj. matrix M=VER
//        for weight matrix M=VER*H

//		int threads=(32<NN)?32:NN;
//		int blocks=(NN-1)/ threads+1;

// NN, threads, blocks in star_kernel.cu

	  extern  const int NN1;
      extern  int threads1;
	  extern int blocks1;
