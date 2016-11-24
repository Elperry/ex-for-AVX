#include <stdio.h>
#include <immintrin.h>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>

#define LENGTH   100000000 // 200 million
using namespace std;

void* malloc_aligned_generic(size_t alignment, size_t size)
{
	char *p1 = (char*)malloc(size + alignment + sizeof(void*));
	p1 += sizeof(void*);
	void *p2 = (void*)((uintptr_t)p1 + alignment - (uintptr_t)p1%alignment);
	*((void**)((uintptr_t)p2 - sizeof(void*))) = p1 - sizeof(void*);
	return p2;
}


void free_aligned(void *memory)
{
	free(*((void**)((uintptr_t)memory - sizeof(void*))));
}

void cosinx(long int n ,long int terms , float* x , float* result){

	for (int i = 0; i < n; i++){
		double value = 1;
		double num = x[i] * x[i];
		double factorial = 2 ;
		int sign = -1;
		for (long int j = 1; j <= terms; j++){
			value += sign * (num / factorial);
			num *= x[i] * x[i];
			factorial *= (2 * j + 1) * (2 * j + 2);
			sign *= -1;

		    }
	
		result[i] = value;
	}

}
#pragma intel optimization_parameter target_arch=avx
__declspec(noinline) void cosinx_AVX(long int n, long int terms, float* x, float* result){
	__m256 value, setOfX, num, factorial,tmp,t;
	for (int i = 0; i < n; i+=8){
		 value =_mm256_set1_ps(1);
		 setOfX = _mm256_load_ps(&x[i]);
		 num = _mm256_mul_ps(setOfX, setOfX);
		 factorial =_mm256_set1_ps(2);
		int sign = -1;
		for (long int j = 1; j <= terms; j++){
			tmp = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(sign), num), factorial);
			value = _mm256_add_ps(value, tmp);
			num = _mm256_mul_ps(_mm256_mul_ps(setOfX, setOfX), num);
			t = _mm256_mul_ps(_mm256_set1_ps(2 * j + 1), _mm256_set1_ps(2 * j + 2));
			factorial = _mm256_mul_ps(factorial,t);
			
			sign *= -1;
			
		}
		
		_mm256_store_ps(&result[i], value);
	}

}



void main()
{

	const unsigned long n =8* 100000;
	const unsigned long terms = 50;
	float *x;
	float *result;

	result = (float *)malloc_aligned_generic(32, sizeof(float)*n);
	x = (float *)malloc_aligned_generic(32, sizeof(float)*n);

	for (int i = 0; i < n; i++)
		x[i] = (float) 3.14 * rand()/RAND_MAX;
	long c1 = clock();
	cosinx_AVX(n, terms, x, result);
	printf("avx time = %ld\n", clock() - c1);
	long clock1 = clock();
	cosinx(n, terms, x, result);
	printf("no avx time = %ld\n", clock() - clock1);

	for (int i = 0; i < n; i++)
		if (result[i] > (cos(x[i]) + 0.01) || result[i] < (cos(x[i]) - 0.01)) { printf("error"); }
	
	free_aligned(x);
	free_aligned(result);

	

	getchar();
	
}