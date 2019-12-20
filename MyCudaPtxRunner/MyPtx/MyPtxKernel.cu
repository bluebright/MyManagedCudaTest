/**
* You can set multiple kernel method!!
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
* You must write on extern "C"
*/
extern "C" {

	__global__ void addKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}

	__global__ void subtractKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] - b[i];
	}

	int main() { return 0; }

}


