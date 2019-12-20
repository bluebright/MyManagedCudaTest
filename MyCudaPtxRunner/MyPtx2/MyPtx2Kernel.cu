/*
* Default code in Managedcuda document
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


extern "C" {

	__global__ void multKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] * b[i];
	}

	int main() { return 0; }

}
