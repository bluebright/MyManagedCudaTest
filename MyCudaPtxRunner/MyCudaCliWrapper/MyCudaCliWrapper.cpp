#include "MyCudaCliWrapper.h"

#include "MyCliKernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace ManagedCu
{
	MyCudaCliWrap::MyCudaCliWrap() {

	}
	
	MyCudaCliWrap::~MyCudaCliWrap() {

	}
	
	void MyCudaCliWrap::RunAdd(cli::array<int> ^c, cli::array<int> ^a, cli::array<int> ^b, int size)
	{
		//extern void addKernel(int* c, int const* a, const int* b);

		pin_ptr<int> pin_a = &a[0];
		pin_ptr<int> pin_b = &b[0];
		pin_ptr<int> pin_c = &c[0];

		int *p_a = pin_a;
		int *p_b = pin_b;
		int *p_c = pin_c;

		int* dev_a = 0;
		int* dev_b = 0;
		int* dev_c = 0;

		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));

		cudaStatus = cudaMemcpy(dev_a, p_a, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(dev_b, p_b, size * sizeof(int), cudaMemcpyHostToDevice);
		
		void* args[] = { &dev_c, &dev_a, &dev_b };

		cudaLaunchKernel(
			(const void*)&addKernel, // pointer to kernel func.
			dim3(1), // grid
			dim3(size), // block
			args  // arguments
		);

		cudaStatus = cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(p_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

	}

}