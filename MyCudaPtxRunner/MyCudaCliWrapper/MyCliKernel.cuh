#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int *c, int const* a, int const* b);