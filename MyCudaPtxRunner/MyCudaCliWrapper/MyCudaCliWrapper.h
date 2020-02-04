#pragma once

#include <stdio.h>
#include <windows.h>
#include <iostream>

using namespace System;
using namespace System::Runtime::InteropServices;

namespace ManagedCu
{

	public ref class MyCudaCliWrap {

	public:
		MyCudaCliWrap();
		~MyCudaCliWrap();

		void RunAdd(cli::array<int> ^c, cli::array<int> ^a, cli::array<int> ^b, int size);
		
	};

}