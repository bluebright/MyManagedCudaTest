using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MyCudaPtxRunner
{
    /// <summary>
    /// MainWindow.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void RunButton_Click(object sender, RoutedEventArgs e)
        {

            int N = 5;
            int deviceID = 0;

            using (CudaContext ctx = new CudaContext(deviceID))
            {

                CudaKernel kernel_1 = ctx.LoadKernel("MyPtxKernel.ptx", "addKernel");
                kernel_1.GridDimensions = 1;
                kernel_1.BlockDimensions = 5;

                CudaKernel kernel_2 = ctx.LoadKernel("MyPtx2Kernel.ptx", "multKernel");
                kernel_2.GridDimensions = 1;
                kernel_2.BlockDimensions = 5;

                CudaKernel kernel_3 = ctx.LoadKernel("MyPtxKernel.ptx", "subtractKernel");
                kernel_2.GridDimensions = 1;
                kernel_2.BlockDimensions = 5;

                CudaKernel kernel_4 = ctx.LoadKernel("MyPtx2Kernel_2.ptx", "multKernel");
                kernel_2.GridDimensions = 1;
                kernel_2.BlockDimensions = 5;

                // Allocate input vectors h_A and h_B in host memory
                int[] h_A = { 1, 2, 3, 4, 5 };
                int[] h_B = { 10, 20, 30, 40, 50 };

                // Allocate vectors in device memory and copy vectors from host memory to device memory 
                CudaDeviceVariable<int> d_A = h_A;
                CudaDeviceVariable<int> d_B = h_B;
                CudaDeviceVariable<int> d_C = new CudaDeviceVariable<int>(N);
                // Invoke kernel
                kernel_1.Run(d_C.DevicePointer, d_A.DevicePointer, d_B.DevicePointer);

                CudaDeviceVariable<int> d_D = h_A;
                CudaDeviceVariable<int> d_E = h_B;
                CudaDeviceVariable<int> d_F = new CudaDeviceVariable<int>(N);
                kernel_2.Run(d_F.DevicePointer, d_D.DevicePointer, d_E.DevicePointer);

                CudaDeviceVariable<int> d_G = h_A;
                CudaDeviceVariable<int> d_H = h_B;
                CudaDeviceVariable<int> d_I = new CudaDeviceVariable<int>(N);
                kernel_3.Run(d_I.DevicePointer, d_G.DevicePointer, d_H.DevicePointer);

                CudaDeviceVariable<int> d_J = h_A;
                CudaDeviceVariable<int> d_K = h_B;
                CudaDeviceVariable<int> d_L = new CudaDeviceVariable<int>(N);
                kernel_4.Run(d_L.DevicePointer, d_J.DevicePointer, d_K.DevicePointer);

                // Copy result from device memory to host memory
                // h_C contains the result in host memory
                int[] h_C = d_C;
                int[] h_F = d_F;
                int[] h_I = d_I;
                int[] h_L = d_L;

                TextBlock_1.Text = h_C[0] + "," + h_C[1] + "," + h_C[2] + "," + h_C[3] + "," + h_C[4] + "\n" +
                    h_I[0] + "," + h_I[1] + "," + h_I[2] + "," + h_I[3] + "," + h_I[4];
                TextBox_1.Text = h_F[0] + "," + h_F[1] + "," + h_F[2] + "," + h_F[3] + "," + h_F[4] + "\n" +
                    h_L[0] + "," + h_L[1] + "," + h_L[2] + "," + h_L[3] + "," + h_L[4];

                d_A.Dispose();
                d_B.Dispose();
                d_C.Dispose();
                d_D.Dispose();
                d_E.Dispose();
                d_F.Dispose();
                d_G.Dispose();
                d_H.Dispose();
                d_I.Dispose();
                d_J.Dispose();
                d_K.Dispose();
                d_L.Dispose();

            }
        }

        private void BtnRun2_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                const int Count = 1024 * 1024;
                using (var state = new SomeState(deviceId: 0))
                {
                    Console.WriteLine("Initializing kernel...");
                    string log;
                    var compileResult = state.LoadKernel(out log);
                    if (compileResult != ManagedCuda.NVRTC.nvrtcResult.Success)
                    {
                        Console.Error.WriteLine(compileResult);
                        Console.Error.WriteLine(log);
                        return;
                    }
                    Console.WriteLine(log);

                    Console.WriteLine("Initializing data...");
                    state.InitializeData(Count);

                    Console.WriteLine("Running kernel...");
                    for (int i = 0; i < 8; i++)
                    {
                        state.MultiplyAsync(2);
                    }

                    Console.WriteLine("Copying data back...");
                    state.CopyToHost(); // note: usually you try to minimize how much you need to
                                        // fetch from the device, as that can be a bottleneck; you should prefer fetching
                                        // minimal aggregate data (counts, etc), or the required pages of data; fetching
                                        // *all* the data works, but should be avoided when possible.

                    Console.WriteLine("Waiting for completion...");
                    state.Synchronize();

                    Console.WriteLine("all done; showing some results");
                    var random = new Random(123456);
                    for (int i = 0; i < 20; i++)
                    {
                        var record = state[random.Next(Count)];
                        Console.WriteLine($"{i}: {nameof(record.Id)}={record.Id}, {nameof(record.Value)}={record.Value}");
                    }

                    Console.WriteLine("Cleaning up...");
                }

                Console.WriteLine("All done; have a nice day");



            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return;
            }
        }

    }

         
    struct SomeBasicType
    {
        // yes, these are public mutable fields; we are explicitly **not**
        // trying to provide abstractions here - we're holding our hands
        // up and saying "you're playing with raw memory, don't screw up"
        public int Id;
        public uint Value;
    }
    unsafe sealed class SomeState : IDisposable
    {
        public void Dispose() => Dispose(true);
        ~SomeState() { Dispose(false); } // we don't want to see this... this means we failed
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this); // dispose was called correctly
            }
            // release the local buffer - note we want to do this while the CUDA context lives
            if (hostBuffer != default(SomeBasicType*))
            {
                var tmp = new IntPtr(hostBuffer);
                hostBuffer = default(SomeBasicType*);
                try
                {
                    DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                }
                catch (Exception ex) { Debug.WriteLine(ex.Message); }
            }

            if (disposing) // clean up managed resources
            {
                Dispose(ref deviceBuffer);
                Dispose(ref defaultStream);
                Dispose(ref ctx);
            }
        }
        // utility method to dispose and wipe fields
        private void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try { field.Dispose(); } catch (Exception ex) { Debug.WriteLine(ex.Message); }
                field = null;
            }
        }

        public SomeState(int deviceId)
        {
            // note that this initializes a lot of things and binds *to the thread*
            ctx = new CudaContext(deviceId, true);

            var props = ctx.GetDeviceInfo();
            defaultBlockCount = props.MultiProcessorCount * 32;
            defaultThreadsPerBlock = props.MaxThreadsPerBlock;
            warpSize = props.WarpSize;
        }

        private int count, defaultBlockCount, defaultThreadsPerBlock, warpSize;
        private SomeBasicType* hostBuffer;
        CudaDeviceVariable<SomeBasicType> deviceBuffer;
        CudaContext ctx;
        CudaStream defaultStream;
        CudaKernel multiply;
        public void InitializeData(int count)
        {
            if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));
            this.count = count;

            // allocate a buffer at the host (meaning: accessible to the CPU)
            // note: for client-side, we *don't* want to just use an array, 
            // as we want the maximum GPU<===>CPU memory transfer speed,
            // which requires fixed pages allocated by the GPU driver
            IntPtr hostPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref hostPointer, count * sizeof(SomeBasicType));
            if (res != CUResult.Success) throw new CudaException(res);
            hostBuffer = (SomeBasicType*)hostPointer;

            // allocate a buffer at the device (meaning: accessible to the GPU)
            deviceBuffer = new CudaDeviceVariable<SomeBasicType>(count);

            // initialize the local data
            for (int i = 0; i < count; i++)
            {
                // we'll just set the key and value to i, so: [{0,0},{1,1},{2,2}...]
                hostBuffer[i].Id = i;
                hostBuffer[i].Value = (uint)i;
            }

            // allocate a stream for async/overlapped operations; note we're only going to use
            // one stream, but in complex code you can use different streams to allow concurrent
            // memory transfer while unrelated kernels execute, etc
            defaultStream = new CudaStream();

            // transfer the local buffer to the server (note that CudaDeviceVariable<T> also exposes
            // various methods to do this, but not this particular set using raw memory and streams)
            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceBuffer.DevicePointer,
                hostPointer, deviceBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        private static int RoundUp(int value, int blockSize)
        {
            if ((value % blockSize) != 0)
            {   // take away the surplus, and add an entire extra block
                value += blockSize - (value % blockSize);
            }
            return value;
        }
        internal void MultiplyAsync(int value)
        {
            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions
            int threadsPerBlock, blockCount;
            if (count <= defaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize); // slight caveat here; if you are using "shuffle" operations, you
                                                            // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            multiply.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            multiply.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            // invoke the kernel
            multiply.RunAsync(defaultStream.Stream, new object[] {
                // note the signature is (N, data, factor)
                count, deviceBuffer.DevicePointer, value
            });
        }

        internal void Synchronize()
        {
            ctx.Synchronize(); // this synchronizes (waits for) **all streams**
            // to synchronize a single stream, use {theStream}.Synchronize();
        }

        internal ManagedCuda.NVRTC.nvrtcResult LoadKernel(out string log)
        {
            //TODO You can change file path
            string path = @"..\..\..\MyCudaPtxRunner\MyKernels.cu";
            ManagedCuda.NVRTC.nvrtcResult result;

            string fileText = File.ReadAllText(path);
            string fileName = System.IO.Path.GetFileNameWithoutExtension(path);
            var headers = new string[0];
            var includeNames = new string[0];

            using (var rtc = new ManagedCuda.NVRTC.CudaRuntimeCompiler(fileText, fileName, headers, includeNames))
            {
                try
                {
                    rtc.Compile(new string[0]); // see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
                    result = ManagedCuda.NVRTC.nvrtcResult.Success;
                }
                catch (ManagedCuda.NVRTC.NVRTCException ex)
                {
                    result = ex.NVRTCError;
                }

                log = rtc.GetLogAsString();

                if (result == ManagedCuda.NVRTC.nvrtcResult.Success)
                {
                    byte[] ptx = rtc.GetPTX();
                    multiply = ctx.LoadKernelFatBin(ptx, "Multiply"); // hard-coded method name from the CUDA kernel
                }
            }
            return result;
        }

        internal void CopyToHost()
        {
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                new IntPtr(hostBuffer), deviceBuffer.DevicePointer, deviceBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        public SomeBasicType this[int index]
        {
            get
            {
                // note: whenever possible, try to avoid doing *per call* range checks; this is purely
                // for illustration; you should prefer working with *blocks* of data in one go
                if (index < 0 || index >= count) throw new IndexOutOfRangeException();
                return hostBuffer[index]; // note: here, the data is copied from localBuffer to the stack
            }
        }


    }

}
