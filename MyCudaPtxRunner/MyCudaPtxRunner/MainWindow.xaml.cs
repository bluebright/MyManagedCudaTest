using ManagedCuda;
using ManagedCuda.BasicTypes;
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

                CudaKernel kernel_4 = ctx.LoadKernel("MyPtx2Kernel_2.ptx", "matrixMultKernel");
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

                
                int[,] mat_A = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
              
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

                //See https://github.com/kunzmi/managedCuda/issues/1
                CudaContext.ProfilerStop();
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

        private void BtnRun3_Click(object sender, RoutedEventArgs e)
        {
            try
            {

                int[] a = { 1, 2, 3, 4, 5 };
                int[] b = { 110, 220, 330, 440, 550 };
                int[] c = new int[5];

                ManagedCu.MyCudaCliWrap obj = new ManagedCu.MyCudaCliWrap();

                obj.RunAdd(c, a, b, 5);

                foreach (int i in c)
                {
                    System.Console.Write("{0} ", i);
                }

                Console.WriteLine();

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

        }
    }



}
