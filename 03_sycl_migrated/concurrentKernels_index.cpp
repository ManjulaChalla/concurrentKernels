// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <CL/sycl.hpp>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <chrono>
using namespace sycl;

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
void clock_block(clock_t *d_o, clock_t clock_count, sycl::nd_item<3> item_ct1) {
 // int i = 0;
  for (int i = item_ct1.get_local_id(2); i < 500000;
       i += item_ct1.get_local_range(2))
  {
    d_o[0]=d_o[0]+i;
  }

}

// Single warp reduction kernel
void sum(clock_t *d_clocks, int N, sycl::nd_item<3> item_ct1, clock_t *s_clocks) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();

  clock_t my_sum = 0;

  for (int i = item_ct1.get_local_id(2); i < N;
       i += item_ct1.get_local_range(2)) {
    my_sum += d_clocks[i];
  }

  s_clocks[item_ct1.get_local_id(2)] = my_sum;
   sycl::nd_item::barrier() 

  for (int i = 16; i > 0; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      s_clocks[item_ct1.get_local_id(2)] +=
          s_clocks[item_ct1.get_local_id(2) + i];
    }

    sycl::nd_item::barrier() 
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
 
  sycl::queue q_ct1 = sycl::queue(default_selector());
  int nkernels = 8;             // number of concurrent kernels
  int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(clock_t);  // number of data bytes
  float kernel_time = 10;                   // time the kernel should run in ms
  float elapsed_time;                       // timing variables
  int cuda_device = 0;

  printf("[%s] - Starting...\n", argv[0]);

  // get number of kernels if overridden on the command line
  if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
    nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
    nstreams = nkernels + 1;
  }

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  std::cout << "Device: "
            << q_ct1.get_device().get_info<sycl::info::device::name>()
            << std::endl;

   std::cout << "> Detected Compute SM "
            << q_ct1.get_device().get_info<sycl::info::device::version>()
            << " hardware with "
            << q_ct1.get_device()
                   .get_info<cl::sycl::info::device::max_compute_units>()
            << " multi-processors" << std::endl;

  // allocate host memory
  clock_t *a = 0;  // pointer to the array data in host memory
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  a = (clock_t *)sycl::malloc_host(nbytes, q_ct1);

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  d_a = (clock_t *)sycl::malloc_device(nbytes, q_ct1);

  // allocate and initialize an array of stream handles
  sycl::queue **streams =
      (sycl::queue **)malloc(nstreams * sizeof(sycl::queue *));

  for (int i = 0; i < nstreams; i++) {
     streams[i] = (sycl::queue *)malloc(nstreams * sizeof(sycl::queue));
    *streams[i] =
        sycl::queue(sycl::default_selector(), property::queue::in_order());
  }

  // create CUDA event handles
  sycl::event start_event, stop_event;
  std::chrono::time_point<std::chrono::steady_clock> start_event_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_event_ct1;
  
  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  sycl::event *kernelEvent;
  std::chrono::time_point<std::chrono::steady_clock> kernelEvent_ct1_i;
  kernelEvent = new sycl::event[nkernels];


  //////////////////////////////////////////////////////////////////////
  // time execution with nkernels streams
  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to
  // prevent hangs reduce time_clocks.
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
  clock_t time_clocks =
      (clock_t)(kernel_time *q_ct1.get_device().get_info<sycl::info::device::max_clock_frequency>());
#endif

  /*
  DPCT1012:0: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  sycl::event stop_event_streams_nstreams_1;
  start_event_ct1 = std::chrono::steady_clock::now();
  start_event = q_ct1.ext_oneapi_submit_barrier();

  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    streams[i]->submit([&](sycl::handler &cgh) {
      auto d_a_i_ct0 = &d_a[i];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            clock_block(d_a_i_ct0, time_clocks, item_ct1);
          });
    });
    total_clocks += time_clocks;
    /*
    DPCT1012:12: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:13: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    kernelEvent_ct1_i = std::chrono::steady_clock::now();
    kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier();

    // make the last stream wait for the kernel event to be recorded
    
    kernelEvent[i] =
             streams[nstreams - 1]->ext_oneapi_submit_barrier({kernelEvent[i]});
  }
  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  streams[nstreams - 1]->submit([&](sycl::handler &cgh) {
    sycl::accessor<clock_t, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        s_clocks_acc_ct1(sycl::range<1>(32), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          sum(d_a, nkernels, item_ct1, s_clocks_acc_ct1.get_pointer());
        });
  });
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  stop_event_streams_nstreams_1 = streams[nstreams - 1]->memcpy(a, d_a, sizeof(clock_t));

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // in this sample we just wait until the GPU is done
  /*
  DPCT1012:16: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:17: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  q_ct1.wait_and_throw();
  stop_event_streams_nstreams_1.wait();
  stop_event_ct1 = std::chrono::steady_clock::now();
  stop_event = q_ct1.ext_oneapi_submit_barrier();
  
  /*
  DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  elapsed_time = std::chrono::duration<float, std::milli>(
                                      stop_event_ct1 - start_event_ct1)
                                      .count();

  printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
         nkernels * kernel_time / 1000.0f);
  printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
         nkernels, kernel_time / 1000.0f);
  printf("Measured time for sample = %.5fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] > total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++) {
     free(streams[i]);
  }

  free(streams);
  delete[] kernelEvent;

  /*
  DPCT1026:1: The call to cudaEventDestroy was removed because this call is
  redundant in SYCL.
  */
  /*
  DPCT1026:2: The call to cudaEventDestroy was removed because this call is
  redundant in SYCL.
  */
  sycl::free(a, q_ct1);
  sycl::free(d_a, q_ct1);

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
