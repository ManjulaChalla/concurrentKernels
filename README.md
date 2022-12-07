﻿# `concurrentkernels` Sample

The `concurrentKernels`sample demonstrates the use of SYCL queues for concurrent execution of several kernels on GPU device.It is implemented using SYCL* by migrating code from original CUDA source code for offloading computations to a GPU/CPU and further demonstrates how to optimize and improve processing time.

| Property               | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes

>**Note**: This sample is based on the [concurrentKernels](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

Optical flow method is based on two assumptions: brightness constancy and spatial
flow smoothness. These assumptions are combined in a single energy functional and
solution is found as its minimum point. The sample includes
both parallel and serial computation, which allows for direct results comparison. The parallel implementation demonstrates the use of Texture memory, shared memory, and cooperative groups. Input to the sample is two image frames and output is the absolute difference value(L1 error) between serial and parallel computation.

This sample contains four versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_sycl_dpct_output`         | Contains output of Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_dpct_migrated`       | Contains Intel® DPC++ Compatibility Tool migrated SYCL code from CUDA code with manual changes done to fix the unmigrated code to work functionally.
| `03_sycl_migrated`            | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Property              | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

HSOptical flow involves image downscaling and upscaling, image warping, computing derivatives, and computation of jacobi iteration.

>**Note**: This sample demonstrates the CUDA HSOptical Flow Estimation using key concepts such as Image processing, Texture memory, shared memory, and cooperative groups.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `concurrentKernel` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `02_sycl_dpct_migrated`, `03_sycl_migrated`, and `04_sycl_migrated_optimized` versions of the program.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `concurrentKernel` Sample

### On Linux

You can run the programs for CPU and GPU. The commands indicate the device target.

1. Run `02_sycl_dpct_migrated` for CPU and GPU.
    ```
    make run_sdm_cpu
    make run_sdm_gpu
    ```

2. Run `03_sycl_migrated` for CPU and GPU.
    ```
    make run_cpu
    make run_gpu
    ```

### Build and Run the `concurrentKernel` Sample in Intel® DevCloud

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`.

1. Open a terminal on a Linux* system.
2. Log in to Intel® DevCloud.
    ```
    ssh devcloud
    ```
3. Download the samples.
    ```
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
4. Change to the sample directory.
5. Configure the sample for a GPU node.
   ```
   qsub  -I  -l nodes=1:gpu:ppn=2 -d .
   ```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

     |Available Nodes  |Command Options
     |:---             |:---
     | GPU	         |`qsub -l nodes=1:gpu:ppn=2 -d .`
     | CPU	         |`qsub -l nodes=1:xeon:ppn=2 -d .`

6. Perform build steps as you would on Linux.
7. Run the programs.
8. Clean up the project files.
    ```
    make clean
    ```
9. Disconnect from the Intel® DevCloud.
    ```
    exit
    ```

### Example Output

The following example is for `03_sycl_migrated` for GPU with inputs files **frame10.ppm** and **frame11.ppm** on **Intel(R) UHD Graphics P630 \[0x3e96\]**.
```
HSOpticalFlow Starting...

Loading "frame10.ppm" ...
Loading "frame11.ppm" ...
Computing optical flow on CPU...
Computing optical flow on GPU...

Running on Intel(R) UHD Graphics P630 [0x3e96]
Processing time on CPU: 2800.426270 (ms)
Processing time on GPU: 1203.821533 (ms)
L1 error : 0.018189
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
