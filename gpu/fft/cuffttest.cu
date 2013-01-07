#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
//#include <cutil_inline.h>
#include <helper_timer.h>


//#define NX      256
//#define BATCH   4
#define MIN_NX      256
#define MAX_NX      16777216
#define MIN_BATCH   1
#define MAX_BATCH   4096
#define MAX_DIM     16777216*4
//#define MAX_DIM     16777216


static cufftHandle plan;
cufftComplex *gpudata;
cufftComplex *fftgpudata;


int main ()
{
    //int deviceCount = 0;
    CUresult err = cuInit(0);
    //CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));
    //printf("There are %d devices supporting CUDA\n", deviceCount);
    
    long long i;
    long long nx;
    long long batch;
    StopWatchInterface * complete_fft_timer;
    StopWatchInterface * complete_fft_timer2;
    StopWatchInterface * piecewise_fft_timer;
    StopWatchInterface * copy_to_gpu_timer;
    StopWatchInterface * fft_only_timer;
    StopWatchInterface * copy_from_gpu_timer;
    
    sdkCreateTimer(&complete_fft_timer);
    sdkCreateTimer(&complete_fft_timer2);
    sdkCreateTimer(&piecewise_fft_timer);
    sdkCreateTimer(&copy_to_gpu_timer);
    sdkCreateTimer(&fft_only_timer);
    sdkCreateTimer(&copy_from_gpu_timer);
    
    //cufftHandle plan;
    cufftComplex *data;
    cufftComplex *result;
    //cufftComplex *gpudata;
    
    cudaMallocHost(&data, sizeof(cufftComplex)*MAX_DIM);
    cudaMallocHost(&result, sizeof(cufftComplex)*MAX_DIM);
    //cudaHostAlloc(&data, sizeof(cufftComplex)*MAX_DIM,cudaHostAllocWriteCombined);
    //cudaHostAlloc(&result, sizeof(cufftComplex)*MAX_DIM,cudaHostAllocWriteCombined);
    
    // get the total global memory of the device
    CUdevice dev=0;
    size_t totalGlobalMem;
	cuDeviceTotalMem(&totalGlobalMem, dev);
    fprintf(stderr,"Total amount of global memory: %u bytes\n", totalGlobalMem);
    
    //fprintf(stderr, "Initializing data... ");
    // generate some random data
    for(i=0; i<MAX_DIM; i++)
    {
        data[i].x=1.0f;
        data[i].y=1.0f;
    }
    //fprintf(stderr, "done\n");

    fprintf(stderr, "Testing cufft C2C 1D fft with cudaMallocHost allocated memory\n");
    //fprintf(stderr, "nx\tbatch\ttime\tcopy_to_gpu\tactual_fft\tcopy_from_gpu\tavg\n");
    for(nx=MIN_NX; nx<=MAX_NX; nx=nx*2)
    //for(nx=4096; nx<=MAX_NX; nx+=4096)
    {
        fprintf(stderr, "Pts\tbatch\ttotal time\ttotal time 2\tsum piecewise\tcopy to gpu\tfft on gpu\tcopy from gpu\ttimes reported in ms\n");
        for(batch=MIN_BATCH;batch<=MAX_BATCH;batch=batch*2)
        {
            if(/*sizeof(cufftComplex)*nx*batch*2 <= totalGlobalMem/2 &&*/ nx*batch<MAX_DIM)
            {  
                //fprintf(stderr, "Allocating %lld bytes\n", sizeof(cufftComplex)*nx*batch*2);
                // allocate device memory for the fft
                cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*nx*batch);
                cudaMalloc((void**)&fftgpudata,sizeof(cufftComplex)*nx*batch);

                cufftPlan1d(&plan,nx,CUFFT_C2C, batch);
                cudaThreadSynchronize();
                
                
                // run the fft
                // allocate device memory and copy over data
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cudaThreadSynchronize();
                // run the fft
                sdkResetTimer(&complete_fft_timer);
                sdkStartTimer(&complete_fft_timer);
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cudaThreadSynchronize();
                sdkStopTimer(&complete_fft_timer);
                // copy the result back
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                
                
                
                
                // run the fft
                // allocate device memory and copy over data
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cudaThreadSynchronize();
                // run the fft
                sdkResetTimer(&complete_fft_timer2);
                sdkStartTimer(&complete_fft_timer2);
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cudaThreadSynchronize();
                sdkStopTimer(&complete_fft_timer2);
                // copy the result back
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                
                
                
                sdkResetTimer(&piecewise_fft_timer);
                sdkStartTimer(&piecewise_fft_timer);
                
                sdkResetTimer(&copy_to_gpu_timer);
                sdkStartTimer(&copy_to_gpu_timer);
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cudaThreadSynchronize();
                sdkStopTimer(&copy_to_gpu_timer);
                
                sdkResetTimer(&fft_only_timer);
                sdkStartTimer(&fft_only_timer);
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cudaThreadSynchronize();
                sdkStopTimer(&fft_only_timer);
                
                sdkResetTimer(&copy_from_gpu_timer);
                sdkStartTimer(&copy_from_gpu_timer);
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                sdkStopTimer(&copy_from_gpu_timer);
                
                sdkStopTimer(&piecewise_fft_timer);
                
                cufftDestroy(plan);
                cudaFree(gpudata);
                cudaFree(fftgpudata);
                cudaThreadSynchronize();
            
                
                fprintf(stderr, "%lld\t%lld\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\n",
                    nx, batch, 
                    sdkGetTimerValue(&complete_fft_timer), sdkGetTimerValue(&complete_fft_timer2), sdkGetTimerValue(&piecewise_fft_timer), 
                    sdkGetTimerValue(&copy_to_gpu_timer), 
                    sdkGetTimerValue(&fft_only_timer), sdkGetTimerValue(&copy_from_gpu_timer),
                    /*sdkGetTimerValue(&complete_fft_timer)/(nx*batch),*/ sizeof(cufftComplex)*nx*batch);
            }
        }
    }
    
    //print fft data
//    for(i=0; i<NX*BATCH; i++)
//    {
//        fprintf(stderr,"%d %f %f\n", i, data[i].x, data[i].y);
//    }
    
    
    return 0;
}
