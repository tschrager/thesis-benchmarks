#include <cuda.h>
#include <stdio.h>
#include <helper_timer.h>

#define NUM_TAPS            4       /* number of multiples of g_iNFFT */

//#define DEF_LEN_SPEC        1024
//int g_iNFFT = DEF_LEN_SPEC;
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
#define MIN_NX        256        /* default value for g_iNFFT */
#define MAX_NX        2097152        /* default value for g_iNFFT */
//int g_iNFFT = DEF_LEN_SPEC;

/* what does this mean? */
#define MIN_BATCH   1
#define MAX_BATCH   32
//int g_iNumSubBands = BATCH_SIZE;

#define MAX_DIM 16777216/NUM_TAPS

#define AVERAGED_ITERATIONS 100

#define DEF_SIZE_READ       33554432    /* 32 MB - block size in VEGAS input
                                           buffer */
int g_iSizeRead = DEF_SIZE_READ;

int g_iNTaps = NUM_TAPS;

/* function that performs the PFB */
__global__ void DoPFB(char4 *pc4Data,
                      float4 *pf4FFTIn,
                      float *pfPFBCoeff,
                      int numtaps)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    char4 c4Data = make_char4(0, 0, 0, 0);

    for (j = 0; j < numtaps; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx = (j * iNFFT) + i;
        /* get the address of the block */
        c4Data = pc4Data[iAbsIdx];
        
        f4PFBOut.x += (float) c4Data.x * pfPFBCoeff[iAbsIdx];
        f4PFBOut.y += (float) c4Data.y * pfPFBCoeff[iAbsIdx];
        f4PFBOut.z += (float) c4Data.z * pfPFBCoeff[iAbsIdx];
        f4PFBOut.w += (float) c4Data.w * pfPFBCoeff[iAbsIdx];
    }

    pf4FFTIn[i] = f4PFBOut;

    return;
}

int main()
{
    long long nx;
    long long batch;
    cudaDeviceProp stDevProp = {0};
    cudaGetDeviceProperties(&stDevProp, 0);
    int g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;
    
    
    StopWatchInterface * complete_pfb_timer;
    StopWatchInterface * piecewise_pfb_timer;
    StopWatchInterface * copy_to_gpu_timer;
    StopWatchInterface * pfb_only_timer;
    StopWatchInterface * copy_from_gpu_timer;
    
    sdkCreateTimer(&complete_pfb_timer);
    sdkCreateTimer(&piecewise_pfb_timer);
    sdkCreateTimer(&copy_to_gpu_timer);
    sdkCreateTimer(&pfb_only_timer);
    sdkCreateTimer(&copy_from_gpu_timer);
    
    //fprintf(stderr,"Threads %d", g_iMaxThreadsPerBlock);
    
    
    // 
    // g_pc4InBuf = (char4*) malloc(g_iSizeFile);
    // CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
    //                              g_pc4InBufRead,
    //                              g_iSizeRead,
    //                              cudaMemcpyHostToDevice));
    
    // copy coefficients
    float *g_pfPFBCoeff = NULL;
    g_pfPFBCoeff = (float *) malloc(MAX_DIM
                                    * g_iNTaps
                                    * sizeof(float));
                                    
                                
    
    float *g_pfPFBCoeff_d = NULL;
    cudaMalloc((void **) &g_pfPFBCoeff_d, MAX_DIM * g_iNTaps * sizeof(float));
    cudaError_t error = cudaGetLastError();
       
    cudaMemcpy(g_pfPFBCoeff_d,
               g_pfPFBCoeff,
               MAX_DIM * g_iNTaps * sizeof(float),
               cudaMemcpyHostToDevice);
                
               
    /* allocate input data */
    char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
    char4* g_pc4Data_d = NULL;              /* raw data starting address */
    char4* g_pc4Data = NULL;              /* raw data starting address */
    cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead);
    
    cudaMallocHost((void **) &g_pc4Data, g_iSizeRead);
    g_pc4DataRead_d = g_pc4Data_d;
    
    /* allocate output data */
    float4* g_pf4FFTIn_d = NULL;
    cudaMalloc((void **) &g_pf4FFTIn_d, MAX_DIM * sizeof(float4));
    if(error != cudaSuccess)
          {
              // print the CUDA error message and exit
              printf("CUDA error: %s\n", cudaGetErrorString(error));
              exit(-1);
          }
    
    for(nx=MIN_NX; nx<=MAX_NX; nx=nx*2)
    //for(nx=MAX_NX; nx>=MIN_NX; nx=nx/2)
    {
        fprintf(stderr, "Pts\tbatch\ttotal time x\tsum piecewise x\tcopy to gpu x\tpfb on gpu x\tcopy from gpu x\tdata size\n");
        for(batch=MIN_BATCH;batch<=MAX_BATCH;batch=batch*2)
        //for(batch=MAX_BATCH;batch>=MIN_BATCH;batch=batch/2)
        {
            
            if(/*sizeof(cufftComplex)*nx*batch*2 <= totalGlobalMem/2 &&*/ nx*batch<MAX_DIM)
            {
                g_dimGPFB.x = (batch * nx) / g_iMaxThreadsPerBlock;
                if(g_dimGPFB.x==0)
                {
                    g_dimGPFB.x = 1;
                }

                if (nx < g_iMaxThreadsPerBlock)
                {
                    g_dimBPFB.x = nx;
                }
                else
                {
                    g_dimBPFB.x = g_iMaxThreadsPerBlock;
                }

                sdkResetTimer(&complete_pfb_timer);
                sdkStartTimer(&complete_pfb_timer);
                for(int pfbiter=0;pfbiter<AVERAGED_ITERATIONS;pfbiter++)
                {
                    cudaMemcpy(g_pc4DataRead_d,g_pc4Data,nx*batch*sizeof(float4),cudaMemcpyHostToDevice);
                    DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                                    g_pf4FFTIn_d,
                                                    g_pfPFBCoeff_d,
                                                    NUM_TAPS);
                }
                cudaThreadSynchronize();
                sdkStopTimer(&complete_pfb_timer);
                
                sdkResetTimer(&piecewise_pfb_timer);
                sdkResetTimer(&copy_to_gpu_timer);
                sdkResetTimer(&pfb_only_timer);
                
                sdkStartTimer(&piecewise_pfb_timer);
                for(int pfbiter=0;pfbiter<AVERAGED_ITERATIONS;pfbiter++)
                {
                    sdkStartTimer(&copy_to_gpu_timer);
                    cudaMemcpy(g_pc4DataRead_d,g_pc4Data,nx*batch*sizeof(float4),cudaMemcpyHostToDevice);
                    sdkStopTimer(&copy_to_gpu_timer); 
                    sdkStartTimer(&pfb_only_timer);
                    DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                                    g_pf4FFTIn_d,
                                                    g_pfPFBCoeff_d,
                                                    4);
                    cudaThreadSynchronize();
                    sdkStopTimer(&pfb_only_timer);                                                        
                }
                cudaThreadSynchronize();
                sdkStopTimer(&piecewise_pfb_timer);
                
                fprintf(stderr, "%lld\t%lld\t%f\t%f\t%f\t%f\t%f\t%d\n",
                    nx, batch, 
                    sdkGetTimerValue(&complete_pfb_timer), sdkGetTimerValue(&piecewise_pfb_timer), 
                    sdkGetTimerValue(&copy_to_gpu_timer), 
                    sdkGetTimerValue(&pfb_only_timer), sdkGetTimerValue(&copy_from_gpu_timer), 
                    (sizeof(float4)*nx*batch));
            } 
        }
    }
    
                                    
    
    (void) cudaFree(g_pc4Data_d);
    
    (void) cudaFree(g_pf4FFTIn_d);
    
    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);
    
    return 0;
}
