#include <cuda.h>
#include <stdio.h>

#define NUM_TAPS            8       /* number of multiples of g_iNFFT */

//#define DEF_LEN_SPEC        1024
//int g_iNFFT = DEF_LEN_SPEC;
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
#define DEF_LEN_SPEC        1024        /* default value for g_iNFFT */
int g_iNFFT = DEF_LEN_SPEC;

/* what does this mean? */
#define DEF_NUM_SUBBANDS    8
int g_iNumSubBands = DEF_NUM_SUBBANDS;

#define DEF_SIZE_READ       33554432    /* 32 MB - block size in VEGAS input
                                           buffer */
int g_iSizeRead = DEF_SIZE_READ;

int g_iNTaps = NUM_TAPS;

/* function that performs the PFB */
__global__ void DoPFB(char4 *pc4Data,
                      float4 *pf4FFTIn,
                      float *pfPFBCoeff)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    char4 c4Data = make_char4(0, 0, 0, 0);

    for (j = 0; j < NUM_TAPS; ++j)
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
    cudaDeviceProp stDevProp = {0};
    cudaGetDeviceProperties(&stDevProp, 0);
    int g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;
    //fprintf(stderr,"Threads %d", g_iMaxThreadsPerBlock);
    
    /* allocate input data */
    char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
    char4* g_pc4Data_d = NULL;              /* raw data starting address */
    cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead);
    g_pc4DataRead_d = g_pc4Data_d;
    
    /* allocate output data */
    float4* g_pf4FFTIn_d = NULL;
    cudaMalloc((void **) &g_pf4FFTIn_d, g_iNumSubBands * g_iNFFT * sizeof(float4));
    
    // 
    // g_pc4InBuf = (char4*) malloc(g_iSizeFile);
    // CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
    //                              g_pc4InBufRead,
    //                              g_iSizeRead,
    //                              cudaMemcpyHostToDevice));
    
    
    float *g_pfPFBCoeff = NULL;
    g_pfPFBCoeff = (float *) malloc(g_iNumSubBands
                                    * g_iNTaps
                                    * g_iNFFT
                                    * sizeof(float));
    
    float *g_pfPFBCoeff_d = NULL;
    cudaMalloc((void **) &g_pfPFBCoeff_d, g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float));
    cudaMemcpy(g_pfPFBCoeff_d,
               g_pfPFBCoeff,
               g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float),
               cudaMemcpyHostToDevice);
    
    
    g_dimGPFB.x = (g_iNumSubBands * g_iNFFT) / g_iMaxThreadsPerBlock;
    
    if (g_iNFFT < g_iMaxThreadsPerBlock)
    {
        g_dimBPFB.x = g_iNFFT;
    }
    else
    {
        g_dimBPFB.x = g_iMaxThreadsPerBlock;
    }
    
    DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                    g_pf4FFTIn_d,
                                    g_pfPFBCoeff_d);
    g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
    
    (void) cudaFree(g_pc4Data_d);
    
    (void) cudaFree(g_pf4FFTIn_d);
    
    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);
    
    return 0;
}
