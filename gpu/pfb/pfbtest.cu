#define NUM_TAPS            8       /* number of multiples of g_iNFFT */

dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);

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
    
    DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                    g_pf4FFTIn_d,
                                    g_pfPFBCoeff_d);
    
    
    return 0;
}
