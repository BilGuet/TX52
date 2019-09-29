

#ifndef _SCAN_BEST_KERNEL_CU_
#define _SCAN_BEST_KERNEL_CU_

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////


template <bool isNP2> __device__ void loadSharedChunkFromMem (float *s_data, const float *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);		    // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
	s_data[ai + bankOffsetA] = g_idata[mem_ai];		// Cache the computational window in shared memory pad values beyond n with zeros
    
    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}


template <bool isNP2> __device__ void loadSharedChunkFromMemInt (int *s_data, const int *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);		    // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
	s_data[ai + bankOffsetA] = g_idata[mem_ai];		// Cache the computational window in shared memory pad values beyond n with zeros
    
    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}

template <bool isNP2> __device__ void storeSharedChunkToMem(float* g_odata, const float* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];			// write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}
template <bool isNP2> __device__ void storeSharedChunkToMemInt (int* g_odata, const int* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];			// write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}


template <bool storeSum> __device__ void clearLastElement( float* s_data, float *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);        
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;		// zero the last element in the scan so it will propagate back to the front
    }
}

template <bool storeSum> __device__ void clearLastElementInt ( int* s_data, int *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);        
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;		// zero the last element in the scan so it will propagate back to the front
    }
}


__device__ unsigned int buildSum(float *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}
__device__ unsigned int buildSumInt (int *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}

__device__ void scanRootToLeaves(float *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

__device__ void scanRootToLeavesInt (int *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum> __device__ void prescanBlock(float *data, int blockIndex, float *blockSums)
{
    int stride = buildSum (data);               // build the sum in place up the tree
    clearLastElement<storeSum> (data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves (data, stride);            // traverse down tree to build the scan 
}
template <bool storeSum> __device__ void prescanBlockInt (int *data, int blockIndex, int *blockSums)
{
    int stride = buildSumInt (data);               // build the sum in place up the tree
    clearLastElementInt <storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeavesInt (data, stride);            // traverse down tree to build the scan 
}

__global__ void uniformAdd (float *g_data, float *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ float uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}
__global__ void uniformAddInt (int *g_data, int *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ int uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}


#endif // #ifndef _SCAN_BEST_KERNEL_CU_

