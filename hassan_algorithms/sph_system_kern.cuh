#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;





	// sph Parameters (stored on both host and device)
	struct sphParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;

		int				szPnts, szHash, szGrid;
		int				stride, pnum;

		float			pradius, psmoothradius;
		float3			pboundmin, pboundmax, pgravity;


		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;

		float Sound_Speed;
		float Shear_Modulus;
		float Young_Modulus;
		float Initial_Density;
		float Poisson_Coef;
		float Delta_X;


		float XSPH_Coef;
		float ArtVis_Coef_1;
		float ArtVis_Coef_2;
		float ArtVis_Coef_3;

		float ArtStr_Coef;
		float DeltaSPH_Coef;

		float Int_Coef;
		float SigYield;
		float JC_A ;
		float JC_B ;
		float JC_C ;
		float JC_N ;
		float JC_M ;
		float JC_Tm ;
		float JC_Tr ;
		float DT;
		float Bulk_Modulus_T;
		float Bulk_Modulus_P;
		float EP_c;

		int				gridAdj[64];
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4


	#ifndef CUDA_KERNEL

		// Declare kernel functions that are available to the host.
		// These are defined in kern.cu, but declared here so host.cu can call them.

		__global__ void insertParticles ( float3* mpos, uint* mgcell, uint* mgndx, int* mgridcnt, int pnum );

		__global__ void countingSortIndex (  uint* mgcell, uint* mgndx, int* mgridoff, uint* mgrid, int pnum );

		__global__ void ComputeConsistency (uint* mgcell, int* mgridcnt, int* mgridoff, uint*  mgrid, float3* mpos, float* mMass, float* mdensity, float3* mL_x,  float3* mL_y, float3* mL_z, float* mSW,float* mSh, float* mType,  int pnum );

		

































		void updateSimParams ( sphParams* cpufp );





		// Prefix Sum
		#include "prefix_sum.cu"
		// NOTE: Template functions must be defined in the header
		template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ float s_data[];
			loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
			prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);
			storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		}
		template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ int s_dataInt [];
			loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
			prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums);
			storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
		}
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);
	#endif


	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295


#endif
