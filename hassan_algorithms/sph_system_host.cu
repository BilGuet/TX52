
#include "cutil_math.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <thrust/sort.h>
#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include "sph_system_host.cuh"
#include "sph_system_kern.cuh"

sphParams		fcuda;		// CPU sph params
sphParams*	mcuda;		// GPU sph params
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		float3*			mpos;
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;
		float*			mMass;

		uint*			mgcell;
		uint*			mgndx;
		uint*			mgrid;

		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;

		uint*			mclr;			// 4 byte color

		uint*			mcluster;

		char*			msortbuf;



		float* 		mEnergy;
		float* 		mdEnergy;
		float* 		mdEnergyP;

		float*			mEnergyC;

		float* 		mS_xx;
 		float* 		mS_yy;
		float* 		mS_zz;
		float* 		mS_xy;
		float* 		mS_xz;
		float* 		mS_yz;

		float* 		mSig_xx;
		float* 		mSig_yy;
		float* 		mSig_zz;
		float* 		mSig_xy;
		float* 		mSig_xz;
		float* 		mSig_yz;

		float*  		mEps_xx;
		float*  		mEps_yy;
		float*  		mEps_zz;
		float*  		mEps_xy;
		float*  		mEps_xz;
		float*  		mEps_yz;

		float*  		mRot_xy;
		float*  		mRot_xz;
		float*  		mRot_yz;

		float* 		mdS_xx;
 		float* 		mdS_yy;
		float* 		mdS_zz;
		float* 		mdS_xy;
		float* 		mdS_xz;
		float* 		mdS_yz;

		float* 		mdS_xxP;
 		float* 		mdS_yyP;
		float* 		mdS_zzP;
		float* 		mdS_xyP;
		float* 		mdS_xzP;
		float* 		mdS_yzP;

		float*			mdDensity;
		float*			mdDensityP;

		float3* 		mL_x;
		float3* 		mL_y;
		float3* 		mL_z;
		float*			mSW;

		float4* 		mArtVisc;
		float3* 		mXSPH;
		float3* 		mXSPHP;
		float3* 		mDeltaSPH;

		float3*		mforceP;
		float*			mType;

		float3* 		mDeltaPos;

                float*                 mSh;

                float*                 mEffP;

                float*                 mT;

		float* 		mSigYield;


		float*						mFix;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





bool cudaCheck ( cudaError_t status, char* msg )
{
	if ( status != cudaSuccess ) {
		printf ( "CUDA ERROR: %s\n", cudaGetErrorString ( status ) );

		return false;
	} else {
		printf ( "%s. OK.\n", msg );
	}
	return true;
}


void cudaExit ()
{
	int argc = 1;
	char* argv[] = {"sphs"};

	cudaDeviceReset();
}

// Initialize CUDA
void cudaInit()
{
	int argc = 1;
	char* argv[] = {"sphs"};

	int count = 0;
	int i = 0;

	cudaError_t err = cudaGetDeviceCount(&count);
	if ( err==cudaErrorInsufficientDriver) { printf( "CUDA driver not installed.\n"); }
	if ( err==cudaErrorNoDevice) { printf ( "No CUDA device found.\n"); }
	if ( count == 0) { printf ( "No CUDA device found.\n"); }

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			if(prop.major >= 1) break;
	}
	if(i == count) { printf ( "No CUDA device found.\n");  }
	cudaSetDevice(i);

	printf( "CUDA initialized.\n");

	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);

	printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );

	mgridactive = 0x0;

	// Allocate the sim parameters
	cudaCheck ( cudaMalloc ( (void**) &mcuda, sizeof(sphParams) ),		"Malloc sphParams mcuda" );

	// Allocate particle buffers
	cudaCheck ( cudaMalloc ( (void**) &mpos, sizeof(float3) ),		"Malloc mpos" );
	cudaCheck ( cudaMalloc ( (void**) &mvel, sizeof(float3)),			"Malloc mvel" );
	cudaCheck ( cudaMalloc ( (void**) &mveleval, sizeof(float3)),		"Malloc mveleval"  );
	cudaCheck ( cudaMalloc ( (void**) &mforce, sizeof(float3)),		"Malloc mforce"  );
	cudaCheck ( cudaMalloc ( (void**) &mpress, sizeof(float) ),		"Malloc mpress"  );
	cudaCheck ( cudaMalloc ( (void**) &mdensity, sizeof(float) ),		"Malloc mdensity"  );
	cudaCheck ( cudaMalloc ( (void**) &mMass, sizeof(float) ),		"Malloc mMass"  );
	cudaCheck ( cudaMalloc ( (void**) &mgcell, sizeof(uint)),			"Malloc mgcell"  );
	cudaCheck ( cudaMalloc ( (void**) &mgndx, sizeof(uint)),			"Malloc mgndx"  );
	cudaCheck ( cudaMalloc ( (void**) &mclr, sizeof(uint)),			"Malloc mclr"  );

	cudaCheck ( cudaMalloc ( (void**) &msortbuf, sizeof(uint) ),		"Malloc msortbu" );

	cudaCheck ( cudaMalloc ( (void**) &mgrid, 1 ),						"Malloc mgrid"  );
	cudaCheck ( cudaMalloc ( (void**) &mgridcnt, 1 ),					"Malloc mgridcnt"  );
	cudaCheck ( cudaMalloc ( (void**) &mgridoff, 1 ),					"Malloc mgridoff" );
	cudaCheck ( cudaMalloc ( (void**) &mgridactive, 1 ),				"Malloc mgridactive");


};

// Compute number of blocks to create
int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( maxThreads, numPnts );
    numBlocks = iDivUp ( numPnts, numThreads );
}

void sphClearCUDA ()
{
	cudaCheck ( cudaFree ( mpos ),			"Free mpos" );
	cudaCheck ( cudaFree ( mvel ),			"Free mvel" );
	cudaCheck ( cudaFree ( mveleval ),		"Free mveleval" );
	cudaCheck ( cudaFree ( mforce ),		"Free mforce" );
	cudaCheck ( cudaFree ( mpress ),		"Free mpress");
	cudaCheck ( cudaFree ( mdensity ),		"Free mdensity" );
	cudaCheck ( cudaFree ( mgcell ),		"Free mgcell" );
	cudaCheck ( cudaFree ( mgndx ),		"Free mgndx" );
	cudaCheck ( cudaFree ( mclr ),			"Free mclr" );


	cudaCheck ( cudaFree ( msortbuf ),		"Free msortbuf" );

	cudaCheck ( cudaFree ( mgrid ),		"Free mgrid" );
	cudaCheck ( cudaFree ( mgridcnt ),		"Free mgridcnt" );
	cudaCheck ( cudaFree ( mgridoff ),		"Free mgridoff" );
	cudaCheck ( cudaFree ( mgridactive ),	"Free mgridactive" );
}


void sphSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk, float Initial_Density,float  Young_Modulus , float Poisson_Coef, float Shear_Modulus, float Sound_Speed, float Delta_X, float XSPH_Coef, float  ArtVis_Coef_1, float ArtVis_Coef_2, float ArtVis_Coef_3, float ArtStr_Coef, float DeltaSPH_Coef, float SigYield, float dt )
{
	fcuda.pnum = num;
	fcuda.gridRes = res;
	fcuda.gridSize = size;
	fcuda.gridDelta = delta;
	fcuda.gridMin = gmin;
	fcuda.gridMax = gmax;
	fcuda.gridTotal = total;
	fcuda.gridSrch = gsrch;
	fcuda.gridAdjCnt = gsrch*gsrch*gsrch;
	fcuda.gridScanMax = res;
	fcuda.gridScanMax -= make_int3( fcuda.gridSrch, fcuda.gridSrch, fcuda.gridSrch );

////////////////////////////////////////////////////////////////////////////
	fcuda.Initial_Density=Initial_Density;
	fcuda.Young_Modulus = Young_Modulus;
	fcuda.Poisson_Coef = Poisson_Coef;
	fcuda.Shear_Modulus = Shear_Modulus ;
	fcuda.Sound_Speed = Sound_Speed;

	fcuda.Delta_X = Delta_X;
	fcuda.XSPH_Coef = XSPH_Coef;

	fcuda.ArtVis_Coef_1 = ArtVis_Coef_1;
	fcuda.ArtVis_Coef_2 = ArtVis_Coef_2;
	fcuda.ArtVis_Coef_3 = ArtVis_Coef_3;

	fcuda.ArtStr_Coef = ArtStr_Coef;

	fcuda.DeltaSPH_Coef = DeltaSPH_Coef;
	fcuda.Int_Coef = int(0.001/dt)+1;
	fcuda.SigYield = SigYield;
	fcuda.DT = dt;

	fcuda.Bulk_Modulus_T = 169.1e9;//Young_Modulus/(3.0*(1.0-Poisson_Coef));
	fcuda.Bulk_Modulus_P = 169.1e9;//(Young_Modulus+5.0e9)/(3.0*(1.0-Poisson_Coef));


	fcuda.JC_A = 792.0e6;
	fcuda.JC_B = 510.0e6;
	fcuda.JC_C = 0.014;
	fcuda.JC_N = 0.26;
	fcuda.JC_M = 1.03;
	fcuda.JC_Tm = 1293.0;
	fcuda.JC_Tr = 293.0;

	fcuda.EP_c = 0.3;
////////////////////////////////////////////////////////////////////////////


	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ )
		for (int z=0; z < gsrch; z++ )
			for (int x=0; x < gsrch; x++ )
				fcuda.gridAdj [ cell++]  = ( y * fcuda.gridRes.z+ z )*fcuda.gridRes.x +  x ;

	/*printf ( "CUDA Adjacency Table\n");
	for (int n=0; n < fcuda.gridAdjCnt; n++ ) {
		printf ( "  ADJ: %d, %d\n", n, fcuda.gridAdj[n] );
	}*/

	// Compute number of blocks and threads

	int threadsPerBlock = 192;

    computeNumBlocks ( fcuda.pnum, threadsPerBlock, fcuda.numBlocks, fcuda.numThreads);				// particles
    computeNumBlocks ( fcuda.gridTotal, threadsPerBlock, fcuda.gridBlocks, fcuda.gridThreads);		// grid cell

	// Allocate particle buffers
    fcuda.szPnts = (fcuda.numBlocks  * fcuda.numThreads);
    //printf ( "CUDA Allocate: \n" );
	//printf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", fcuda.pnum, fcuda.numBlocks, fcuda.numThreads, fcuda.numBlocks*fcuda.numThreads, fcuda.szPnts);
    //printf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", fcuda.gridTotal, fcuda.gridBlocks, fcuda.gridThreads, fcuda.gridBlocks*fcuda.gridThreads, fcuda.szGrid, (int) fcuda.gridRes.x, (int) fcuda.gridRes.y, (int) fcuda.gridRes.z );

	cudaCheck ( cudaMalloc ( (void**) &mpos,		fcuda.szPnts*sizeof(float3) ),	"Malloc mpos" );
	cudaCheck ( cudaMalloc ( (void**) &mvel,		fcuda.szPnts*sizeof(float3) ),	"Malloc mvel" );
	cudaCheck ( cudaMalloc ( (void**) &mveleval,	fcuda.szPnts*sizeof(float3) ),	"Malloc mveleval" );
	cudaCheck ( cudaMalloc ( (void**) &mforce,	fcuda.szPnts*sizeof(float3) ),		"Malloc mforce" );
	cudaCheck ( cudaMalloc ( (void**) &mpress,	fcuda.szPnts*sizeof(float) ),		"Malloc mpress" );
	cudaCheck ( cudaMalloc ( (void**) &mdensity,	fcuda.szPnts*sizeof(float) ),	"Malloc mdensity" );
	cudaCheck ( cudaMalloc ( (void**) &mMass,		fcuda.szPnts*sizeof(float) ),	"Malloc mMass" );
	cudaCheck ( cudaMalloc ( (void**) &mgcell,	fcuda.szPnts*sizeof(uint) ),		"Malloc mgcell" );
	cudaCheck ( cudaMalloc ( (void**) &mgndx,		fcuda.szPnts*sizeof(uint)),		"Malloc mgndx" );
	cudaCheck ( cudaMalloc ( (void**) &mclr,		fcuda.szPnts*sizeof(uint) ),	"Malloc mclr" );


	cudaCheck ( cudaMalloc ( (void**) &mforceP,	fcuda.szPnts*sizeof(float3) ),		"Malloc mforceP" );

	cudaCheck ( cudaMalloc ( (void**) &mdDensity,		fcuda.szPnts*sizeof(float) ),	"Malloc mdDensity" );
	cudaCheck ( cudaMalloc ( (void**) &mdDensityP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdDensityP" );

	cudaCheck ( cudaMalloc ( (void**) &mS_xx,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_xx" );
	cudaCheck ( cudaMalloc ( (void**) &mS_yy,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_yy" );
	cudaCheck ( cudaMalloc ( (void**) &mS_zz,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_zz" );
	cudaCheck ( cudaMalloc ( (void**) &mS_xy,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_xy" );
	cudaCheck ( cudaMalloc ( (void**) &mS_xz,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_xz" );
	cudaCheck ( cudaMalloc ( (void**) &mS_yz,		fcuda.szPnts*sizeof(float) ),	"Malloc mS_yz" );

	cudaCheck ( cudaMalloc ( (void**) &mdS_xx,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xx" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_yy,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_yy" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_zz,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_zz" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_xy,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xy" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_xz,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xz" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_yz,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_yz" );

	cudaCheck ( cudaMalloc ( (void**) &mdS_xxP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xxP" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_yyP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_yyP" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_zzP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_zzP" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_xyP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xyP" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_xzP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_xzP" );
	cudaCheck ( cudaMalloc ( (void**) &mdS_yzP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdS_yzP" );

	cudaCheck ( cudaMalloc ( (void**) &mSig_xx,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_xx" );
	cudaCheck ( cudaMalloc ( (void**) &mSig_yy,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_yy" );
	cudaCheck ( cudaMalloc ( (void**) &mSig_zz,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_zz" );
	cudaCheck ( cudaMalloc ( (void**) &mSig_xy,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_xy" );
	cudaCheck ( cudaMalloc ( (void**) &mSig_xz,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_xz" );
	cudaCheck ( cudaMalloc ( (void**) &mSig_yz,		fcuda.szPnts*sizeof(float) ),	"Malloc mSig_yz" );

	cudaCheck ( cudaMalloc ( (void**) &mEps_xx,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_xx" );
	cudaCheck ( cudaMalloc ( (void**) &mEps_yy,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_yy" );
	cudaCheck ( cudaMalloc ( (void**) &mEps_zz,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_zz" );
	cudaCheck ( cudaMalloc ( (void**) &mEps_xy,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_xy" );
	cudaCheck ( cudaMalloc ( (void**) &mEps_xz,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_xz" );
	cudaCheck ( cudaMalloc ( (void**) &mEps_yz,		fcuda.szPnts*sizeof(float) ),	"Malloc mEps_yz" );

	cudaCheck ( cudaMalloc ( (void**) &mRot_xy,		fcuda.szPnts*sizeof(float) ),	"Malloc mRot_xy" );
	cudaCheck ( cudaMalloc ( (void**) &mRot_xz,		fcuda.szPnts*sizeof(float) ),	"Malloc mRot_xz" );
	cudaCheck ( cudaMalloc ( (void**) &mRot_yz,		fcuda.szPnts*sizeof(float) ),	"Malloc mRot_yz" );

	cudaCheck ( cudaMalloc ( (void**) &mL_x,		fcuda.szPnts*sizeof(float3) ),	"Malloc mL_x" );
	cudaCheck ( cudaMalloc ( (void**) &mL_y,		fcuda.szPnts*sizeof(float3) ),	"Malloc mL_y" );
	cudaCheck ( cudaMalloc ( (void**) &mL_z,		fcuda.szPnts*sizeof(float3) ),	"Malloc mL_z" );

	cudaCheck ( cudaMalloc ( (void**) &mSW,		fcuda.szPnts*sizeof(float) ),	"Malloc mSW" );


	cudaCheck ( cudaMalloc ( (void**) &mArtVisc,		fcuda.szPnts*sizeof(float4) ),	"Malloc mArtVisc" );
	cudaCheck ( cudaMalloc ( (void**) &mXSPH,		fcuda.szPnts*sizeof(float3) ),	"Malloc mXSPH" );
	cudaCheck ( cudaMalloc ( (void**) &mXSPHP,		fcuda.szPnts*sizeof(float3) ),	"Malloc mXSPHP" );
	cudaCheck ( cudaMalloc ( (void**) &mDeltaSPH,		fcuda.szPnts*sizeof(float3) ),	"Malloc mDeltaSPH" );
	cudaCheck ( cudaMalloc ( (void**) &mDeltaPos,		fcuda.szPnts*sizeof(float3) ),	"Malloc mDeltaSPH" );


	cudaCheck ( cudaMalloc ( (void**) &mEnergy,		fcuda.szPnts*sizeof(float) ),	"Malloc mEnergy" );
	cudaCheck ( cudaMalloc ( (void**) &mdEnergy,		fcuda.szPnts*sizeof(float) ),	"Malloc mdEnergy" );
	cudaCheck ( cudaMalloc ( (void**) &mdEnergyP,		fcuda.szPnts*sizeof(float) ),	"Malloc mdEnergy" );

	cudaCheck ( cudaMalloc ( (void**) &mEnergyC,		fcuda.szPnts*sizeof(float) ),	"Malloc mEnergyC" );

	cudaCheck ( cudaMalloc ( (void**) &mType,		fcuda.szPnts*sizeof(float) ),	"Malloc mType" );
	cudaCheck ( cudaMalloc ( (void**) &mSh,		fcuda.szPnts*sizeof(float) ),	"Malloc mSh" );
        cudaCheck ( cudaMalloc ( (void**) &mEffP,		fcuda.szPnts*sizeof(float) ),	"Malloc mEffP" );
        cudaCheck ( cudaMalloc ( (void**) &mT,		fcuda.szPnts*sizeof(float) ),	"Malloc mT" );
	cudaCheck ( cudaMalloc ( (void**) &mSigYield,		fcuda.szPnts*sizeof(float) ),	"Malloc mSigYield" );
	cudaCheck ( cudaMalloc ( (void**) &mFix,		fcuda.szPnts*sizeof(float) ),	"Malloc mFix" );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	cudaCheck ( cudaMemset ( mL_x,0	,	fcuda.szPnts*sizeof(float3) ),	"Memset mL_x" );
	cudaCheck ( cudaMemset ( mL_y,0	,	fcuda.szPnts*sizeof(float3) ),	"Memset mL_y" );
	cudaCheck ( cudaMemset ( mL_z,0	,	fcuda.szPnts*sizeof(float3) ),	"Memset mL_z" );

	
	cudaCheck ( cudaMemset ( mSW,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mSW" );
	cudaCheck ( cudaMemset ( mdDensityP,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mdDensityP" );
	cudaCheck ( cudaMemset ( mdDensity,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mdDensity" );

	cudaCheck ( cudaMemset ( mEnergy,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mEnergy" );
	cudaCheck ( cudaMemset ( mdEnergy,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mdEnergy" );
	cudaCheck ( cudaMemset ( mdEnergyP,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mdEnergyP" );
	cudaCheck ( cudaMemset ( mEnergyC,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mEnergyC" );
        cudaCheck ( cudaMemset ( mEffP,0	,	fcuda.szPnts*sizeof(float) ),	"Memset mEffP" );










///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int temp_size = 4*(sizeof(float)*3) + 2*sizeof(float) + 2*sizeof(int) + sizeof(uint);
	cudaCheck ( cudaMalloc ( (void**) &msortbuf,	fcuda.szPnts*temp_size ),		"Malloc msortbuf" );

	// Allocate grid
	fcuda.szGrid = (fcuda.gridBlocks * fcuda.gridThreads);
	cudaCheck ( cudaMalloc ( (void**) &mgrid,		fcuda.szPnts*sizeof(int) ),		"Malloc mgrid" );
	cudaCheck ( cudaMalloc ( (void**) &mgridcnt,	fcuda.szGrid*sizeof(int) ),		"Malloc mgridcnt" );
	cudaCheck ( cudaMalloc ( (void**) &mgridoff,	fcuda.szGrid*sizeof(int) ),		"Malloc mgridoff" );
	cudaCheck ( cudaMalloc ( (void**) &mgridactive, fcuda.szGrid*sizeof(int) ),	"Malloc mgridactive" );

	// Transfer sim params to device
	updateSimParams ( &fcuda );

	cudaThreadSynchronize ();
    //////////////////////////////////////////////////////////////
    printf ( "CUDA Allocate: \n" );
    printf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", fcuda.pnum, fcuda.numBlocks, fcuda.numThreads, fcuda.numBlocks*fcuda.numThreads, fcuda.szPnts);
    printf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", fcuda.gridTotal, fcuda.gridBlocks, fcuda.gridThreads, fcuda.gridBlocks*fcuda.gridThreads, fcuda.szGrid, (int) fcuda.gridRes.x, (int) fcuda.gridRes.y, (int) fcuda.gridRes.z );


    //////////////////////////////////////////////////////////////
	// Prefix Sum - Preallocate Block sums for Sorting
	deallocBlockSumsInt ();
	preallocBlockSumsInt ( fcuda.gridTotal );
}

void sphParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl )
{

	fcuda.psmoothradius = sr;
	fcuda.pradius = pr;


	fcuda.pboundmin = bmin;
	fcuda.pboundmax = bmax;


	printf ( "Bound Min: %f %f %f\n", bmin.x, bmin.y, bmin.z );
	printf ( "Bound Max: %f %f %f\n", bmax.x, bmax.y, bmax.z );



	// Transfer sim params to device
	updateSimParams ( &fcuda );

	cudaThreadSynchronize ();
}

void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, float* mass, uint* cluster, uint* gnext, char* clr, float* type, float* Sh , float* T, float* Fix, float* SigYield)
{
	// Send particle buffers
	int numPoints = fcuda.pnum;
	cudaCheck( cudaMemcpy ( mpos,		pos,			numPoints*sizeof(float3), cudaMemcpyHostToDevice ), 	"Memcpy mpos ToDev" );
	cudaCheck( cudaMemcpy ( mvel,		vel,			numPoints*sizeof(float3), cudaMemcpyHostToDevice ), 	"Memcpy mvel ToDev" );
	cudaCheck( cudaMemcpy ( mveleval, veleval,		numPoints*sizeof(float3), cudaMemcpyHostToDevice ), 		"Memcpy mveleval ToDev"  );
	cudaCheck( cudaMemcpy ( mforce,	force,			numPoints*sizeof(float3), cudaMemcpyHostToDevice ), 	"Memcpy mforce ToDev"  );
	cudaCheck( cudaMemcpy ( mpress,	pressure,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 	"Memcpy mpress ToDev"  );
	cudaCheck( cudaMemcpy ( mdensity, density,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mdensity ToDev"  );
	cudaCheck( cudaMemcpy ( mMass, mass,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mMass ToDev"  );
	cudaCheck( cudaMemcpy ( mType, type,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mType ToDev"  );
	cudaCheck( cudaMemcpy ( mSh, Sh,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mSh ToDev"  );
	cudaCheck( cudaMemcpy ( mT, T,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mT ToDev"  );
	cudaCheck( cudaMemcpy ( mFix, Fix,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mFix ToDev"  );
	cudaCheck( cudaMemcpy ( mSigYield, SigYield,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ), 		"Memcpy mSigYield ToDev"  );


	cudaThreadSynchronize ();
}

void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* SW, float* density, float* Mass, uint* cluster, uint* gnext, char* clr, float* Energy, float* EnergyC, float* Sh , float* T, float* L_x, float* L_y, float* L_z, uint* gcell)
{
	// Return particle buffers
	int numPoints = fcuda.pnum;
	cudaCheck( cudaMemcpy ( pos,		mpos,			numPoints*sizeof(float3), cudaMemcpyDeviceToHost ),	"Memcpy mpos FromDev"  );
	cudaCheck( cudaMemcpy ( SW,	mSW,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ), 		"Memcpy pressure FromDev" );
	cudaCheck( cudaMemcpy ( density,	mdensity,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ), 		"Memcpy density FromDev" );
	
	cudaCheck( cudaMemcpy ( gcell,	mgcell,	fcuda.pnum*sizeof(uint),		cudaMemcpyDeviceToHost ),  "Memcpy mgcell FromDev");



	cudaCheck( cudaMemcpy ( L_x,		mL_x,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ), 		"Memcpy mL_x FromDev" );
	cudaCheck( cudaMemcpy ( L_y,		mL_y,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ), 		"Memcpy mL_y FromDev" );
	cudaCheck( cudaMemcpy ( L_z,		mL_z,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ), 		"Memcpy mL_z FromDev" );


	cudaThreadSynchronize ();
}

/* Neighbours research (Don't touch please) */
void InsertParticlesCUDA ( uint* gcell, uint* ccell, int* gcnt )
{
	cudaMemset ( mgridcnt, 0,			fcuda.gridTotal * sizeof(int));

	insertParticles<<< fcuda.numBlocks, fcuda.numThreads>>> (  mpos, mgcell, mgndx, mgridcnt, fcuda.pnum );

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: InsertParticlesCUDA: %s\n", cudaGetErrorString(error) );
	}
	cudaThreadSynchronize ();


}

void PrefixSumCellsCUDA ( int* goff )
{
	// Prefix Sum - determine grid offsets
    prescanArrayRecursiveInt ( mgridoff, mgridcnt, fcuda.gridTotal, 0);
	cudaThreadSynchronize ();

	// Transfer data back if requested
	if ( goff != 0x0 ) {
		cudaCheck( cudaMemcpy ( goff,	mgridoff, fcuda.gridTotal * sizeof(int),  cudaMemcpyDeviceToHost ),  "Memcpy mgoff FromDev" );
	}
}

void CountingSortIndexCUDA ( uint* ggrid )
{
	// Counting Sort - pass one, determine grid counts
	cudaMemset ( mgrid,	GRID_UCHAR,	fcuda.pnum * sizeof(int) );

	countingSortIndex <<< fcuda.numBlocks, fcuda.numThreads>>> ( mgcell, mgndx, mgridoff, mgrid, fcuda.pnum);
	cudaThreadSynchronize ();

	// Transfer data back if requested
	if ( ggrid != 0x0 ) {
		cudaCheck( cudaMemcpy ( ggrid,	mgrid, fcuda.pnum * sizeof(uint), cudaMemcpyDeviceToHost ), "Memcpy mgrid FromDev" );
	}
}

/*-------------------------------------------------------------------------*/



void ComputeConsistencyCuda ()
{

	ComputeConsistency<<< fcuda.numBlocks, fcuda.numThreads>>> ( mgcell, mgridcnt, mgridoff, mgrid, mpos,  mMass, mdensity, mL_x, mL_y, mL_z, mSW, mSh, mType, fcuda.pnum );
    	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: computeConsistencyCUDA: %s\n", cudaGetErrorString(error) );
	}
	cudaThreadSynchronize ();


}




///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////











///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

#include <assert.h>

inline bool isPowerOfTwo(int n) { return ((n&(n-1))==0) ; }

inline int floorPow2(int n) {
	//#ifdef WIN32
		//return 1 << (int)logb((float)n);
	//#else
		int exp;
		frexp((float)n, &exp);
		return 1 << (exp - 1);
	//#endif
}

#define BLOCK_SIZE 256

float**			g_scanBlockSums = 0;
int**			g_scanBlockSumsInt = 0;
unsigned int	g_numEltsAllocated = 0;
unsigned int	g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelsAllocated = level;

    numElts = maxNumElements;
    level = 0;

    do {
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
			cudaCheck ( cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(float)), "Malloc prescanBlockSums g_scanBlockSums");
        numElts = numBlocks;
    } while (numElts > 1);

}
void preallocBlockSumsInt (unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSumsInt = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated = level;

    numElts = maxNumElements;
    level = 0;

    do {
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) cudaCheck ( cudaMalloc((void**) &g_scanBlockSumsInt[level++], numBlocks * sizeof(int)), "Malloc prescanBlockSumsInt g_scanBlockSumsInt");
        numElts = numBlocks;
    } while (numElts > 1);
}

void deallocBlockSums()
{
	if ( g_scanBlockSums != 0x0 ) {
		for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
			cudaCheck ( cudaFree(g_scanBlockSums[i]), "Malloc deallocBlockSums g_scanBlockSums");

		free( (void**)g_scanBlockSums );
	}

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}
void deallocBlockSumsInt()
{
	if ( g_scanBlockSums != 0x0 ) {
		for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
			cudaCheck ( cudaFree(g_scanBlockSumsInt[i]), "Malloc deallocBlockSumsInt g_scanBlockSumsInt");
		free( (void**)g_scanBlockSumsInt );
	}

    g_scanBlockSumsInt = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}



void prescanArrayRecursive (float *outArray, const float *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1) {
        prescan<true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be added to each block to
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive (g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        uniformAdd<<< grid, threads >>> (outArray, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescan<false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescan<false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}

void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1) {
        prescanInt <true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSumsInt[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescanInt <true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be added to each block to
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursiveInt (g_scanBlockSumsInt[level], g_scanBlockSumsInt[level], numBlocks, level+1);

        uniformAddInt <<< grid, threads >>> (outArray, g_scanBlockSumsInt[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAddInt <<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescanInt <false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescanInt <false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}


void prescanArray ( float *d_odata, float *d_idata, int num )
{
	// preform prefix sum
	preallocBlockSums( num );
    prescanArrayRecursive ( d_odata, d_idata, num, 0);
	deallocBlockSums();
}
void prescanArrayInt ( int *d_odata, int *d_idata, int num )
{
	// preform prefix sum
	preallocBlockSumsInt ( num );
    prescanArrayRecursiveInt ( d_odata, d_idata, num, 0);
	deallocBlockSumsInt ();
}

char* d_idata = NULL;
char* d_odata = NULL;

void prefixSum ( int num )
{
	prescanArray ( (float*) d_odata, (float*) d_idata, num );
}

void prefixSumInt ( int num )
{
	prescanArrayInt ( (int*) d_odata, (int*) d_idata, num );
}

void prefixSumToGPU ( char* inArray, int num, int siz )
{
    cudaCheck ( cudaMalloc( (void**) &d_idata, num*siz ),	"Malloc prefixumSimToGPU idata");
    cudaCheck ( cudaMalloc( (void**) &d_odata, num*siz ),	"Malloc prefixumSimToGPU odata" );
    cudaCheck ( cudaMemcpy( d_idata, inArray, num*siz, cudaMemcpyHostToDevice),	"Memcpy inArray->idata" );
}
void prefixSumFromGPU ( char* outArray, int num, int siz )
{
	cudaCheck ( cudaMemcpy( outArray, d_odata, num*siz, cudaMemcpyDeviceToHost), "Memcpy odata->outArray" );
	cudaCheck ( cudaFree( d_idata ), "Free idata" );
    cudaCheck ( cudaFree( d_odata ), "Free odata" );
	d_idata = NULL;
	d_odata = NULL;
}
