#define CUDA_KERNEL
#include "sph_system_kern.cuh"

#include "cutil_math.h"

#include "radixsort.cu"						// Build in RadixSort

__constant__ sphParams		simData;
__constant__ uint				gridActive;


#define PI 3.14159265358



///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Danger//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////Don't touch//////////////////////////////////////////////////////////
__global__ void insertParticles ( float3* mpos, uint* mgcell, uint* mgndx, int* mgridcnt, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;;	// particle index
	if ( i >= pnum ) return;

	//printf("I'm here %d \n",i);

	register float3 gridMin = simData.gridMin;
	register float3 gridDelta = simData.gridDelta;
	register int3 gridRes = simData.gridRes;
	register int3 gridScan = simData.gridScanMax;
	register float poff = simData.psmoothradius ;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (mpos[i] - gridMin) * gridDelta;
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		mgcell[i] = gs;						// Grid cell insert.
		mgndx[i] = atomicAdd ( &mgridcnt[ gs ], 1 );		// Grid counts.

		gcf = (make_float3(-poff,-poff,-poff) + mpos[i] - gridMin) * gridDelta;
		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	} else {
		mgcell[i] = GRID_UNDEF;
	}
}

// the mutex variable
__device__ int g_mutex = 0;


// Counting Sort - Index
__global__ void countingSortIndex ( uint* mgcell, uint* mgndx, int* mgridoff, uint* mgrid, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;;		// particle index
	if ( i >= pnum ) return;

	uint icell = mgcell[i];
	uint indx =  mgndx[i];
	int sort_ndx = mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
	if ( icell != GRID_UNDEF ) {
		mgrid[ sort_ndx ] = i;					// index sort, grid refers to original particle order
	}
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

__device__ float Dxkernel(float r, float h)
{

   float  q,l, factor;
   float dw=0.0f;
    q = r/h;



	factor = 3.0/(2.0*PI*h*h*h);
	if(q>0.0 && q<1.0)
	{
	 dw = factor*(-3.0*q+(9.0/4.0)*q*q)/(r*h);
        }
	else if(q>=1.0 && q<2.0)
	{
	  dw = -factor*((2.0-q)*(2.0-q))/(r*h);
	}
        else
        {
            dw = 0.0;
        }
	return dw;
}
__device__ float kernel(float r, float h)
{
	float nr, q, w=0.0f, factor;

   	 q = r/h;

   	
        factor = 3.0/(2.0*PI*h*h*h);
	if(q>=0.0 && q<1.0)
	{
	 w = factor*(1.0-1.5*q*q+(3.0/4.0)*q*q*q);
        }
	else if(q>=1.0 && q<2.0)
	{
	  w = factor*((2.0-q)*(2.0-q)*(2.0-q));
	}
        else
        {
            w = 0.0;
        }
	return w;


}





__device__ void contributeConsistency ( int i, float3 p, float Shi, float Type,int cell, int* mgridcnt, int* mgridoff, uint* mgrid, float3* mpos,  float* mMass, float* mdensity, float3& Lx, float3& Ly, float3& Lz, float& dSW, float* mSh , float* mType)
{
	float3 dist;
	float dsq, c, sum;


			Lx.x = 0.0;
			Lx.y = 0.0;
			Lx.z = 0.0;
			Ly.x = 0.0;
			Ly.y = 0.0;
			Ly.z = 0.0;
			Lz.x = 0.0;
			Lz.y = 0.0;
			Lz.z = 0.0;
			dSW = 0.0;

	if ( mgridcnt[cell] == 0 ) return;

	int cfirst = mgridoff[ cell ];
	int clast = cfirst + mgridcnt[ cell ];
	int j;
	float3 dxkernel;
	float W;
	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
		j = mgrid[ cndx ];
		dist =  p-mpos[ j ];
		float massj = mMass[ j ];
		float densj = mdensity[ j ];
		float mdj = massj/densj;


		dsq = sqrtf(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

		float Hij = 0.5*(Shi+mSh[j]);
		if ( dsq <= 2.f*Hij )
		{

			W = kernel(dsq,Hij);
			dxkernel = Dxkernel(dsq,Hij)*(dist);


			dSW += (mdj)*W;
			Lx.x += -(mdj)*dist.x*dxkernel.x;
			Lx.y += -(mdj)*dist.x*dxkernel.y;
			Lx.z += -(mdj)*dist.x*dxkernel.z;

			Ly.x += -(mdj)*dist.y*dxkernel.x;
			Ly.y += -(mdj)*dist.y*dxkernel.y;
			Ly.z += -(mdj)*dist.y*dxkernel.z;

			Lz.x += -(mdj)*dist.z*dxkernel.x;
			Lz.y += -(mdj)*dist.z*dxkernel.y;
			Lz.z += -(mdj)*dist.z*dxkernel.z;


		}
	}


}

__global__ void ComputeConsistency (uint* mgcell, int* mgridcnt, int* mgridoff, uint*  mgrid, float3* mpos, float* mMass, float* mdensity, float3* mL_x,  float3* mL_y, float3* mL_z, float* mSW, float* mSh, float* mType, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;;	// particle index
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;



	///////////////////////////////////////////////////////////////////////////////

	float3 _Lx = make_float3(0.0,0.0,0.0);
	float3 _Ly = make_float3(0.0,0.0,0.0);
	float3 _Lz = make_float3(0.0,0.0,0.0);
	float3 a_Lx = make_float3(0.0,0.0,0.0);
	float3 a_Ly = make_float3(0.0,0.0,0.0);
	float3 a_Lz = make_float3(0.0,0.0,0.0);
  float a_dSW = 0.0;
 	float _dSW = 0.0;
	///////////////////////////////////////////////////////////////////////////////

	float3 pos = mpos[ i ];
	for (int c=0; c < simData.gridAdjCnt; c++)
	{
		 contributeConsistency ( i, pos, mSh[i], mType[i], gc + simData.gridAdj[c],  mgridcnt, mgridoff,  mgrid, mpos,  mMass, mdensity, a_Lx, a_Ly, a_Lz, a_dSW, mSh, mType );
		 _Lx += a_Lx ;
		 _Ly += a_Ly ;
		 _Lz += a_Lz ;
		 _dSW+= a_dSW;

	}
	__syncthreads();

	float Dter = 1.f;
	Dter = fabs(_Lx.x*(_Ly.y*_Lz.z-_Lz.y*_Ly.z) -_Ly.x*(_Lx.y*_Lz.z-_Lz.y*_Lx.z)+_Lz.x*(_Lx.y*_Ly.z-_Ly.y*_Lx.z));

	/////////////////////////////////////////////////////////////////////////////////////////
	if(Dter>1.0e-16)
	{
	mL_x[i] = make_float3( (_Ly.y*_Lz.z-_Lz.y*_Ly.z)/Dter,(_Lx.z*_Lz.y-_Lz.z*_Lx.y)/Dter,(_Lx.y*_Ly.z-_Ly.y*_Lx.z)/Dter);
	mL_y[i] = make_float3( (_Ly.z*_Lz.x-_Lz.z*_Ly.x)/Dter,(_Lx.x*_Lz.z-_Lz.x*_Lx.z)/Dter,(_Lx.z*_Ly.x-_Ly.z*_Lx.x)/Dter);
	mL_z[i] = make_float3( (_Ly.x*_Lz.y-_Lz.x*_Ly.y)/Dter,(_Lx.y*_Lz.x-_Lz.y*_Lx.x)/Dter,(_Lx.x*_Ly.y-_Ly.x*_Lx.y)/Dter);
	}
	else
	{
	mL_x[i] = make_float3(1.f,0.0,0.0);
	mL_y[i] = make_float3(0.0,1.f,0.0);
	mL_z[i] = make_float3(0.0,0.0,1.f);

	}
	mSW[i] = _dSW;

}








































void updateSimParams ( sphParams* cpufp )
{
	cudaError_t status;

        status = cudaMemcpyToSymbol ( simData, cpufp, sizeof(sphParams) );


	printf ( "SIM PARAMETERS:\n" );
	printf ( "  CPU: %p\n", cpufp );
	printf ( "  GPU: %p\n", &simData );
}
