#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<cmath>
#include <iomanip>
#include<vector>
#include <cstring>
#include "sph_defs.h"
#include "sph_system.h"
#include "sph_system_host.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
	#include<thrust/device_ptr.h>
	#include<thrust/for_each.h>

#define EPSILON			0.00001f			//for collision detection

using namespace std;




void sphSystem::TransferToCUDA ()
{
	/* function by hassan to send to GPU */
	CopyToCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mMass, mClusterCell, mGridNext, (char*) mClr , mType , mSh , mT, mFix, mSigYield  );
}
void sphSystem::TransferFromCUDA ()
{
	/* function by hassan to receive from GPU */
	CopyFromCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mSW, mDensity, mMass, mClusterCell, mGridNext, (char*) mClr,  mEnergy,  mEnergyC, mSh, mT, (float*) mL_x, (float*) mL_y, (float*) mL_z, mGridCell);
}

//------------------------------ Initialization
sphSystem::sphSystem ()
{
	// Pointers (e.g. 0x0 is specific to pointers)
	mNumPoints = 0;
	mMaxPoints = 0;
	mPackBuf = 0x0;
	mPackGrid = 0x0;


	mPos = 0x0;
	mClr = 0x0;
	mVel = 0x0;
	mVelEval = 0x0;
	mAge = 0x0;
	mPressure = 0x0;
	mDensity = 0x0;
	mForce = 0x0;
	mClusterCell = 0x0;
	mGridNext = 0x0;
	mNbrNdx = 0x0; /* Number of index */
	mNbrCnt = 0x0;

	m_Grid = 0x0;
	m_GridCnt = 0x0;



	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;

	m_Param [ PEXAMPLE ]	= 1;
	m_Param [ PGRID_DENSITY ] = 7890.0;
	m_Param [ PNUM ]		= 3000000; /* Maximum Number of particles */



}

//------------------------------*****************************------------------------------\\

void sphSystem::Setup ( bool bStart )
{
	m_Frame = 0;
	m_Time = 0;
	mNumPoints = 0;
	SetupDefaultParams ();

	m_Param [PGRIDSIZE] = 2.0*m_Param[PSMOOTHRADIUS];

	AllocateParticles ( m_Param[PNUM] );

	SetupSpacing ();

	SetupAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.0, m_Param[PNUM] ); // Create the particles

	SetupGridAllocate ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0 );	// Setup grid


	sphSetupCUDA ( NumPoints(), m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, (int) m_Vec[PEMIT_RATE].x, Initial_Density, Young_Modulus , Poisson_Coef, Shear_Modulus, Sound_Speed, Delta_X, XSPH_Coef, ArtVis_Coef_1, ArtVis_Coef_2, ArtVis_Coef_3, ArtStr_Coef, DeltaSPH_Coef, SigYield, m_DT );

	float3 grav = m_Vec[PPLANE_GRAV_DIR];

	sphParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY], *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF], m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );

	TransferToCUDA ();		// Initial transfer */

}


//------------------------------*****************************------------------------------\\

void sphSystem::SetupDefaultParams ()
{
		Initial_Density=7850.0;
		Young_Modulus = 210.0e9;
		Poisson_Coef = 0.3;
		Shear_Modulus = Young_Modulus/(2.0*(1.0 +Poisson_Coef)) ;
		Sound_Speed = sqrt(Young_Modulus/ Initial_Density);

		Delta_X = 0.001;


		XSPH_Coef = 0.0;

		ArtVis_Coef_1 = 0.25;
		ArtVis_Coef_2 = 0.25;
		ArtVis_Coef_3 = 0.1;

		ArtStr_Coef = 0.2;

		DeltaSPH_Coef = 0.1;
                SigYield = 792.0e6;

                Bulk_Modulus_T = Young_Modulus/(3.0*(1.0-Poisson_Coef));
		Bulk_Modulus_P = (Young_Modulus)/(3.0*(1.0-Poisson_Coef));


		m_Param [ PSIMSCALE ] =		1.0;			// unit size
		m_Param [ PRESTDENSITY ] =	Initial_Density;			// kg / m^3

		m_Vec [ PVOLMIN ]=make_float3 ( -0.020, -0.020, -0.020);
		m_Vec [ PVOLMAX ]=make_float3( 0.070, 0.070, 0.070);
		m_Vec [ PINITMIN ]=make_float3 ( -0.020, -0.020, -0.020);
		m_Vec [ PINITMAX ]=make_float3( 0.070, 0.070, 0.070);
		m_Param [ PSPACING ] = Delta_X;				// Fixed spacing		Dx = x-axis density
		m_Param [ PSMOOTHRADIUS ] =	1.5*Delta_X;		// Search radius
		m_Param [ PRADIUS ] =		1.5*Delta_X;
		m_Param [PGRIDSIZE] = 2.0*m_Param[PSMOOTHRADIUS] ;
		m_Param[PMASS] = Delta_X*Delta_X*Delta_X*Initial_Density;
		m_DT = 0.2*m_Param [ PSMOOTHRADIUS ]/(Sound_Speed+1000.0);




}

//------------------------------*****************************------------------------------\\

void sphSystem::RunSimulation (int iteration)
{
	/* Neighbours research (Don't touch please) */
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortIndexCUDA ( 0x0 );
	/* --- */

	ComputeConsistencyCuda ();
}
//------------------------------*****************************------------------------------\\




void sphSystem::displayParticles(int iteration)
{
	cout<<" Number of particles = "<<NumPoints()<<endl;
	FILE *fileT = NULL;

	char bufferT[255];
	snprintf(bufferT, sizeof(char)*255,"data/dataT_%i.csv",iteration);
	fileT= fopen(bufferT,"w");

	fprintf(fileT,"X, Y, Z, Lxx, Lxy, Lxz, Lyx, Lyy, Lyz, Lzx, Lzy, Lzz, W \n");
	///////////////////////////////////////////////////////////////

	for (int i = 0; i < NumPoints(); i++)
	{
		if( mGridCell[i]!= GRID_UNDEF)
		{
		fprintf(fileT,"%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f , %f, %f \n" , mPos[i].x, mPos[i].y, mPos[i].z, mL_x[i].x, mL_x[i].y, mL_x[i].z, mL_y[i].x, mL_y[i].y, mL_y[i].z, mL_z[i].x, mL_z[i].y, mL_z[i].z, mSW[i] );
		}


	}
	 fclose(fileT);
}

//------------------------------*****************************------------------------------\\

void sphSystem::SetupSpacing ()
{
	m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);


	// Particle Boundaries
	m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];

	m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];

}


//------------------------------*****************************------------------------------\\
/* Generation of particles (Discretization, memory already allocated in an other function ) */
void sphSystem::SetupAddVolume ( float3 min, float3 max, float spacing, float offs, int total )
{
        float MASS = 0.0, z, x, y;
				int p;
				for (int k = 0; k < 15; k++)
				{
                                        for (int i = 0; i <15; i++)
                                            {
                                                    for (int j = 0; j <15; j++)
                                                        {
                                                            	x = i*spacing+spacing;
								y = j*spacing+spacing;
								z = k*spacing+spacing;
                                                                p = AddParticle ();
								if ( p != -1 )
								{
								    *(mPos+p)=make_float3 ( x,y,z);
								    *(mMass + p) =*(mDensity+p)*(spacing)*(spacing)*(spacing);
								    *(mSh + p) = 1.5*spacing;
								    
								}
              }
            }

			}

}



//------------------------------*****************************------------------------------\\


void sphSystem::SetupGridAllocate ( float3 min, float3 max, float sim_scale, float cell_size, float border )
{
	float world_cellsize = cell_size / sim_scale;

	m_GridMin = min;
	m_GridMax = max;
	m_GridSize = m_GridMax;
	m_GridSize = make_float3(m_GridSize.x- m_GridMin.x, m_GridSize.y- m_GridMin.y, m_GridSize.z- m_GridMin.z);
	m_GridRes.x = ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
	m_GridRes.y = ceil ( m_GridSize.y / world_cellsize );
	m_GridRes.z = ceil ( m_GridSize.z / world_cellsize );
	m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
	m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
	m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
	m_GridDelta = make_float3(m_GridRes.x, m_GridRes.y, m_GridRes.z);		// delta = translate from world space to cell #
	m_GridDelta = make_float3(m_GridDelta.x/m_GridSize.x, m_GridDelta.y/m_GridSize.y,m_GridDelta.z/m_GridSize.z);

	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);


	m_Param[PSTAT_GMEM] = 12 * m_GridTotal;		// Grid memory used

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_GridSrch = 3; // Number of neighboor cells to search in one direction
	if ( m_GridSrch < 2 ) m_GridSrch = 2;
	m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

	if ( m_GridSrch > 6 ) {
		printf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}




}


//------------------------------*****************************------------------------------\\

void sphSystem::Empty ()
{
	free ( mPos );
	free ( mClr );
	free ( mVel );
	free ( mVelEval );
	free ( mAge );
	free ( mPressure );
	free ( mDensity );
	free ( mForce );
	free ( mClusterCell );
	free ( mGridCell );
	free ( mGridNext );
	free ( mNbrNdx );
	free ( mNbrCnt );

	sphClearCUDA();

	cudaExit ();


}



//------------------------------*****************************------------------------------\\

int sphSystem::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return -1;
	int n = mNumPoints;
	*(mPos + n)=make_float3 ( 0,0,0 );
	*(mVel + n)=make_float3 ( 0,0,0 );
	*(mVelEval + n)=make_float3 ( 0,0,0 );
	*(mForce + n)=make_float3 ( 0,0,0 );
	*(mPressure + n) = 0.000000;
	*(mDensity + n) = Initial_Density;
	*(mMass + n) = 0.0;
	*(mType+n) = 0.0;
	*(mGridNext + n) = -1;
	*(mClusterCell + n) = -1;
	*(mEnergy + n) = 0.000000;
	*(mEnergyC + n) = 0.000000;
	*(mType + n) =0.0;
	mNumPoints++;
	return n;
}


//------------------------------*****************************------------------------------\\

void sphSystem::ClearNeighborTable ()
{
	if ( m_NeighborTable != 0x0 )	free (m_NeighborTable);
	if ( m_NeighborDist != 0x0)		free (m_NeighborDist );
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;
	m_NeighborNum = 0;
	m_NeighborMax = 0;
}

//------------------------------*****************************------------------------------\\

void sphSystem::ResetNeighbors ()
{
	m_NeighborNum = 0;
}

//------------------------------*****************************------------------------------\\

// Allocate new neighbor tables, saving previous data
int sphSystem::AddNeighbor ()
{
	if ( m_NeighborNum >= m_NeighborMax ) {
		m_NeighborMax = 2*m_NeighborMax + 1;
		int* saveTable = m_NeighborTable;
		m_NeighborTable = (int*) malloc ( m_NeighborMax * sizeof(int) );
		if ( saveTable != 0x0 ) {
			memcpy ( m_NeighborTable, saveTable, m_NeighborNum*sizeof(int) );
			free ( saveTable );
		}
		float* saveDist = m_NeighborDist;
		m_NeighborDist = (float*) malloc ( m_NeighborMax * sizeof(float) );
		if ( saveDist != 0x0 ) {
			memcpy ( m_NeighborDist, saveDist, m_NeighborNum*sizeof(int) );
			free ( saveDist );
		}
	};
	m_NeighborNum++;
	return m_NeighborNum-1;
}

//------------------------------*****************************------------------------------\\

void sphSystem::ClearNeighbors ( int i )
{
	*(mNbrCnt+i) = 0;
}

//------------------------------*****************************------------------------------\\

int sphSystem::AddNeighbor( int i, int j, float d )
{
	int k = AddNeighbor();
	m_NeighborTable[k] = j;
	m_NeighborDist[k] = d;
	if (*(mNbrCnt+i) == 0 ) *(mNbrNdx+i) = k;
	(*(mNbrCnt+i))++;
	return k;
}


//------------------------------*****************************------------------------------\\

void sphSystem::InsertParticles ()
{


}

//------------------------------*****************************------------------------------\\

int sphSystem::getGridCell ( int p, int3& gc )
{
	return getGridCell ( *(mPos+p), gc );
}

//------------------------------*****************************------------------------------\\

int sphSystem::getGridCell ( float3& pos, int3& gc )
{
	gc.x = (int)( (pos.x - m_GridMin.x) * m_GridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - m_GridMin.y) * m_GridDelta.y);
	gc.z = (int)( (pos.z - m_GridMin.z) * m_GridDelta.z);
	return (int)( (gc.y*m_GridRes.z + gc.z)*m_GridRes.x + gc.x);
}



//------------------------------*****************************------------------------------\\




void sphSystem::AllocateParticles ( int cnt )
{
	int nump = 0;		// number to copy from previous data

	float3* srcPos = mPos;
	mPos = (float3*)		malloc ( cnt*sizeof(float3) );
	if ( srcPos != 0x0 )	{ std::memcpy ( mPos, srcPos, nump *sizeof(float3)); /*free ( srcPos );*/ }

	char* srcClr = mClr;
	mClr = (char*)			malloc ( cnt*sizeof(char) );
	if ( srcClr != 0x0 )	{ memcpy ( mClr, srcClr, nump *sizeof(char));  }

	float3* srcVel = mVel;
	mVel = (float3*)		malloc ( cnt*sizeof(float3) );
	if ( srcVel != 0x0 )	{ memcpy ( mVel, srcVel, nump *sizeof(float3));  }

	float3* srcVelEval = mVelEval;
	mVelEval = (float3*)	malloc ( cnt*sizeof(float3) );
	if ( srcVelEval != 0x0 ) { memcpy ( mVelEval, srcVelEval, nump *sizeof(float3));  }

	unsigned short* srcAge = mAge;
	mAge = (unsigned short*) malloc ( cnt*sizeof(unsigned short) );
	if ( srcAge != 0x0 )	{ memcpy ( mAge, srcAge, nump *sizeof(unsigned short));  }

	float* srcPress = mPressure;
	mPressure = (float*) malloc ( cnt*sizeof(float) );
	if ( srcPress != 0x0 ) { memcpy ( mPressure, srcPress, nump *sizeof(float));  }

	float* srcDensity = mDensity;
	mDensity = (float*) malloc ( cnt*sizeof(float) );
	if ( srcDensity != 0x0 ) { memcpy ( mDensity, srcDensity, nump *sizeof(float));  }

	float* srcMass = mMass;
	mMass = (float*) malloc ( cnt*sizeof(float) );
	if ( srcMass != 0x0 ) { memcpy ( mMass, srcMass, nump *sizeof(float));  }

	float* srcEnergyC = mEnergyC;
	mEnergyC = (float*) malloc ( cnt*sizeof(float) );
	if ( srcEnergyC != 0x0 ) { memcpy ( mEnergyC, srcEnergyC, nump *sizeof(float));  }

	float* srcEnergy = mEnergy;
	mEnergy = (float*) malloc ( cnt*sizeof(float) );
	if ( srcEnergy != 0x0 ) { memcpy ( mEnergy, srcEnergy, nump *sizeof(float));  }

	float* srcmType = mType;
	mType = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmType != 0x0 ) { memcpy ( mType, srcmType, nump *sizeof(float));  }

	float* srcmSh = mSh;
	mSh = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmSh != 0x0 ) { memcpy ( mSh, srcmSh, nump *sizeof(float));  }

	float* srcmT = mT;
	mT = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmT != 0x0 ) { memcpy ( mT, srcmT, nump *sizeof(float));  }

	float* srcmSigYield = mSigYield;
	mSigYield = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmSigYield != 0x0 ) { memcpy ( mSigYield, srcmSigYield, nump *sizeof(float));  }

	float* srcmFix = mFix;
	mFix = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmFix != 0x0 ) { memcpy ( mFix, srcmFix, nump *sizeof(float));  }

	float3* srcForce = mForce;
	mForce = (float3*)	malloc ( cnt*sizeof(float3) );
	if ( srcForce != 0x0 )	{ memcpy ( mForce, srcForce, nump *sizeof(float3));  }

	uint* srcCell = mClusterCell;
	mClusterCell = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcCell != 0x0 )	{ memcpy ( mClusterCell, srcCell, nump *sizeof(uint));  }

	uint* srcGCell = mGridCell;
	mGridCell = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcGCell != 0x0 )	{ memcpy ( mGridCell, srcGCell, nump *sizeof(uint));  }

	uint* srcNext = mGridNext;
	mGridNext = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcNext != 0x0 )	{ memcpy ( mGridNext, srcNext, nump *sizeof(uint));  }

	uint* srcNbrNdx = mNbrNdx;
	mNbrNdx = (uint*)		malloc ( cnt*sizeof(uint) );
	if ( srcNbrNdx != 0x0 )	{ memcpy ( mNbrNdx, srcNbrNdx, nump *sizeof(uint));  }

	uint* srcNbrCnt = mNbrCnt;
	mNbrCnt = (uint*)		malloc ( cnt*sizeof(uint) );
	if ( srcNbrCnt != 0x0 )	{ memcpy ( mNbrCnt, srcNbrCnt, nump *sizeof(uint));  }

	m_Param[PSTAT_PMEM] = 68 * 2 * cnt;


        float* srcMax_H = Max_H;
	Max_H = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmFix != 0x0 ) { memcpy ( Max_H, srcMax_H, nump *sizeof(float));  }
	float* srcMax_R = Max_R;
	Max_R = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmFix != 0x0 ) { memcpy ( Max_R, srcMax_R, nump *sizeof(float));  }

        float* srcMax_V = Max_V;
	Max_V = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmFix != 0x0 ) { memcpy ( Max_V, srcMax_V, nump *sizeof(float));  }


	float* srcmSW = mSW;
	mSW = (float*) malloc ( cnt*sizeof(float) );
	if ( srcmSW != 0x0 ) { memcpy ( mSW, srcmSW, nump *sizeof(float));  }
	



	float3* srcmL_x = mL_x;
	mL_x = (float3*)		malloc ( cnt*sizeof(float3) );
	if ( srcmL_x != 0x0 )	{ memcpy ( mL_x, srcmL_x, nump *sizeof(float3));  }

	float3* srcmL_y = mL_y;
	mL_y = (float3*)		malloc ( cnt*sizeof(float3) );
	if ( srcmL_y != 0x0 )	{ memcpy ( mL_y, srcmL_y, nump *sizeof(float3));  }

	float3* srcmL_z = mL_z;
	mL_z = (float3*)		malloc ( cnt*sizeof(float3) );
	if ( srcmL_z != 0x0 )	{ memcpy ( mL_z, srcmL_z, nump *sizeof(float3));  }











	mMaxPoints = cnt;
}
