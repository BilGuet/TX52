#ifndef DEF_sph_SYS
	#define DEF_sph_SYS

	#include <iostream>
	#include <vector>
	#include <string>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include "device_launch_parameters.h"
	#include "cuda_runtime.h"
	#include <thrust/sort.h>
	#include<thrust/device_ptr.h>
	#include<thrust/for_each.h>
	#define MAX_PARAM			50
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	

	// Scalar params
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PMAX_FRAC			22
	#define PDRAWMODE			23
	#define PDRAWSIZE			24
	#define PDRAWGRID			25	
	#define PDRAWTEXT			26	
	#define PCLR_MODE			27
	#define PPOINT_GRAV_AMT		28
	#define PSTAT_OCCUPY		29
	#define PSTAT_GRIDCNT		30
	#define PSTAT_NBR			31
	#define PSTAT_NBRMAX		32
	#define PSTAT_SRCH			33
	#define PSTAT_SRCHMAX		34
	#define PSTAT_PMEM			35
	#define PSTAT_GMEM			36
	#define PTIME_INSERT		37
	#define PTIME_SORT			38
	#define PTIME_COUNT			39
	#define PTIME_PRESS			40
	#define PTIME_FORCE			41
	#define PTIME_ADVANCE		42
	#define PTIME_RECORD		43
	#define PTIME_RENDER		44
	#define PTIME_TOGPU			45
	#define PTIME_FROMGPU		46
	#define PFORCE_FREQ			47
	

	// Vector params
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PEMIT_POS			6
	#define PEMIT_ANG			7
	#define PEMIT_DANG			8
	#define PEMIT_SPREAD		9
	#define PEMIT_RATE			10
	#define PPOINT_GRAV_POS		11	
	#define PPLANE_GRAV_DIR		12	

	// Booleans
	#define PRUN				0
	#define PDEBUG				1	
	#define PUSE_CUDA			2	
	#define	PUSE_GRID			3
	#define PWRAP_X				4
	#define PWALL_BARRIER		5
	#define PLEVY_BARRIER		6
	#define PDRAIN_BARRIER		7		
	#define PPLANE_GRAV_ON		11	
	#define PPROFILE			12
	#define PCAPTURE			13

	#define Bsph				2

	struct NList {
		int num;
		int first;
	};
	

	class sphSystem {
	public:
		sphSystem ();
		void displayParticles(int iteration);

		// Particle Utilities
		void AllocateParticles ( int cnt );
		int AddParticle ();
		int NumPoints ()		{ return mNumPoints; }
		// Setup
		void Setup ( bool bStart );
		void SetupDefaultParams ();
		void SetupExampleParams ( bool bStart );
		void SetupSpacing ();
		void SetupAddVolume ( float3 min, float3 max, float spacing, float offs, int total );
		void SetupGridAllocate ( float3 min, float3 max, float sim_scale, float cell_size, float border );
		// Neighbor Search
		void InsertParticles ();
		// Simulation
		void RunSimulation (int iteration);
		void Empty ();
		void TransferToCUDA ();
		void TransferFromCUDA ();
		float GetDT()		{ return m_DT; }

		int getGridCell ( int p, int3& gc );
	
		int getGridCell ( float3& pos, int3& gc );


		// Acceleration Neighbor Tables
		void AllocateNeighborTable ();
		void ClearNeighborTable ();
		void ResetNeighbors ();
		int GetNeighborTableSize ()	{ return m_NeighborNum; }
		void ClearNeighbors ( int i );
		int AddNeighbor();
		int AddNeighbor( int i, int j, float d );
		// GPU Support functions
		void AllocatePackBuf ();
		void PackParticles ();
		void UnpackParticles ();
		// Parameters			
		float3 GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec ( int p, float3 v );
			
		
		


		//void ReduceEnergy(float *EC, float *EI, float &Ec, float &Ei, unsigned int count);
		
	private:

		std::string				mSceneName;
		// Time
		int						m_Frame;		
		float						m_DT;
		float						m_Time;	
		// Simulation Parameters
		float						m_Param [ MAX_PARAM ];			// see defines above
		float3						m_Vec [ MAX_PARAM ];
		

				
		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mGoodPoints;

		// Simulation variables
		
		char*							mClr;
		float3*						mVelEval;
		unsigned short*						mAge;

		float3*						mPos;
		float3*						mVel;
		float*							mPressure;
		float*							mDensity;
		float*							mMass;
		float3*						mForce;
		float* 						mEnergy;

		float* 						mS_xx;
 		float* 						mS_yy; 
		float* 						mS_zz; 
		float* 						mS_xy; 
		float* 						mS_xz; 
		float* 						mS_yz; 
		
		float* 						mSig_xx; 
		float* 						mSig_yy; 
		float* 						mSig_zz; 
		float* 						mSig_xy; 
		float* 						mSig_xz; 
		float* 						mSig_yz;

		float*  						mEps_xx; 
		float*  						mEps_yy; 
		float*  						mEps_zz; 
		float*  						mEps_xy; 
		float*  						mEps_xz; 
		float*  						mEps_yz;

		float*  						mRot_xy; 
		float*  						mRot_xz; 
		float*  						mRot_yz; 

		float* 						mdS_xx;
 		float* 						mdS_yy; 
		float* 						mdS_zz; 
		float* 						mdS_xy; 
		float* 						mdS_xz; 
		float* 						mdS_yz;


		float*						mdDensity;
		float*						mType;		
		float*						mFix;
                
                float*      Max_H;
                float*      Max_R;
		float*      Max_V;
		
		//Grid Parameters 
		typedef unsigned int uint;
		uint*						mGridCell;
		uint*						mClusterCell;
		uint*						mGridNext;
		uint*						mNbrNdx;
		uint*						mNbrCnt;
		// Acceleration Grid
		uint*						m_Grid;
		uint*						m_GridCnt;
		int						m_GridTotal;			// total # cells
		int3						m_GridRes;			// resolution in each axis
		float3						m_GridMin;			// volume of grid (may not match domain volume exactly)
		float3				m_GridMax;		
		float3				m_GridSize;					// physical size in each axis
		float3				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];
		// Acceleration Neighbor Table
		int						m_NeighborNum;
		int						m_NeighborMax;
		int*						m_NeighborTable;
		float*						m_NeighborDist;
		char*						mPackBuf;
		int*						mPackGrid;
		int						mVBO[3];		

		float* mEnergyC;
		float* mSh;
		float* mT;
		float* mSigYield;

		float Initial_Density;
		float Young_Modulus ;
		float Poisson_Coef ;
		float Shear_Modulus  ;
		float Sound_Speed;

		float Delta_X ;
		float XSPH_Coef ;

		float ArtVis_Coef_1 ;
		float ArtVis_Coef_2 ;
		float ArtVis_Coef_3 ;

		float ArtStr_Coef ;
	
		float DeltaSPH_Coef ;

		float SigYield; 

		float Bulk_Modulus_T;
		float Bulk_Modulus_P;


		float3* mL_x; 
		float3* mL_y; 
		float3* mL_z;

		float* mSW;

		
		
	};	

#endif
