#ifndef DEF_HOST_CUDA
	#define DEF_HOST_CUDA

	#include <vector_types.h>
	#include <driver_types.h>			// for cudaStream_t

	#define TOTAL_THREADS			1000000
	#define BLOCK_THREADS			256
	#define MAX_NBR					80	
	
	#define COLOR(r,g,b)	( (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int		uint;
	typedef unsigned short		ushort;
	typedef unsigned char		uchar;

	#define OFFSET_POS		0
	#define OFFSET_VEL		12
	#define OFFSET_VELEVAL	24
	#define OFFSET_FORCE	36
	#define OFFSET_PRESS	48
	#define OFFSET_DENS		52
	#define OFFSET_CELL		56
	#define OFFSET_GCONT	60
	#define OFFSET_CLR		64

	extern "C"
	{

	void cudaInit();
	void cudaExit();

	void sphClearCUDA ();
	void sphSetupCUDA (  int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk, float Initial_Density,float  Young_Modulus , float Poisson_Coef, float Shear_Modulus, float Sound_Speed, float Delta_X, float XSPH_Coef, float  ArtVis_Coef_1, float ArtVis_Coef_2, float ArtVis_Coef_3, float ArtStr_Coef, float DeltaSPH_Coef , float SigYield, float dt);
	void sphParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl );

	void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, float* mass, uint* cluster, uint* gnext, char* clr , float* type, float* Sh, float* T, float* Fix, float* SigYield);
	void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, float* mMass, uint* cluster, uint* gnext, char* clr , float* Energy, float* EnergyC, float* Sh, float* T, float* mL_x, float* mL_y, float* mL_z, uint* gcell);
	
	void InsertParticlesCUDA ( uint* gcell, uint* ccell, int* gcnt );	
	void PrefixSumCellsCUDA ( int* goff );
	void CountingSortIndexCUDA ( uint* ggrid );	
	void CountingSortFullCUDA ( uint* ggrid );
	void ComputePressureStressCUDA ();
	void ComputeConsistencyCuda ();
	void ComputeNumericalCorrectionCuda ( );
	void ComputeAllThingsCuda ();
	void ComputeAllRateCuda ( );
	void ComputeInternalForcesCuda (  );
	void advanceEulerParticlesCuda ( float dt);
	
	void AdvanceCorrectorParticlesCuda (  float dt, float m_Time );

	void AdvancePredectorParticlesCuda (  float dt, float m_Time );

	void preallocBlockSumsInt(unsigned int num);
	void deallocBlockSumsInt();
	void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level);
	

	void prefixSumToGPU ( char* inArray, int num, int siz );
	void prefixSumFromGPU ( char* outArray, int num, int siz );
	void prefixSum ( int num );
	void prefixSumInt ( int num );
	
	void prescanArray ( float* outArray, float* inArray, int numElements );
	void prescanArrayInt ( int* outArray, int* inArray, int numElements );
	
	}

#endif
