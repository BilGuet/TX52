// C++ LIBRARIES :
//_________________
#include<iostream>
#include<cmath>
//#include <vector>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <mpi.h>
#include <gsl/gsl_sf_bessel.h>
//_________________

// System of units : Kg, m, s, N, Pa
const int DIM = 2; // space dimension (1 , 2 , 3)
float DX = 0.2e-3; // space discretization size [m]

// MPI VARIABLES AND CONSTANTS
int SIZE; // number of processors
// RANGE : particles per processor
const int RANGE = 1574;//894;
int RANK; //
int N_MIN;
int N_MAX;

// SPH FORMULATION
// NTOTAL : number of particles
const int NTOTAL = 62925;
const int NSPH = 62925;
const float H = 0.22e-3;
const int MAXNIAC = 110; // maximum particles interactions
const double ALPHA_H = 2.800000; // searching neighboors coefficient : rij < alpha_h x hij
const int SEARCH_INC = 25; // search neighboors each SEARCH_INC iterations
const double W0 = 0.557042; //

// MATERIAL PARAMETERS
const double RHO_0 = 1060.000000; // material density (Kg/m^3)
const double C_SPEED = 1520.000000; // material sound speed (m/s)
const double PARAM_MU = 1.0e6; // Shear modulus [Pa]

// SIMULATION PARAMETERS
const int MAX_ITERATION = 40000; // maximum number of iterations
const int SAVE_INC = 25; // saving state each SAVE_INC iterations
const double DT = 6.0e-9; // reference time step (s)
double dt = DT; // time step (s)
double t = 0.0; // time (s)
unsigned int iteration = 0;

// 2D GRID DECLARATION
// -----------------------------------------------
const double DX_GRID = 0.8e-3;
const double X_GRID_MIN = -1.0e-3;	const double Y_GRID_MIN = -100.0e-3;	const double X_GRID_MAX = 100.0e-3;	const double Y_GRID_MAX = 40.0e-3;
const int NX_GRID = floor((X_GRID_MAX - X_GRID_MIN)/DX_GRID);	const int NY_GRID = floor((Y_GRID_MAX - Y_GRID_MIN)/DX_GRID);
const int NX_NODE = NX_GRID + 1;	const int NY_NODE = NY_GRID + 1;
const int N_GRID = NX_NODE * NY_NODE;	const int M_GRID = NX_GRID * NY_GRID;
const int GRID_MAX_PA = 20000;

auto NODES = new double[N_GRID][DIM];	auto VERTEX = new int[M_GRID][4];
auto GRID_NB_PIC = new int[M_GRID]; /* particles in cell number */
auto GRID_PIC_INDEX = new int[M_GRID][GRID_MAX_PA];
auto PA_PIC_INDEX = new int[NTOTAL];
// -----------------------------------------------

// GAUSSIAN INTEGRATION (32 points)
auto GAUSS_POINTS = new float[32];
auto GAUSS_WEIGHTS = new float[32];
// -----------------------------------------------


// PROJECTILE FOR PINBALL ALGO.
const int N_MASTER = 14;
const int MAX_SLAVES = 20000;

const float PROJECTILE_RADIUS = 3.15e-3; // [m]
double PROJECTILE_MASS; // [Kg]
auto r_master = new float[N_MASTER]; // [m]
auto z_master = new float[N_MASTER]; // [m]

double projectile_velocity; // [m/s]
double projectile_acceleration; // [m/s**2]
double projectile_acceleration_min; // [m/s**2]
double projectile_total_force; // [N]

auto SLAVES_LIST = new int[MAX_SLAVES];
int N_SLAVES;

auto Fr_contact = new float[NTOTAL];
auto Fz_contact = new float[NTOTAL];


auto free_surface_list = new float[NTOTAL];
auto contact_surface_list = new float[NTOTAL];

// -----------------------------------------------
// pour faire beau
auto skin_kind = new float[NTOTAL];
// -----------------------------------------------

int NWALL = 0;
int NRP = 0;


// LIBRARIES
// -----------------------------------------------
// -----------------------------------------------
// LIBRARIES SPH, UTILS, GEOMETRY, PARALLEL, ECT...  :
// -----------------------------------------------
/*
   _________________________
  |                        |
  |           SPH          |
  |________________________|

*/
#include "../src/material/ELASTIC_WAVE.H"
#include "../src/sph/kernel/Gaussian/KERNEL_3D.H"
#include "../src/sph/CYLINDRICAL_KERNEL.H"
#include "../src/sph/RECHERCHE_VOISINS.H"
#include "../src/sph/DENSITY_SUMMATION.H"
#include "../src/sph/STRESS_EVALUATION.H"
#include "../src/sph/MOMENTUM_EQ.H"
#include "../src/sph/CONTINUITY_EQ.H"
#include "../src/sph/MONAGHAN_ART_VISC.H"
#include "../src/sph/ARTIFICIAL_VISCOSITY.H"
#include "../src/sph/EXTERNAL_FORCES.H"
#include "../src/sph/PCI_INTEGRATOR.H"
#include "../src/sph/CONSISTENCY_EQS.H"
#include "../src/sph/TEST_SCALAR_FUNCTION.H"
#include "../src/sph/PINBALL_ALGO.H"
#include "../src/sph/VELOCITY_GRADIENT.H"
#include "../src/sph/RUSANOV_FLUX.H"
#include "../src/sph/ARTIFICIAL_PRESSURE.H"
#include "../src/sph/XSPH.H"
/*
   _________________________
  |                        |
  |        PARALLEL        |
  |________________________|

*/
#include "../src/parallel/receive_fcn.h"
#include "../src/parallel/send_fcn.h"

/*
   _________________________
  |                        |
  |          UTILS         |
  |________________________|

*/
#include "../src/utils/2d/output.cpp"
//#include "../src/utils/2d/output_wall.h"
//#include "../src/utils/2d/read_blocks.h"
#include "../src/utils/2d/output_particles_rank.H"
#include "../src/utils/GAUSS_TABLE_UTILS.H"

/*
   _________________________
  |                        |
  |          WALL          |
  |________________________|

*/
//#include "../src/wall/wall.h"
//#include "../src/wall/wall.cpp"



/*
     _________________________
    /                        /
   /         GEOMETRY       /
  /________________________/

*/
#include "../src/geometry/pave_2d.h"



/*
     _________________________
    /                        /
   /         2D GRID        /
  /________________________/

*/
#include "../src/grid/grid_2d.h"
#include "../src/grid/2d/particle_in_cell.h"
#include "../src/utils/2d/output_grid.h"
// -----------------------------------------------


// TABLES
// -----------------------------------------------
// POSITION AND VELOCITY VECTORS :
float r[NTOTAL];	float z[NTOTAL];
float vr[NTOTAL];	float vz[NTOTAL];

// DENSITY,  MASS AND PRESSURE :
auto density = new float[NTOTAL];	auto mass = new float[NTOTAL];	auto hsml = new float[NTOTAL];	auto pressure = new float[NTOTAL];
auto volume = new float[NTOTAL];
auto C_fcn = new float[NTOTAL];
auto density_ = new float[NTOTAL];
auto alpha_hsml = new float[NTOTAL];


// RATES :
auto ar = new float[RANGE];	auto az = new float[RANGE];
auto density_rate = new float[RANGE];

// KERNEL FUNCTION :
int niac[RANGE];	int pairs[RANGE][MAXNIAC];	float G[RANGE][MAXNIAC];	float W[RANGE][MAXNIAC];	auto dW = new float[RANGE][MAXNIAC][3];

// PCI INTEGRATOR :
auto vr_min = new float[RANGE];	auto vz_min = new float[RANGE];
auto density_min = new float[RANGE];
auto ar_min = new float[RANGE];	auto az_min = new float[RANGE];
auto density_rate_min = new float[RANGE];

auto dSrr_min = new float[RANGE];	auto dStt_min = new float[RANGE];	auto dSzz_min = new float[RANGE];	auto dSrz_min = new float[RANGE];


//  XSPH
auto vr_x = new float[RANGE]; auto vz_x = new float[RANGE];

// STRESS-STRAIN TENSORS :
// deviatoric stress tensor
auto Srr = new float[NTOTAL];	auto Stt = new float[NTOTAL];	auto Szz = new float[NTOTAL];	auto Srz = new float[NTOTAL];
auto dSrr = new float[RANGE];	auto dStt = new float[RANGE];	auto dSzz = new float[RANGE];	auto dSrz = new float[RANGE]; /* Stress rate */
// -----------------------------------------------

/* L Consistency Matrix (not Sym) */
    auto Lxx = new float[NTOTAL]; auto Lyx = new float[NTOTAL];   auto Lzx = new float[NTOTAL];
    auto Lxy = new float[NTOTAL]; auto Lyy = new float[NTOTAL];   auto Lzy = new float[NTOTAL];
    auto Lxz = new float[NTOTAL]; auto Lyz = new float[NTOTAL];   auto Lzz = new float[NTOTAL];

    auto Det_L = new float[NTOTAL];

    auto C0_FCN = new float[NTOTAL];
    auto C0_fcn = new float[NTOTAL];

    auto Dxx = new float[NTOTAL]; auto Dyy = new float[NTOTAL];   auto Dzz = new float[NTOTAL];
    auto Dxz = new float[NTOTAL]; auto Dzx = new float[NTOTAL];

    auto Dxx_c = new float[NTOTAL]; auto Dyy_c = new float[NTOTAL];   auto Dzz_c = new float[NTOTAL];
    auto Dxz_c = new float[NTOTAL]; auto Dzx_c = new float[NTOTAL];

    /* Test : */
    auto TEST_ = new float[NTOTAL];
    auto approx_fcn = new float[NTOTAL];

    auto eigenvalue_ = new float[NTOTAL];

// wall WALL(0.0,0.0,0.0, 10.0,0.0,0.0, 0.0,20.0,0.0); // Sol

// MPI DECLARATIONS
MPI_Status status;
#include "../src/sph/SPH_FORMULATION.H"
