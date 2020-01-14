#include <iostream>
#include <string>
#include <ctime>
#include "Point.h"
#include "GetPointsFromVTK.h"
#include "GetPointsFromCSV.h"
#include "SaveCoefficientValues.h"
#include "ComputeWUsingConvolutionMatrix.h"
#include "ComputeWUsingNumberNeighbor.cuh"
#include "ExtractVolume.h"
#include "ComputeNormalVector.cuh"
#include "FreeSurfaceAlgo.cuh"
#include "SaveCoefficientValues.h"

// command line : ./main file_name k


int main(int argc, char* argv[])
{
    // points coordinates
    std::vector<Point> points;

    // number of neighbors (= 10 if not precised by user)
    // used only with matrix covariance
    unsigned int k = 10;

    /////////////////////////   FILE READING, DO NOT TOUCH //////////////////////
                if(argc == 1) {
                    std::cout << "No input file or k. Exiting" << std::endl;
                    return 0;
                }
                else if(argc == 2) {
                    std::cout << "k will be set at 10 by default" << std::endl;
                }
                else {
                    k = atoi(argv[2]);
                }

                std::string s = argv[1];
                std::ifstream file (s.c_str());
                // check that the file exist
                if(!file.good()) {
                    std::cout << "File does not exist. Exiting" << std::endl;
                    return 0;
                }

                if(s.substr(s.find_last_of(".") + 1) == "vtk") {
                    GetPointsFromVTK(file, points);
                }
                else if (s.substr(s.find_last_of(".") + 1) == "csv") {
                    GetPointsFromCSV(file, points);
                }
                else {
                    std::cout << "File has to be VTK or CSV. Exiting." << std::endl;
                    return 0;
                }
    
                std::cout << std::endl << "There is " << points.size() << " points." << std::endl;
                std::cout << "press ENTER to continue...";
                getchar();
    /////////////////////////   FILE READING, DO NOT TOUCH //////////////////////

    
    ExtractVolume(file, points);

    clock_t begin = clock();

    std::vector<double> W;
    std::vector<int> flags;
    std::vector<double> eigenValues;
    std::vector<Vector> normals;
    FreeSurfaceAlgo(points, eigenValues, normals, flags);
    SaveCoefficientValues(points, flags);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << elapsed_secs << std::endl;

    return 0;
}
