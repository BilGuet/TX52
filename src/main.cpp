#include <iostream>
#include <string>

#include "Point.h"
#include "GetPointsFromVTK.h"
#include "GetPointsFromCSV.h"
#include "SaveCoefficientValues.h"
#include "ComputeWUsingConvolutionMatrix.h"
#include "ComputeWUsingDistance.h"
#include "ComputeWUsingNumberNeighbor.cuh"

int main(int argc, char* argv[])
{
    //points coordinates
    std::vector<Point> points;

    //number of neighbors (= 10 if not precised by user)
    unsigned int k = 10;

    /////////////////////////   FILE READING, DO NOT TOUCH //////////////////////

    if(argc == 1)
    {
        std::cout << "No input file or k. Exiting" << std::endl;
        return 0;
    }
    else if(argc == 2)
    {
        std::cout << "k will be set at 10 by default" << std::endl;
    }
    else
    {
        k = atoi(argv[2]);
    }

    std::string s = argv[1];
    std::ifstream file (s.c_str());
    // check that the file exist
    if(!file.good())
    {
        std::cout << "File does not exist. Exiting" << std::endl;
        return 0;
    }

    if(s.substr(s.find_last_of(".") + 1) == "vtk")
    {
        GetPointsFromVTK(file, points);
    }
    else if (s.substr(s.find_last_of(".") + 1) == "csv")
    {
        GetPointsFromCSV(file, points);
    }
    else
    {
        std::cout << "File has to be VTK or CSV. Exiting." << std::endl;
        return 0;
    }
    
    /////////////////////////////////////////////////////////////////////////////

    std::cout << std::endl << "There is " << points.size() << " points." << std::endl;
    std::cout << "press ENTER to continue...";
    getchar();
    
    //final value
    auto W = std::vector<double>(points.size(), 0);

    ComputeWUsingConvolutionMatrix(points, W, k);
    
    std::cout << std::endl << "Saving the values in file..." << std::endl << std::endl;
    SaveCoefficientValues(points, W, k);

    return 0;
}
