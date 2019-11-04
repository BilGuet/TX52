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

    std::string s;
    if(argc == 1)
    {
        std::cout << "No input file. Exiting" << std::endl;
        return 0;
    }
    else
    {
        s = argv[1];
    }

    // check that the file exist
    std::ifstream file (s.c_str());
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

    std::cout << std::endl << "There is " << points.size() << " points." << std::endl;
    std::cout << "press ENTER to continue...";
    getchar();
    
    //final value
    auto W = std::vector<double>(points.size(), 0);

    ComputeWUsingConvolutionMatrix(points, W);
    
    std::cout << std::endl << "Saving the values in file..." << std::endl << std::endl;
    SaveCoefficientValues(points, W);

    return 0;
}
