#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Point.h"
//#include "vtkUnstructuredGridReader.h"
//#include "vtkUnstructuredGrid.h"


//extract the points coordinates from the vtk file
void GetPointsFromVTK(std::ifstream&  file, std::vector<Point>& p)
{
/*
//// USING VTK LIBRARY ////
    auto reader = vtkUnstructuredGridReader::New();
    reader->SetFileName(file_name.c_str());
    reader->Update();
    auto dataSet = reader->GetOutput();

    auto nb_points = dataSet->GetNumberOfPoints();
    for (auto i = 0; i < nb_points; ++i)
    {
        auto actual_point = dataSet->GetPoint(i);

        //add the point to the vector
        p.push_back({actual_point[0], actual_point[1]});
    }
*/

    unsigned int n = 0;
    std::string line;
    std::string a = "";
    std::string cell;
    std::vector<std::string> parsedRow;

    while(a != "POINTS" && std::getline(file, line))
    {
        std::stringstream lineStream(line);
        parsedRow.clear();
        while (std::getline(lineStream, cell, ' '))
        {
            parsedRow.push_back(cell);
        }
        if (parsedRow.size() > 0)
        {
            a = parsedRow[0];
        }
    }
    n = atof(parsedRow[1].c_str());
    
    unsigned int i = 0;
    while(i < n && std::getline(file, line))
    {
        std::stringstream lineStream(line);
        parsedRow.clear();
        while (std::getline(lineStream, cell, ' '))
        {
            parsedRow.push_back(cell);
        }
        p.push_back({atof(parsedRow[0].c_str()), atof(parsedRow[1].c_str())});
        i++;
    }
}
