#pragma once

#include <iostream>
#include <vector>
#include "Point.h"
#include "Vector.h"

//void SaveCoefficientValues(const std::vector<Point>& points, const std::vector<int>& flags, const std::vector<Vector>& normals)
void SaveCoefficientValues(const std::vector<Point>& points, const std::vector<double>& W)
{
    std::cout << "Saving values.." << std::endl;

    FILE* fileT = NULL;
    char bufferT[255];

    //snprintf(bufferT, sizeof(char) * 255, "../data_P_%u_k_%u.csv", points.size(), k);
    snprintf(bufferT, sizeof(char) * 255, "data/test.csv");
    fileT = fopen(bufferT, "w");

    fprintf(fileT, "X, Y, Z, W\n");
    for (int i = 0; i < points.size(); i++)
    {
        fprintf(fileT, "%.6f, %.6f, %.6f, %.6f\n", points[i].x, points[i].y, 0, W[i]);
    }

    fclose(fileT);
}
