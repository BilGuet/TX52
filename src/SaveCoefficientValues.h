#pragma once

#include<vector>

#include "Point.h"

void SaveCoefficientValues( const std::vector<Point>& points, const std::vector<double>& W)
{
    FILE* fileT = NULL;
    char bufferT[255];

    snprintf(bufferT, sizeof(char) * 255, "../data/data_%u_%u.csv", points.size(), k);
    fileT = fopen(bufferT, "w");

    fprintf(fileT, "X, Y, Z, W\n");
    for (int i = 0; i < points.size(); i++)
    {
        fprintf(fileT, "%.6f, %.6f, %.6f, %.4f\n", points[i].x, points[i].y, 0, W[i]);
    }

    fclose(fileT);
}
