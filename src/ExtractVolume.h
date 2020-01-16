#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

void getMasse(std::ifstream& file, std::vector<double>& mass, const size_t nbPoints)
{
    std::string line;
    std::string cell;
    std::string a = "";

    // search for the mass section in the file
    while (a != "mass" && std::getline(file, line))
    {
        std::vector<std::string> parsedRow;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ' '))
        {
            parsedRow.push_back(cell);
        }
        if (parsedRow.size() > 1)
        {
            a = parsedRow[1];
        }
        else
        {
            a = "";
        }
    }

    unsigned int i = 0;
    // ignore first line
    std::getline(file, line);
    while (i < nbPoints && std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::getline(lineStream, cell);
        mass.push_back(abs(atof(cell.c_str())));
        i++;
    }
}

void getDensity(std::ifstream& file, std::vector<double>& density, const size_t nbPoints)
{
    std::string line;
    std::string cell;
    std::string a = "";

    // search for the density section in the file
    while (a != "density" && std::getline(file, line))
    {
        std::vector<std::string> parsedRow;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ' '))
        {
            parsedRow.push_back(cell);
        }
        if (parsedRow.size() > 1)
        {
            a = parsedRow[1];
        }
        else
        {
            a = "";
        }
    }

    unsigned int i = 0;
    // ignore first line
    std::getline(file, line);
    while (i < nbPoints && std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::getline(lineStream, cell);
        density.push_back(abs(atof(cell.c_str())));
        i++;
    }
}

void ExtractVolume(std::ifstream& file, std::vector<Point>& points)
{
    std::cout << "Computing volumes..." << std::endl;

    std::vector<double> mass;
    std::vector<double> density;

    file.seekg(0);
    getMasse(file, mass, points.size());
    
    file.seekg(0);
    getDensity(file, density, points.size());

    for (size_t i = 0; i < points.size(); i++)
    {
        //double etha = 2.0 * M_PI * points[i].x * density[i];
        //V.push_back(mass[i] / etha);
        //V.push_back(pow(0.0005, 2)* 1060.0 / density[i]);
        points[i].v = (pow(0.0005, 2)* 1060.0 / density[i]);
    }
}
