#pragma once

#include <sstream>
#include <fstream>
#include <vector>

#include "Point.h"

void GetPointsFromCSV(std::ifstream& file, std::vector<Point>& p)
{
    unsigned int nb = 0;
    std::string line;

    //ignoring the first line
    std::getline(file, line);

    //extract each line of the file
    while (nb < 225 && std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;

        //extract all the lements of the line
        while (std::getline(lineStream, cell, ','))
        {
            parsedRow.push_back(cell);
        }

        //save only X and Y
        p.push_back({ atof(parsedRow[0].c_str()), atof(parsedRow[1].c_str()) });
        nb++;
    }
}
