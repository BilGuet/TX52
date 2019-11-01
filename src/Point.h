#pragma once

#define k 20

struct Point
{
    Point(double _x, double _y) : x(_x), y(_y) {}
    
    static double Distance(Point p, Point s);

    double x;
    double y;
};
