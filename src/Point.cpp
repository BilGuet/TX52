#include <cmath>

#include "Point.h"

double Point::Distance(Point p, Point q)
{
    return sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));
}
