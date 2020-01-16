#include "../../3rdPartyLibraries/eigen/Eigen/Eigenvalues"
#include "../../3rdPartyLibraries/eigen/Eigen/Dense"
#include "SolveEigenvalues.h"

std::pair<double, double> SolveEigenvalues(double CovXX, double CovXY, double CovYX, double CovYY)
{
    Eigen::Matrix2d Cov;
    // create the Eigen matrix
    Cov(0, 0) = CovXX;
    Cov(0, 1) = CovXY;
    Cov(1, 0) = CovYX;
    Cov(1, 1) = CovYY;
    
    // instanciate an eigenSolver
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(Cov);
    
    // compute eigenvalues
    Eigen::Vector2d eigenvalues = solver.eigenvalues();

    
    return { eigenvalues[0], eigenvalues[1] };

/*
    // own implementation
    double det = pow(CovXX + CovXY, 2) - 4 * (CovXX * CovYY - CovXY * CovYX);

    if (det >= 0) {
        double e1 = (CovXX + CovXY + sqrt(det)) / 2;
        double e2 = (CovXX + CovXY - sqrt(det)) / 2;
        return { e1, e2 };
    }
    else {
        double e1 = (CovXX + CovXY) / 2 + sqrt(-det) / 2;
        double e2 = (CovXX + CovXY) / 2 + sqrt(-det) / 2;
        return { e1, e2 };
    }
*/
}