#include "../../3rdPartyLibraries/eigen/Eigen/Eigenvalues"
#include "../../3rdPartyLibraries/eigen/Eigen/Dense"
#include "SolveEigenvalues.h"

std::vector<double> SolveEigenvalues(double CovXX, double CovXY, double CovYX, double CovYY)
{
    Eigen::Matrix2d Cov;
    Cov(0, 0) = CovXX;
    Cov(0, 1) = CovXY;
    Cov(1, 0) = CovYX;
    Cov(1, 1) = CovYY;

    // instanciate an eigenSolver
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(Cov);

    // compute eigenvalues
    Eigen::Vector2d eigenvalues = solver.eigenvalues();

    // save the eigenvalues
    std::vector<double> result;
    result.push_back(eigenvalues[0]);
    result.push_back(eigenvalues[1]);

    return result;
}