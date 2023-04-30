/**
 * @file EigenPTSC.hpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 * 
 *    Simple wrapper for OsqpEigen QP solver 
 *    The QP problem is of the following form:
 *
 *      min 	1 / 2 * x^T * A_qp * x + b_qp^T * x
 *       x
 *
 *      s.t.	A_eq * x + b_eq = 0
 *            A_ieq * x + b_ieq <= 0
 */
#ifndef OSQP_EIGEN_OPTIMIZATION_HPP_
#define OSQP_EIGEN_OPTIMIZATION_HPP_

#include <iostream>
#include <vector>
#include <OsqpEigen/OsqpEigen.h>

#include "QpProblem.hpp"

struct OsqpSettings
{
  bool verbosity = false;
  double alpha = 1.0;
  double absolute_tolerance = 1e-6;
  double relative_tolerance = 1e-6;
  bool warm_start = false;
  int max_iteration = 10000;
  double time_limit = 0; //0 -> disabled
  bool adaptive_rho = true;
  int adaptive_rho_interval = 0; //0 -> automatic
}; 

class OsqpEigenOpt 
{    
public:
  OsqpEigenOpt( );
  OsqpEigenOpt(	const SparseQpProblem &sparse_qp_problem, 
                const OsqpSettings &settings = OsqpSettings());

  void initializeSolver(const SparseQpProblem &sparse_qp_problem, 
                        const OsqpSettings &settings );

  void setGradientAndInit(Eigen::VectorXd &b_qp); 

  Eigen::VectorXd solveProblem();

  bool checkFeasibility(); 

private:
  OsqpEigen::Solver solver_;

  uint32_t n_; //number of optimization variables
  uint32_t m_; //number of constraints

  Eigen::VectorXd b_qp_, lower_bound_, upper_bound_;
  Eigen::SparseMatrix<double> linearConstraintsMatrix_;

  static void setSparseBlock( Eigen::SparseMatrix<double> &output_matrix, 
                              const Eigen::SparseMatrix<double> &input_block,
                              uint32_t i, uint32_t j );

  void setSolverSettings(const OsqpSettings &settings);
};

#endif //OSQP_EIGEN_OPTIMIZATION_HPP_