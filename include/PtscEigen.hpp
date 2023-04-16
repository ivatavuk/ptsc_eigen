/**
 * @file PtscEigen.hpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#ifndef PTSCEIGEN_HPP_
#define PTSCEIGEN_HPP_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "OsqpEigenOptimization.hpp"

using VecNd = Eigen::VectorXd;
using MatNd = Eigen::MatrixXd;

namespace PtscEigen {

struct Task
{ 
  /**
   * @brief 
   *   Structure that stores a quadratic cost function task of the following form: 
   *      E(x) = ||A*x - b||^2  
   */
  Task(const MatNd &t_A, const VecNd &t_b);
  double getTaskCostValue(const VecNd &x);

  MatNd A;
  VecNd b;
};

class SVD 
{
  /**
   * @brief 
   *	SVD decomposition:
    *	A = U * Sigma * V^T
    */
public:
  SVD(const MatNd &A);

  uint32_t rank_;
  
  MatNd pseudoinverse();
  MatNd null();
  MatNd A_;
  MatNd U_, Sigma_, V_;
  VecNd singular_values_;
  
private:
  void compute();
  MatNd getSigmaPInv();
};

class PTSC 
{
public:
  /**
   * @brief 
   * Constructs the unconstrained PTSC problem:
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   */
  PTSC(const std::vector<Task> &tasks,
       const OsqpSettings &osqp_settings = OsqpSettings());

  /**
   * @brief 
   * Constructs the bound constrained PTSC problem:
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   *                lower_bounds <= x <= upper_bounds
   */
  PTSC( const std::vector<Task> &tasks, 
        const VecNd &lower_bounds, const VecNd &upper_bounds,
        const OsqpSettings &osqp_settings = OsqpSettings() );

  /**
   * @brief 
   * Constructs the fully constrained PTSC problem:
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   *                lower_bounds <= x <= upper_bounds
   *                A_eq x + b_eq = 0
   *                A_ieq x + b_ieq  <= 0
   */
  PTSC( const std::vector<Task> &tasks,
        const MatNd &A_eq, const VecNd &b_eq,
        const MatNd &A_ieq, const VecNd &b_ieq,  
        const VecNd &lower_bounds, const VecNd &upper_bounds,
        const OsqpSettings &osqp_settings = OsqpSettings() );

  void updateProblem(const std::vector<Task> &tasks);

  VecNd solve();

private:
  
  std::vector<Task> tasks_;  //TODO: add check same dimensions!

  uint32_t problem_N_; //problem dimension
  uint32_t N_priorities_; //number of task space control problems - size of tasks_ 

  VecNd lower_bounds_, upper_bounds_;

  MatNd A_eq_, A_ieq_;
  VecNd b_eq_, b_ieq_;
  
  std::vector<MatNd> C_dashed_;
  std::vector<MatNd> A_dashed_;

  enum ProblemType
  {   
    UNCONSTRAINED, 
    BOUND_CONSTRAINED, 
    FULLY_CONSTRAINED
  };  

  ProblemType PTSC_type_;

  void checkBoundDimensions(const VecNd &lower_bounds, const VecNd &upper_bounds);
  VecNd solveUnconstrained();
  VecNd solveConstrained();
  VecNd solveOnePriorityQP( const MatNd &Ai_dashed, const VecNd &bi_dashed, 
                            const MatNd &C_dashed, const VecNd &d_dashed,
                            bool &problem_feasible);

  OsqpSettings osqp_settings_;
};

}

#endif //PTSCEIGEN_HPP_