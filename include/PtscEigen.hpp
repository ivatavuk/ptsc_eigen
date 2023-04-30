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

namespace PtscEigen {

/**
 * @brief Task structure stores a quadratic cost function of the form 
 *              E(x) = ||A*x - b||^2
 */
struct Task
{ 
  /**
   * @brief Construct a new Task object
   * 
   * @param t_A Matrix A
   * @param t_b Vector b 
   */
  Task(const Eigen::MatrixXd &t_A, const Eigen::VectorXd &t_b);

  /**
   * @brief Calculate task cost value
   * 
   * Calculates ||A*x - b||^2
   * 
   * @param x Input vector
   * @return double task cost value
   */
  double calcTaskCostValue(const Eigen::VectorXd &x);

  Eigen::MatrixXd A; /**< Matrix A of a task ||A*x - b||^2 */
  Eigen::VectorXd b; /**< Vector b of a task ||A*x - b||^2 */
};

/**
 * @brief Singular Value Decomposition class
 *          A = U * Sigma * V^T
 */
class SVD 
{
public:
  /**
   * @brief Construct a new SVD object and compute the decomposition U * Sigma * V^T
   * @param A Input matrix
   */
  SVD(const Eigen::MatrixXd &A);

  /**
   * @brief Calculate the pseudoinverse of A using SVD
   * @return Eigen::MatrixXd pinv(A) 
   */
  Eigen::MatrixXd pseudoinverse();

  /**
   * @brief Calculate the nullspace of A using SVD
   * @return Eigen::MatrixXd null(A)
   */
  Eigen::MatrixXd null();

  Eigen::MatrixXd A_; /**< @brief Input matrix A, where A = U * Sigma * V^T */
  Eigen::MatrixXd U_; /**< @brief Matrix U of U * Sigma * V^T */
  Eigen::MatrixXd Sigma_; /**< @brief Matrix Sigma of U * Sigma * V^T */
  Eigen::MatrixXd V_; /**< @brief Matrix V of U * Sigma * V^T */
  Eigen::VectorXd singular_values_; /**< @brief Vector containing singular values of A */
  uint32_t rank_; /**< @brief SVD rank */
    
private:
  /**
   * @brief Compute the SVD decomposition
   * Computes U_, Sigma_ and V_
   */
  void compute();

  /**
   * @brief Calculate the pseudoinverse of the Sigma_ matrix
   * @return Eigen::MatrixXd pinv(Sigma_)
   */
  Eigen::MatrixXd calcSigmaPInv();
};

/**
 * @brief Prioritized Task Space Control class
 * 
 * Stores and solves the unconstrained PTSC problem, 
 * bound constrained PTSC problem and the fully constrained PTSC problem
 */
class PTSC 
{
public:
  /**
   * @brief Construct the unconstrained PTSC problem:
   * 
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   * 
   * @param tasks std::vector<Task> with decreasing priorities
   * @param osqp_settings OsqpSettings object
   */
  PTSC(const std::vector<Task> &tasks,
       const OsqpSettings &osqp_settings = OsqpSettings());

  /**
   * @brief 
   * Constructs the bound constrained PTSC problem:
   * 
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   *                lower_bounds <= x <= upper_bounds
   * 
   * @param tasks std::vector<Task> with decreasing priorities
   * @param lower_bounds lower bound on x
   * @param upper_bounds upper bound on x
   * @param osqp_settings OsqpSettings object
   */
  PTSC( const std::vector<Task> &tasks, 
        const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
        const OsqpSettings &osqp_settings = OsqpSettings() );

  /**
   * @brief 
   * Constructs the fully constrained PTSC problem:
   * 
   *          min       E_N(x)
   *           x
   *          s.t.  E_k(x) = h_k, \foreach k < N 
   *                lower_bounds <= x <= upper_bounds
   *                A_eq x + b_eq = 0
   *                A_ieq x + b_ieq  <= 0
   * 
   * @param tasks std::vector<Task> with decreasing priorities
   * @param A_eq Equality constraint matrix A_eq
   * @param b_eq Equality constraint vector b_eq
   * @param A_ieq Inequality constraint matrix A_ieq
   * @param b_ieq Inequality constraint vector b_ieq
   * @param lower_bounds lower bound on x
   * @param upper_bounds upper bound on x
   * @param osqp_settings OsqpSettings object
   */
  PTSC( const std::vector<Task> &tasks,
        const Eigen::MatrixXd &A_eq, const Eigen::VectorXd &b_eq,
        const Eigen::MatrixXd &A_ieq, const Eigen::VectorXd &b_ieq,  
        const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
        const OsqpSettings &osqp_settings = OsqpSettings() );

  /**
   * @brief Update tasks for the PTSC problem
   * @param tasks std::vector<Task> with decreasing priorities
   */
  void updateProblem(const std::vector<Task> &tasks);

  /**
   * @brief Solve the PTSC problem
   * @return Eigen::VectorXd solution vector x
   */
  Eigen::VectorXd solve();

private:
  std::vector<Task> tasks_;  /**< @brief std::vector<Task> with decreasing priorities */

  uint32_t problem_N_; /**< @brief Dimension of the optimization variable vector x */
  uint32_t N_priorities_; /**< @brief Number of priorities - size of tasks_*/ 

  Eigen::VectorXd lower_bounds_; /**< @brief Lower bounds on x */
  Eigen::VectorXd upper_bounds_; /**< @brief Upper bounds on x */

  Eigen::MatrixXd A_eq_; /**< @brief Equality constraint matrix A_eq_ */
  Eigen::VectorXd b_eq_; /**< @brief Equality constraint vector b_eq_ */
  Eigen::MatrixXd A_ieq_; /**< @brief Inequality constraint matrix A_ieq_ */
  Eigen::VectorXd b_ieq_; /**< @brief Inequality constraint vector b_ieq_ */
  
  std::vector<Eigen::MatrixXd> C_dashed_;
  std::vector<Eigen::MatrixXd> A_dashed_; 

  /**
   * @enum PTSC problem type
   */
  enum ProblemType
  {  
    UNCONSTRAINED, 
    BOUND_CONSTRAINED, 
    FULLY_CONSTRAINED
  };  

  ProblemType PTSC_type_;

  void checkBoundDimensions(const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds);
  Eigen::VectorXd solveUnconstrained();
  Eigen::VectorXd solveConstrained();
  Eigen::VectorXd solveOnePriorityQP( const Eigen::MatrixXd &Ai_dashed, const Eigen::VectorXd &bi_dashed, 
                                      const Eigen::MatrixXd &C_dashed, const Eigen::VectorXd &d_dashed,
                                      bool &problem_feasible);

  OsqpSettings osqp_settings_;
};

}

#endif //PTSCEIGEN_HPP_