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
 * @brief Task structure stores a quadratic cost function of the form E = ||A*x - b||^2 
 * @details \f[ E(\boldsymbol{x}) = ||\boldsymbol{A} \boldsymbol{x} - \boldsymbol{b}||^2 \f]
 */
struct Task
{ 
  /**
   * @brief Construct a new Task object
   * @param t_A Matrix \a A
   * @param t_b Vector \a b 
   */
  Task(const Eigen::MatrixXd &t_A, const Eigen::VectorXd &t_b);

  /**
   * @brief Calculate task cost value
   * @param x Input vector
   * @return Task cost value
   */
  double calcTaskCostValue(const Eigen::VectorXd &x);

  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};

/**
 * @brief Singular Value Decomposition class
 *          A = U * Sigma * V^T
 * 
 * @details \f[ \boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T \f]
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

  uint32_t getRank() const;
    
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

  Eigen::MatrixXd A_;
  Eigen::MatrixXd U_;
  Eigen::MatrixXd Sigma_;
  Eigen::MatrixXd V_;
  Eigen::VectorXd singular_values_;
  uint32_t rank_;
};

/**
 * @brief Prioritized Task Space Control class
 * @details Stores and solves the unconstrained PTSC problem, 
 * bound constrained PTSC problem and the fully constrained PTSC problem
 */
class PTSC 
{
public:
  /**
   * @brief Construct the unconstrained PTSC problem
   * @details
   * \f[\begin{equation}
	 *       \begin{aligned}
	 *	          h_i = & \ \underset{\boldsymbol{x}}{\text{min}} & & E_N(\boldsymbol{x})\\
	 *	          & \ \ \text{s.t.} & & E_k(\boldsymbol{x}) = h_k, \ \forall k < N
	 *       \end{aligned}
   *     \end{equation}\f]
   * @param tasks std::vector<Task> with decreasing priorities
   * @param osqp_settings OsqpSettings object
   */
  PTSC(const std::vector<Task> &tasks,
       const OsqpSettings &osqp_settings = OsqpSettings());

  /**
   * @brief 
   * Constructs the bound constrained PTSC problem
   * @details
   * \f[\begin{equation}
	 *       \begin{aligned}
	 *	          h_i = & \ \underset{\boldsymbol{x}}{\text{min}} & & E_N(\boldsymbol{x})\\
	 *	          & \ \ \text{s.t.} & & E_k(\boldsymbol{x}) = h_k, \ \forall k < N\\
	 *	          & & & \underline{\boldsymbol{x}} \leq \boldsymbol{x} \leq \overline{\boldsymbol{x}}
	 *       \end{aligned}
   *     \end{equation}\f]
   * @param tasks std::vector<Task> with decreasing priorities
   * @param lower_bounds lower bound on \a x
   * @param upper_bounds upper bound on \a x
   * @param osqp_settings OsqpSettings object
   */
  PTSC( const std::vector<Task> &tasks, 
        const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
        const OsqpSettings &osqp_settings = OsqpSettings() );

  /**
   * @brief 
   * Constructs the fully constrained PTSC problem
   * @details
   * \f[\begin{equation}
	 *       \begin{aligned}
	 *	          h_i = & \ \underset{\boldsymbol{x}}{\text{min}} & & E_N(\boldsymbol{x})\\
	 *	          & \ \ \text{s.t.} & & E_k(\boldsymbol{x}) = h_k, \ \forall k < N\\
	 *	          & & & \underline{\boldsymbol{x}} \leq \boldsymbol{x} \leq \overline{\boldsymbol{x}}\\
	 *	          & & & \boldsymbol{A_{eq}}\ \boldsymbol{x} + \boldsymbol{b_{eq}} = 0\\
	 *	          & & & \boldsymbol{A_{ieq}}\ \boldsymbol{x} + \boldsymbol{b_{ieq}} \leq  0
	 *       \end{aligned}
   *     \end{equation}\f]
   * @param tasks std::vector<Task> with decreasing priorities
   * @param A_eq Equality constraint matrix \a A_eq
   * @param b_eq Equality constraint vector \a b_eq
   * @param A_ieq Inequality constraint matrix \a A_ieq
   * @param b_ieq Inequality constraint vector \a b_ieq
   * @param lower_bounds lower bound on \a x
   * @param upper_bounds upper bound on \a x
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
   * @return Eigen::VectorXd solution vector \a x
   */
  Eigen::VectorXd solve();

private:
  std::vector<Task> tasks_;  /**< @brief std::vector<Task> with decreasing priorities */

  uint32_t problem_N_; /**< @brief Dimension of the optimization variable vector x */
  uint32_t N_priorities_; /**< @brief Number of priorities - size of tasks_*/ 

  Eigen::VectorXd lower_bounds_; /**< @brief Lower bounds on \a x */
  Eigen::VectorXd upper_bounds_; /**< @brief Upper bounds on \a x */

  Eigen::MatrixXd A_eq_;
  Eigen::VectorXd b_eq_;
  Eigen::MatrixXd A_ieq_;
  Eigen::VectorXd b_ieq_;
  
  std::vector<Eigen::MatrixXd> C_dashed_;
  std::vector<Eigen::MatrixXd> A_dashed_; 

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