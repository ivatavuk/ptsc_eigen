/**
 * @file PtscEigen.cpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#include "PtscEigen.hpp"
#include <iostream>

using PtscEigen::Task;
using PtscEigen::SVD;
using PtscEigen::PTSC;
// -------------- Task ------------------
Task::Task(const Eigen::MatrixXd &t_A, const Eigen::VectorXd &t_b)
  : A(t_A), b(t_b)
{
  if (A.rows() != b.size())
    throw std::invalid_argument("Task constructor => Matrix sizes do not match!\n");
}

double Task::calcTaskCostValue(const Eigen::VectorXd &x) 
{
  Eigen::VectorXd res = x.transpose() * A.transpose() * A * x +
              2 * x.transpose() * A.transpose() * b + 
              b.transpose() * b;
  return res(0);
}


// -------------- SVD ------------------

SVD::SVD(const Eigen::MatrixXd &A) : A_(A)
{
  compute();
}

void SVD::compute()
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_);
  svd.compute(A_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U_ = svd.matrixU();
  V_ = svd.matrixV();
  singular_values_ = svd.singularValues();
  Sigma_ = Eigen::MatrixXd::Zero(U_.rows(), V_.rows()) ;
  for(uint32_t i = 0; i < Sigma_.rows() && i < Sigma_.cols(); i++) //TODO: sparse
  {
    Sigma_(i,i) = singular_values_(i);
  }
  rank_ = svd.rank();
}

Eigen::MatrixXd SVD::pseudoinverse() 
{	
  return V_ * calcSigmaPInv() * U_.transpose();
}

Eigen::MatrixXd SVD::null() 
{
  return V_.block(0, rank_, V_.rows(), V_.cols() - rank_);
}

Eigen::MatrixXd SVD::calcSigmaPInv() 
{
  Eigen::MatrixXd temp_mat = Sigma_;
  for(uint32_t i = 0; i < Sigma_.rows() && i < Sigma_.cols(); i++) 
  {
    if(temp_mat(i,i) != 0.0)
    {
      temp_mat(i,i) = 1.0 / temp_mat(i,i);
    }
  }
  return temp_mat.transpose();
}

// -------------- PTSC ------------------
PTSC::PTSC( const std::vector<Task> &tasks,
            const OsqpSettings &osqp_settings)
  : tasks_(tasks), problem_N_(tasks_[0].A.cols()),  
  N_priorities_(tasks_.size()), 
  osqp_settings_(osqp_settings)
{
  PTSC_type_ = UNCONSTRAINED;
}

PTSC::PTSC( const std::vector<Task> &tasks, 
            const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
            const OsqpSettings &osqp_settings ) 
  : tasks_(tasks), problem_N_(tasks_[0].A.cols()),  
  N_priorities_(tasks_.size()), 
  lower_bounds_(lower_bounds), upper_bounds_(upper_bounds), 
  osqp_settings_(osqp_settings)
{
  PTSC_type_ = BOUND_CONSTRAINED;
  checkBoundDimensions(lower_bounds, upper_bounds);
}

PTSC::PTSC( const std::vector<Task> &tasks,
            const Eigen::MatrixXd &A_eq, const Eigen::VectorXd &b_eq,
            const Eigen::MatrixXd &A_ieq, const Eigen::VectorXd &b_ieq,  
            const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
            const OsqpSettings &osqp_settings )
  : tasks_(tasks), problem_N_(tasks_[0].A.cols()),  
  N_priorities_(tasks_.size()),
  lower_bounds_(lower_bounds), upper_bounds_(upper_bounds),
  A_eq_(A_eq), b_eq_(b_eq), A_ieq_(A_ieq), b_ieq_(b_ieq), 
  osqp_settings_(osqp_settings)
{
  PTSC_type_ = FULLY_CONSTRAINED;
  checkBoundDimensions(lower_bounds, upper_bounds);
}

void PTSC::checkBoundDimensions(const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds)
{
  if (upper_bounds.rows() != lower_bounds.rows()) 
  {
    throw std::runtime_error("PTSC::setBounds => upper and lower bound constraint vectors - sizes do not match");
  }
  else if(upper_bounds.rows() != problem_N_) 
  {
    throw std::runtime_error("PTSC::setBounds => bound constraints - size does not match the problem size");
  }
}

void PTSC::updateProblem(const std::vector<Task> &tasks)
{
  if(tasks.size() != N_priorities_)
  {
    throw std::runtime_error("PTSC::updateProblem => incorrect number of priorities");
  }
  tasks_ = tasks;
}

Eigen::VectorXd PTSC::solve() 
{
  if (PTSC_type_ == UNCONSTRAINED) 
    return solveUnconstrained();
  if (PTSC_type_ == BOUND_CONSTRAINED || PTSC_type_ == FULLY_CONSTRAINED)
    return solveConstrained();
  
  std::runtime_error("PTSC::solve => Unknown PTSC_type");
  return Eigen::VectorXd(); 
}

Eigen::VectorXd PTSC::solveUnconstrained() 
{
  Eigen::MatrixXd C_dashed = Eigen::MatrixXd::Identity(problem_N_, problem_N_);
  Eigen::VectorXd d_dashed = Eigen::VectorXd::Zero(problem_N_);
  for(auto task : tasks_) 
  {
    Eigen::MatrixXd Ai_dashed = task.A * C_dashed;
    Eigen::VectorXd bi_dashed = task.b - task.A * d_dashed;

    SVD Ai_dashed_svd(Ai_dashed);
    d_dashed = d_dashed + C_dashed * Ai_dashed_svd.pseudoinverse() * bi_dashed;
    if (Ai_dashed_svd.rank_ == Ai_dashed.cols()) 
    {
      return d_dashed;
    }
    C_dashed = C_dashed * Ai_dashed_svd.null();
  }
  return d_dashed;
}

Eigen::VectorXd PTSC::solveConstrained() 
{
  Eigen::MatrixXd C_dashed = Eigen::MatrixXd::Identity(problem_N_, problem_N_);
  Eigen::VectorXd d_dashed = Eigen::VectorXd::Zero(problem_N_);
  bool problem_feasible = false;
  for(auto task : tasks_) 
  { 
    Eigen::MatrixXd Ai_dashed = task.A * C_dashed;
    Eigen::VectorXd bi_dashed = task.b - task.A * d_dashed;
    
    SVD Ai_dashed_svd(Ai_dashed);

    Eigen::VectorXd di = solveOnePriorityQP(Ai_dashed, bi_dashed, C_dashed, d_dashed, 
                                            problem_feasible);

    if (!problem_feasible)
    {
      return d_dashed;
    }
    d_dashed = d_dashed + C_dashed * di;
    if (Ai_dashed_svd.rank_ == Ai_dashed.cols()) 
    {
      //TODO Add debug option
      return d_dashed;
    }

    C_dashed = C_dashed * Ai_dashed_svd.null(); 
    
  }
  return d_dashed;
}

Eigen::VectorXd PTSC::solveOnePriorityQP( const Eigen::MatrixXd &Ai_dashed, const Eigen::VectorXd &bi_dashed, 
                                          const Eigen::MatrixXd &C_dashed, const Eigen::VectorXd &d_dashed,
                                          bool &problem_feasible)
{
  DenseQpProblem qp_problem;
  qp_problem.A_qp = Ai_dashed.transpose() * Ai_dashed;
  qp_problem.b_qp = (-bi_dashed.transpose() * Ai_dashed).transpose();

  if(PTSC_type_ == BOUND_CONSTRAINED)
  {
    qp_problem.A_ieq = Eigen::MatrixXd::Zero(C_dashed.rows() * 2, C_dashed.cols());
    qp_problem.A_ieq << C_dashed, 
                        -C_dashed;
                        
    qp_problem.b_ieq = Eigen::VectorXd::Zero(d_dashed.rows() * 2);
    qp_problem.b_ieq << d_dashed - upper_bounds_, 
                        -d_dashed + lower_bounds_;
  }
  if(PTSC_type_ == FULLY_CONSTRAINED)
  {
    qp_problem.A_ieq = Eigen::MatrixXd::Zero(C_dashed.rows() * 2 + A_ieq_.rows(), C_dashed.cols());
    qp_problem.A_ieq << C_dashed, 
                        -C_dashed, 
                        A_ieq_ * C_dashed;
    qp_problem.b_ieq = Eigen::VectorXd::Zero(d_dashed.rows() * 2 + b_ieq_.rows());
    qp_problem.b_ieq << d_dashed - upper_bounds_, 
                        -d_dashed + lower_bounds_, 
                        b_ieq_ + A_ieq_ * d_dashed;
    qp_problem.A_eq = A_eq_ * C_dashed;
    qp_problem.b_eq = b_eq_ + A_eq_ * d_dashed;
  }
  OsqpEigenOpt my_osqp_eigen_opt(qp_problem, osqp_settings_);
  auto solution = my_osqp_eigen_opt.solveProblem();
  problem_feasible = my_osqp_eigen_opt.checkFeasibility();

  return solution;
}