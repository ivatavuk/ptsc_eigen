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
Task::Task(const MatNd &t_A, const VecNd &t_b)
  : A(t_A), b(t_b)
{
  if (A.rows() != b.size())
    throw std::invalid_argument("Task constructor => Matrix sizes do not match!\n");
}

double Task::calcTaskCostValue(const VecNd &x) 
{
  VecNd res = x.transpose() * A.transpose() * A * x +
              2 * x.transpose() * A.transpose() * b + 
              b.transpose() * b;
  return res(0);
}


// -------------- SVD ------------------

SVD::SVD(const MatNd &A) : A_(A)
{
  compute();
}

void SVD::compute()
{
  Eigen::JacobiSVD<MatNd> svd(A_);
  svd.compute(A_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U_ = svd.matrixU();
  V_ = svd.matrixV();
  singular_values_ = svd.singularValues();
  Sigma_ = MatNd::Zero(U_.rows(), V_.rows()) ;
  for(uint32_t i = 0; i < Sigma_.rows() && i < Sigma_.cols(); i++) //TODO: sparse
  {
    Sigma_(i,i) = singular_values_(i);
  }
  rank_ = svd.rank();
}

MatNd SVD::pseudoinverse() 
{	
  return V_ * calcSigmaPInv() * U_.transpose();
}

MatNd SVD::null() 
{
  return V_.block(0, rank_, V_.rows(), V_.cols() - rank_);
}

MatNd SVD::calcSigmaPInv() 
{
  MatNd temp_mat = Sigma_;
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
            const VecNd &lower_bounds, const VecNd &upper_bounds,
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
            const MatNd &A_eq, const VecNd &b_eq,
            const MatNd &A_ieq, const VecNd &b_ieq,  
            const VecNd &lower_bounds, const VecNd &upper_bounds,
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

void PTSC::checkBoundDimensions(const VecNd &lower_bounds, const VecNd &upper_bounds)
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

VecNd PTSC::solve() 
{
  if (PTSC_type_ == UNCONSTRAINED) 
    return solveUnconstrained();
  if (PTSC_type_ == BOUND_CONSTRAINED || PTSC_type_ == FULLY_CONSTRAINED)
    return solveConstrained();
  
  std::runtime_error("PTSC::solve => Unknown PTSC_type");
  return VecNd(); 
}

VecNd PTSC::solveUnconstrained() 
{
  MatNd C_dashed = MatNd::Identity(problem_N_, problem_N_);
  VecNd d_dashed = VecNd::Zero(problem_N_);
  for(auto task : tasks_) 
  {
    MatNd Ai_dashed = task.A * C_dashed;
    VecNd bi_dashed = task.b - task.A * d_dashed;

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

VecNd PTSC::solveConstrained() 
{
  MatNd C_dashed = MatNd::Identity(problem_N_, problem_N_);
  VecNd d_dashed = VecNd::Zero(problem_N_);
  bool problem_feasible = false;
  for(auto task : tasks_) 
  { 
    MatNd Ai_dashed = task.A * C_dashed;
    VecNd bi_dashed = task.b - task.A * d_dashed;
    
    SVD Ai_dashed_svd(Ai_dashed);

    VecNd di = solveOnePriorityQP(Ai_dashed, bi_dashed, C_dashed, d_dashed, 
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

VecNd PTSC::solveOnePriorityQP( const MatNd &Ai_dashed, const VecNd &bi_dashed, 
                                const MatNd &C_dashed, const VecNd &d_dashed,
                                bool &problem_feasible)
{
  DenseQpProblem qp_problem;
  qp_problem.A_qp = Ai_dashed.transpose() * Ai_dashed;
  qp_problem.b_qp = (-bi_dashed.transpose() * Ai_dashed).transpose();

  if(PTSC_type_ == BOUND_CONSTRAINED)
  {
    qp_problem.A_ieq = MatNd::Zero(C_dashed.rows() * 2, C_dashed.cols());
    qp_problem.A_ieq << C_dashed, 
                        -C_dashed;
                        
    qp_problem.b_ieq = VecNd::Zero(d_dashed.rows() * 2);
    qp_problem.b_ieq << d_dashed - upper_bounds_, 
                        -d_dashed + lower_bounds_;
  }
  if(PTSC_type_ == FULLY_CONSTRAINED)
  {
    qp_problem.A_ieq = MatNd::Zero(C_dashed.rows() * 2 + A_ieq_.rows(), C_dashed.cols());
    qp_problem.A_ieq << C_dashed, 
                        -C_dashed, 
                        A_ieq_ * C_dashed;
    qp_problem.b_ieq = VecNd::Zero(d_dashed.rows() * 2 + b_ieq_.rows());
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