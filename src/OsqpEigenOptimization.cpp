/**
 * @file OsqpEigenOptimization.cpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#include "OsqpEigenOptimization.hpp"

OsqpEigenOpt::OsqpEigenOpt() 
{
}

OsqpEigenOpt::OsqpEigenOpt( const SparseQpProblem &qp_problem, 
                            const OsqpSettings &settings) 
  : n_(qp_problem.A_qp.rows()), 
  m_(qp_problem.upper_bound.rows() + qp_problem.A_eq.rows() + qp_problem.A_ieq.rows()),
  linearConstraintsMatrix_(m_, n_)
{
  initializeSolver(qp_problem, settings);
}

void OsqpEigenOpt::initializeSolver(const SparseQpProblem &qp_problem, 
                                    const OsqpSettings &settings ) 
{
  setSolverSettings(settings);

  solver_.data()->setNumberOfVariables(n_);
  solver_.data()->setNumberOfConstraints(m_);
  
  solver_.data()->clearHessianMatrix();
  solver_.data()->setHessianMatrix(qp_problem.A_qp);
  b_qp_ = qp_problem.b_qp;
  solver_.data()->setGradient(b_qp_);

  solver_.data()->clearLinearConstraintsMatrix();

  SparseMat linearConstraintsMatrix(m_, n_);
  SparseMat identMatrix_n(qp_problem.upper_bound.rows(), qp_problem.upper_bound.rows());
  identMatrix_n.setIdentity();

  setSparseBlock(linearConstraintsMatrix_, identMatrix_n, 0, 0);
  setSparseBlock(linearConstraintsMatrix_, qp_problem.A_eq, identMatrix_n.rows(), 0);
  setSparseBlock(linearConstraintsMatrix_, qp_problem.A_ieq, identMatrix_n.rows() + qp_problem.A_eq.rows(), 0);
  solver_.data()->setLinearConstraintsMatrix(linearConstraintsMatrix_);

  // bounds on optimization variables
  VecNd lower_bound_x = qp_problem.lower_bound;
  VecNd upper_bound_x = qp_problem.upper_bound;
  
  // equality constraint bounds
  VecNd lower_bound_eq = -qp_problem.b_eq;
  VecNd upper_bound_eq = -qp_problem.b_eq;

  // inequality constraint bounds
  VecNd lower_bound_ieq = -inf * VecNd::Ones(qp_problem.b_ieq.size());
  VecNd upper_bound_ieq = -qp_problem.b_ieq;

  VecNd lower_bound(lower_bound_x.size() + lower_bound_eq.size() + lower_bound_ieq.size());
  VecNd upper_bound(upper_bound_x.size() + upper_bound_eq.size() + upper_bound_ieq.size());

  lower_bound << lower_bound_x, lower_bound_eq, lower_bound_ieq;
  upper_bound << upper_bound_x, upper_bound_eq, upper_bound_ieq;

  lower_bound_ = lower_bound;
  upper_bound_ = upper_bound;
  solver_.data()->setBounds(lower_bound_, upper_bound_);

  solver_.clearSolver();
  solver_.initSolver();
}

void OsqpEigenOpt::setGradientAndInit(VecNd &b_qp ) 
{
  b_qp_ = b_qp;
  solver_.data()->setGradient(b_qp);
  solver_.data()->setLinearConstraintsMatrix(linearConstraintsMatrix_);
  solver_.clearSolver();
  solver_.initSolver();
}

VecNd OsqpEigenOpt::solveProblem()
{
  solver_.solveProblem();
  return solver_.getSolution();
}

bool OsqpEigenOpt::checkFeasibility() //Call this after calling solve
{
  return !( (int) solver_.getStatus() == OSQP_PRIMAL_INFEASIBLE || (int) solver_.getStatus() == OSQP_PRIMAL_INFEASIBLE_INACCURATE );
}

void OsqpEigenOpt::setSparseBlock( Eigen::SparseMatrix<double> &output_matrix, const Eigen::SparseMatrix<double> &input_block,
                                          uint32_t i, uint32_t j ) 
{
  if((input_block.rows() > output_matrix.rows() - i) || (input_block.cols() > output_matrix.cols() - j))
  {
    std::cout << "input_block.cols() = " << input_block.cols() << "\n";
    std::cout << "input_block.rows() = " << input_block.rows() << "\n";
    std::cout << "output_matrix.cols() - i = " << output_matrix.cols() - i << "\n";
    std::cout << "output_matrix.rows() - j = " << output_matrix.rows() - j << "\n";
    throw std::runtime_error("setSparseBlock: Can't fit block");
  }
  for (int k=0; k < input_block.outerSize(); ++k)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it(input_block,k); it; ++it)
    {
      output_matrix.insert(it.row() + i, it.col() + j) = it.value();
    }
  }
}

void OsqpEigenOpt::setSolverSettings(const OsqpSettings &settings)
{
  solver_.settings()->setVerbosity(settings.verbosity);
  solver_.settings()->setAlpha(settings.alpha);
  solver_.settings()->setAbsoluteTolerance(settings.absolute_tolerance);
  solver_.settings()->setRelativeTolerance(settings.relative_tolerance);
  solver_.settings()->setWarmStart(settings.warm_start);
  solver_.settings()->setMaxIteration(settings.max_iteration);
  solver_.settings()->setAdaptiveRho(settings.adaptive_rho);
  solver_.settings()->setAdaptiveRhoInterval(settings.adaptive_rho_interval);
  solver_.settings()->setTimeLimit(settings.time_limit);
}