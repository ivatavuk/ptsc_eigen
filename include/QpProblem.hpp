/**
 * @file QpProblem.hpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 *
 *    The QP problem is of the following form:
 *
 *      min 	1 / 2 * x^T * A_qp * x + b_qp^T * x
 *       x
 *
 *      s.t.	A_eq * x + b_eq = 0
 *            A_ieq * x + b_ieq <= 0
 */

#ifndef OP_PROBLEM_HPP_
#define OP_PROBLEM_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>

// QP description
struct DenseQpProblem
{
  Eigen::MatrixXd A_qp, A_eq, A_ieq;
  Eigen::VectorXd b_qp, b_eq, b_ieq, upper_bound, lower_bound;

  DenseQpProblem(){};

  DenseQpProblem(Eigen::MatrixXd t_A_qp, Eigen::VectorXd t_b_qp, Eigen::MatrixXd t_A_eq, Eigen::VectorXd t_b_eq,
                 Eigen::MatrixXd t_A_ieq, Eigen::VectorXd t_b_ieq)
    : A_qp(t_A_qp), A_eq(t_A_eq), A_ieq(t_A_ieq), b_qp(t_b_qp), b_eq(t_b_eq), b_ieq(t_b_ieq){};
  friend std::ostream& operator<<(std::ostream& stream, const DenseQpProblem& qp_problem)
  {
    stream << "QpProblem Cost function:\n";
    stream << "A_qp = \n" << qp_problem.A_qp << "\nb_qp = \n" << qp_problem.b_qp << "\n";
    stream << "\nConstraints:\n";
    if (qp_problem.b_eq.rows() > 0)
      stream << "A_eq = \n" << qp_problem.A_eq << "\nb_eq = \n" << qp_problem.b_eq << "\n";
    if (qp_problem.b_ieq.rows() > 0)
      stream << "A_ieq = \n" << qp_problem.A_ieq << "\nb_ieq = \n" << qp_problem.b_ieq << "\n";
    return stream;
  }
};

struct SparseQpProblem
{
  Eigen::SparseMatrix<double> A_qp, A_eq, A_ieq;
  Eigen::VectorXd b_qp, b_eq, b_ieq, upper_bound, lower_bound;
  SparseQpProblem(Eigen::SparseMatrix<double> t_A_qp, Eigen::VectorXd t_b_qp, Eigen::SparseMatrix<double> t_A_eq,
                  Eigen::VectorXd t_b_eq, Eigen::SparseMatrix<double> t_A_ieq, Eigen::VectorXd t_b_ieq)
    : A_qp(t_A_qp), A_eq(t_A_eq), A_ieq(t_A_ieq), b_qp(t_b_qp), b_eq(t_b_eq), b_ieq(t_b_ieq){};
  SparseQpProblem(Eigen::SparseMatrix<double> t_A_qp, Eigen::VectorXd t_b_qp, Eigen::SparseMatrix<double> t_A_eq,
                  Eigen::VectorXd t_b_eq, Eigen::SparseMatrix<double> t_A_ieq, Eigen::VectorXd t_b_ieq,
                  Eigen::VectorXd t_lower_bound, Eigen::VectorXd t_upper_bound)
    : A_qp(t_A_qp)
    , A_eq(t_A_eq)
    , A_ieq(t_A_ieq)
    , b_qp(t_b_qp)
    , b_eq(t_b_eq)
    , b_ieq(t_b_ieq)
    , upper_bound(t_upper_bound)
    , lower_bound(t_lower_bound){
      // TODO check dimensions??
    };

  SparseQpProblem(DenseQpProblem dense_qp_prob)
  {
    A_qp = dense_qp_prob.A_qp.sparseView();
    A_eq = dense_qp_prob.A_eq.sparseView();
    A_ieq = dense_qp_prob.A_ieq.sparseView();

    b_qp = dense_qp_prob.b_qp;
    b_eq = dense_qp_prob.b_eq;
    b_ieq = dense_qp_prob.b_ieq;
  };

  friend std::ostream& operator<<(std::ostream& stream, const SparseQpProblem& qp_problem)
  {
    stream << "QpProblem Cost function:\n";
    stream << "A_qp = \n" << Eigen::MatrixXd(qp_problem.A_qp) << "\nb_qp = \n" << qp_problem.b_qp << "\n";
    stream << "\nConstraints:\n";
    if (qp_problem.b_eq.rows() > 0)
      stream << "A_eq = \n" << Eigen::MatrixXd(qp_problem.A_eq) << "\nb_eq = \n" << qp_problem.b_eq << "\n";
    if (qp_problem.b_ieq.rows() > 0)
      stream << "A_ieq = \n" << Eigen::MatrixXd(qp_problem.A_ieq) << "\nb_ieq = \n" << qp_problem.b_ieq << "\n";
    return stream;
  }
};

#endif  // OP_PROBLEM_HPP_