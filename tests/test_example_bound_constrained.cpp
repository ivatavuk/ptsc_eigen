/**
 * @file test.cpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#include "PtscEigen.hpp"

int main()
{  
  const Eigen::Matrix<double, 3,1> expected_solution {1, 2, 5};
  auto x_dimension = 3;

  Eigen::MatrixXd A_1(x_dimension, x_dimension), 
                  A_2(x_dimension, x_dimension), 
                  A_3(x_dimension, x_dimension);
  Eigen::VectorXd b_1(x_dimension), b_2(x_dimension), b_3(x_dimension);
  Eigen::VectorXd x_lower(x_dimension), x_upper(x_dimension);

  x_lower << 1, 1, 1;
  x_upper << 5, 5, 5;

  A_1 <<  1, 1, 0, 
          0, 0, 0,
          0, 0, 0;
  b_1 <<  3, 0, 0;
  
  A_2 <<  1, 1, 0, 
          1, 0, 0,
          0, 0, 0;
  b_2 <<  -2, -2, 0;
  
  A_3 <<  1, 1, 0, 
          0, 1, 0,
          0, 1, 1;
  b_3 <<  0, 0, 9;
  
  std::vector<PtscEigen::Task> E_vector{  PtscEigen::Task(A_1, b_1),
                                          PtscEigen::Task(A_2, b_2),
                                          PtscEigen::Task(A_3, b_3) };
  
  PtscEigen::PTSC ptsc(E_vector, x_lower, x_upper);

  Eigen::VectorXd solution = ptsc.solve();
  
  if(expected_solution.isApprox(solution, 1e-5))
    return 0;
  else 
    return -1;
}