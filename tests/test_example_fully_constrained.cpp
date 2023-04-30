/**
 * @file test.cpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#include "PtscEigen.hpp"
#include "ChronoCall.hpp"

int main()
{  
  const Eigen::Matrix<double, 3,1> expected_solution {1.25, 1.75, 1.75};
  auto x_dimension = 3;

  Eigen::MatrixXd A_1(x_dimension, x_dimension), 
                  A_2(x_dimension, x_dimension), 
                  A_3(x_dimension, x_dimension);

  Eigen::VectorXd b_1(x_dimension), b_2(x_dimension), b_3(x_dimension);
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


  Eigen::VectorXd x_lower(x_dimension), x_upper(x_dimension);
  x_lower << 1, 1, 1;
  x_upper << 5, 5, 5;

  Eigen::MatrixXd A_ieq(1,3), A_eq(1, 3);
  Eigen::VectorXd b_ieq(1), b_eq(1);

  A_eq << 1, 0, 1;
  b_eq << -3;

  A_ieq << 0, 1, 1;
  b_ieq << -3.5;
  std::vector<PtscEigen::Task> E_vector{  PtscEigen::Task(A_1, b_1),
                                          PtscEigen::Task(A_2, b_2),
                                          PtscEigen::Task(A_3, b_3) };
  
  PtscEigen::PTSC ptsc(E_vector, A_eq, b_eq, A_ieq, b_ieq, x_lower, x_upper);

  Eigen::VectorXd solution;
  ChronoCall(microseconds,
    solution = ptsc.solve();
  );
  
  std::cout << "solution =\n" << solution << "\n";
  
  if(expected_solution.isApprox(solution, 1e-5))
    std::cout << "Solved correctly!\n";
  else 
    std::cout << "Solved incorrectly!\n";
  
  return 0;
}