# PTSC Eigen

Prioritized Task-Space control solver using the Eigen linear algebry library, OSQP quadratic programming solver and the OsqpEigen wrapper for OSQP.

Implements both [unconstrained](http://www.delasa.net/iros09/) and [constrained](http://www.delasa.net/feature/index.html) version of the Prioritized Task Space Control algorithm by [Martin de Lasa](http://www.delasa.net/) et al. 

Note that the PTSC problem subject solely to equality constraints can be solved by an unconstrained PTSC solver, by using the constraint term as a task with the highest priority, and checking if the constraint is satisfied. If the constraint is not satisfied, the constrained problem in infeasible. 

## Mathematical notation
Lets denote the vector of optimization variables as $\boldsymbol{x}$ and the number of priorities of the PTSC problem as $N$.

The quadratic cost function of the $i$-th priority has the following form:

$$E_i(\boldsymbol{x}) = ||\boldsymbol{A}_i\boldsymbol{x} - \boldsymbol{b}_i||^2$$

and is defined by a matrix $\boldsymbol{A}_i$ and a vector $\boldsymbol{b}_i$.

Solution to the $i$-th priority of the PTSC problem is:

$$\begin{equation}
	\begin{aligned}
		h_i = & \ \underset{\boldsymbol{x}}{\text{min}} & & E_i(\boldsymbol{x})\\
		& \ \ \text{s.t.} & & E_k(\boldsymbol{x}) = h_k, \ \forall k < i\\
		& & & \boldsymbol{A_{eq}}\ \boldsymbol{x} + \boldsymbol{b_{eq}} = 0\\
		& & & \boldsymbol{A_{ieq}}\ \boldsymbol{x} + \boldsymbol{b_{ieq}} \leq  0
	\end{aligned}
\end{equation}$$

where $h_i$ is the optimal cost of the $i$-th priority.
Solution to the $i$-th priority is given by minimizing the $i$-th cost function $E_i(\boldsymbol{x})$, while the cost functions of higher priorities $E_k(\boldsymbol{x}), \forall k < i$ remain in their respective minimums.

## 📄 Dependences
This project depends on [`osqp`](https://github.com/osqp/osqp) and [`osqp-eigen`](https://github.com/robotology/osqp-eigen)

It is recommended to build osqp-eigen from source, with:

    -DOSQP_EIGEN_DEBUG_OUTPUT=OFF 
    
to suppress infeasibility warnings.

## 🛠️ Usage

### ⚙️ Build from source

  ```
  git clone https://github.com/ivatavuk/ptsc_eigen.git
  cd ptsc_eigen
  mkdir build
  cmake ..
  make
  make install
  ```

## 🖥️ Using the library

### Including the library in your project

**ptsc-eigen** provides native `CMake` support which allows the library to be easily used in `CMake` projects.
**ptsc-eigen** exports a CMake target called `PtscEigen::PtscEigen` which can be imported using the `find_package` CMake command and used by calling `target_link_libraries` as in the following example:
```cmake
project(myproject)
find_package(PtscEigen REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PtscEigen::PtscEigen)
```

### Minimal example

For minimal examples on different types of PTSC problems check the test folder

## 📝 License

Materials in this repository are distributed under the following license:

> All software is licensed under the BSD 3-Clause License.

  
