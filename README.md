# PTSC Eigen

[![Build Status](https://github.com/ivatavuk/ptsc_eigen/actions/workflows/build_status.yml/badge.svg)](https://github.com/ivatavuk/ptsc_eigen/actions/workflows/build_status.yml)


Prioritized Task-Space control solver using the Eigen linear algebra library, OSQP quadratic programming solver and the OsqpEigen wrapper for OSQP.

Implements both [unconstrained](http://www.delasa.net/iros09/) and [constrained](http://www.delasa.net/feature/index.html) version of the Prioritized Task Space Control algorithm by [Martin de Lasa](http://www.delasa.net/) et al.  

## Notation

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
This project depends on [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page), [`osqp`](https://github.com/osqp/osqp) and [`osqp-eigen`](https://github.com/robotology/osqp-eigen)

It is recommended to build osqp-eigen from source, with:

    -DOSQP_EIGEN_DEBUG_OUTPUT=OFF 
    
to suppress infeasibility warnings.

## 🛠️ Usage

### ⚙️ Build from source

  ```
  git clone https://github.com/ivatavuk/ptsc_eigen.git
  cd ptsc_eigen
  mkdir build
  cd build
  cmake ..
  make
  make install
  ```

## 🖥️ Using the library

### Including the library in your project

**ptsc_eigen** provides native `CMake` support which allows the library to be easily used in `CMake` projects.
**ptsc_eigen** exports a CMake target called `PtscEigen::PtscEigen` which can be imported using the `find_package` CMake command and used by calling `target_link_libraries` as in the following example:
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

  
