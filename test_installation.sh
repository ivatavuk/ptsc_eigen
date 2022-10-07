#!/bin/bash
echo -e "-----------------------------------------"
echo -e "|    Testing ptsc_eigen installation    |"
echo -e "-----------------------------------------\n"
echo -e "---------------------------"
echo -e "|  Unconstrained example  |"
echo -e "---------------------------\n"
./build/test_example_unconstrained
echo ""
echo -e "-------------------------------"
echo -e "|  Bound constrained example  |"
echo -e "-------------------------------\n"
./build/test_example_bound_constrained
echo ""
echo -e "-------------------------------"
echo -e "|  Fully constrained example  |"
echo -e "-------------------------------\n"
./build/test_example_fully_constrained
echo ""