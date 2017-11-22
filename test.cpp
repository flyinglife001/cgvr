/************************************************************************************//**
 *  @file       test.cpp
 *
 *  @brief      Brief descriptinon of test.cpp 
 *
 *  @date       2017-08-03 11:33
 *
 ***************************************************************************************/

#include <armadillo>
#include <iostream>
using namespace arma;

int main()
{
  sp_mat A;
  A.sprandu(20, 20, 0.1);
  cout<<A<<endl;

  cout<<eye(20,20)*A<<endl;

}

