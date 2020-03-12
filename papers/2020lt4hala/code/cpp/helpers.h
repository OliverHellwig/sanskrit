#ifndef __H_HELPERS
#define __H_HELPERS
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <set>
#include <chrono>
//#include <boost/math/special_functions/digamma.hpp>
//#include <boost/multiprecision/gmp.hpp>
//#include "arpr/fprecision.h"
// [[Rcpp::plugins(cpp11)]]

typedef int inttype;
typedef float ftype;
typedef std::vector<std::vector<ftype>> matrixf;
typedef std::vector<std::vector<inttype>> matrixi;
typedef std::vector<std::vector<inttype>>::size_type sztype;

const size_t n_reest_features = 5;

/**
 * Converts an Rcpp index vector c(...) into an std vector
 * Note: Each value is decreased by 1!
 */
std::vector<inttype> r2s(const Rcpp::IntegerVector &in, const inttype dec = 1){
	std::vector<inttype> out(in.size(),0);
	for(int i=0;i<in.size();i++){
		out[i] = in[i]-dec; // decrease the R counter, if required.
	}
	return out;
}

std::vector<ftype> r2s(const Rcpp::NumericVector &in){
  std::vector<ftype> out(in.size(),0);
  for(int i=0;i<in.size();i++){
    out[i] = in[i];
  }
  return out;
}


std::vector<std::vector<ftype>> r2s(const Rcpp::NumericMatrix &m)
{
  size_t R = m.rows(), C = m.cols();
  std::vector<std::vector<ftype> > out(R);
  for(size_t r=0;r<R;r++){
    out[r] = std::vector<ftype>(C,(ftype)0);
    for(size_t c=0;c<C;c++){
      out[r][c] = m(r,c);
    }
  }
  return out;
}

std::vector<std::vector<inttype>> r2s(const Rcpp::IntegerMatrix &m)
{
  size_t R = m.rows(), C = m.cols();
  std::vector<std::vector<inttype>> out(R);
  for(size_t r=0;r<R;r++){
    out[r] = std::vector<inttype>(C,0);
    for(size_t c=0;c<C;c++){
      out[r][c] = m(r,c);
    }
  }
  return out;
}

// --- std -> R
Rcpp::IntegerVector s2r(const std::vector<inttype> &in, const inttype inc = 1){
  Rcpp::IntegerVector v(in.size(), 0);
  for(size_t i=0;i<in.size();i++){
    v[i] = in[i]+inc;
  }
  return v;
}

Rcpp::NumericVector s2r(const std::vector<float> &in){
  Rcpp::NumericVector v(in.size(), 0);
  for(size_t i=0;i<in.size();i++){
    v[i] = in[i];
  }
  return v;
}

Rcpp::NumericMatrix s2r(const matrixf &in){
  Rcpp::NumericMatrix m(in.size(), in[0].size());
  for(size_t r=0;r<in.size();r++){
    for(size_t c=0;c<in[0].size();c++){
      m(r,c) = in[r][c];
    }
  }
  return m;
}

Rcpp::IntegerMatrix s2r(const matrixi &in, const inttype inc = 1){
  const size_t rows = in.size(), cols = in[0].size();
  Rcpp::IntegerMatrix m(rows,cols);
  for(size_t r=0;r<rows;r++){
    for(size_t c=0;c<cols;c++){
      m(r,c) = in[r][c];
    }
  }
  return m;
}

void print(const matrixi &m){
  for(size_t r=0;r<m.size();r++){
    for(size_t c=0;c<m[0].size();c++){
      Rcpp::Rcout << m[r][c] << " ";
    }
    Rcpp::Rcout << std::endl;
  }
}

void print(const std::vector<ftype> &v){
  for(size_t n=0;n<v.size();n++){
    Rcpp::Rcout << v[n] << " ";
  }
  Rcpp::Rcout << std::endl;
}

template<class T> std::vector<ftype> rowsums(const std::vector<std::vector<T> > &m) {
  std::vector<ftype> rs;
  for (size_t r = 0; r<m.size(); r++) {
    rs.push_back(std::accumulate(m[r].begin(), m.at(r).end(), (ftype)0) );
  }
  return rs;
}

template<class T> std::vector<ftype> colsums(const std::vector<std::vector<T> > &m) {
  std::vector<ftype> cs;
  const size_t cols = m[0].size();
  for(size_t col=0;col<cols;col++){
    T sum = (T)0;
    for(size_t r = 0; r<m.size(); r++){
      //rs.push_back(std::accumulate(m[r].begin(), m.at(r).end(), (ftype)0) );
      sum+=m[r][col];
    }
    cs.push_back(sum);
  }
  return cs;
}

template<class T>std::vector<std::vector<T>> matrix(inttype nrow, inttype ncol, const T fill = (T)0){
  std::vector<std::vector<T>> m;
  for(inttype r =0;r<nrow;r++){
    m.push_back(std::vector<T>(ncol,fill));
  }
  return m;
}

void check_matrix(const std::vector<std::vector<int>> &m){
  for(sztype i=0;i<m.size();i++){
    for(sztype j=0;j<m[i].size();j++){
      if(m[i][j] < 0){
        std::cout << "Error in matrix!";
      }
    }
  }
}

template<class T>std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &in){
  const size_t rows=in.size(), cols=in[0].size();
  std::vector<std::vector<T>> out = matrix<T>(cols,rows);
  for(size_t r=0;r<rows;++r){
    for(size_t c=0;c<cols;++c){
      out[c][r] = in[r][c];
    }
  }
  return out;
}

/**
 * Copied from 
 * http://web.science.mq.edu.au/~mjohnson/code/digamma.c
 */
double digamma(double x) {
  double result = 0, xx, xx2, xx4;
  assert(x > 0);
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}

inline ftype save_div(const ftype a, const ftype b){
  const ftype eps = 1e-10;
  if(b<eps){
    return a/eps;
  }
  return a/b;
}

std::vector<ftype> hyperparameter_optimization_minka_LOO(const std::vector<ftype> & params, const std::vector<std::vector<inttype>> &counts, const size_t reest_itrs = 5)
{
  std::vector<ftype> params_new = params;
  const size_t K = counts.size();
  for(size_t ritr=0;ritr<reest_itrs;++ritr){
    const ftype param_sum = std::accumulate(params_new.begin(), params_new.end(), 0.f);
    std::vector<ftype> rs = rowsums<inttype>(counts);
    for(size_t m=0;m<params.size();m++){ // for each parameter value ...
      float g = params_new[m], u = 0.f, v = 0.f;
      for(size_t k=0;k<K;k++){
        //u+=(ftype)counts[k][m]/(((ftype)counts[k][m]) - 1.f + g);
        u+= save_div( (ftype)counts[k][m],  (((ftype)counts[k][m]) - 1.f + g) );
        v+= save_div( rs[k], (rs[k]-1.f+param_sum) );
      }
      //params_new[m] = params_new[m]*u/v;
      params_new[m] = params_new[m]*save_div(u,v);
    }
  }
  return params_new;
}


template<class T, class U>std::vector<std::vector<T>> zeros_like(const std::vector<std::vector<U>> &m){
  const size_t rows = m.size(), cols = m[0].size();
  std::vector<std::vector<T>> out(rows);
  for(size_t r=0;r<rows;r++){
    out.at(r) = std::vector<T>(cols, (T)0);
  }
  return out;
}


#endif