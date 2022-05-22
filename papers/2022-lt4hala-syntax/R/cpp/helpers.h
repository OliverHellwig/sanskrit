#ifndef __H_GLOBAL_HELPERS
#define __H_GLOBAL_HELPERS

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <set>
#include <map>
#include <functional>
#include <numeric>
#include <malloc.h>
#include "types.h"
typedef unsigned short int usint;
#define ALIGNMENT 64

#ifdef _USE_RCPP
#define COUT Rcpp::Rcout
#else
#define COUT std::cout
#endif




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



std::vector<std::string> r2s(const Rcpp::CharacterVector &in){
  std::vector<std::string> out(in.size(),"");
  for(int i=0;i<in.size();i++){
    out[i] = in[i];
  }
  return out;
}


std::vector<std::vector<ftype>> r2s(const Rcpp::NumericMatrix &m)
{
  size_t R = m.rows(), C = m.cols();
  std::vector<std::vector<ftype>> out(R);
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


/*
for memory aligned allocation
*/
inline const inttype aligned_size(const inttype n)
{
	return 32 * (inttype)std::ceil(double(n)/32.0);
}

inline const inttype aligned_size(const inttype n, const size_t alignment)
{
	return ((inttype)alignment) * (inttype)std::ceil(double(n)/double(alignment));
}

template<class T> std::vector<T> numline(std::stringstream &s, const size_t maxRead=0, const inttype increment=0){
	std::string tok;
	std::vector<T> v;
	while (std::getline(s, tok, ' ')) {
		std::stringstream ss(tok);
		T i;
		ss >> i;
		v.push_back(i+increment);
		if(maxRead>0 && v.size() == maxRead){break;}
	}
	return v;
}

/*
various print functions
*/
template<class T> void print(const std::vector<T> &v, const std::string prefix = ""){
	std::cout << prefix;
	for(auto x : v){
		std::cout << " " << x;
	}
	std::cout << std::endl;
}


template <class T> std::vector<std::vector<T>> matrix(const inttype nrow, const inttype ncol) {
	std::vector<std::vector<T>> m;
	for (inttype r = 0; r<nrow; ++r){
		m.push_back(std::vector<T>(ncol, (T)0));
	}
	return m;
}


template <class T> std::vector<T*> aligned_matrix(const inttype nrow, const inttype ncol) {
	std::vector<T*> m;
	inttype cols = aligned_size(ncol,ALIGNMENT);
	for (inttype r = 0; r<nrow; ++r){
		T * p = (T*)_aligned_malloc(cols*sizeof(T), ALIGNMENT);
		for(inttype i=0;i<cols;++i){
			p[i] = (T)0;
		}
		m.push_back( p );
	}
	return m;
}

template <class T> void delete_aligned_matrix(std::vector<T*> &m){
	for(size_t r=0;r<m.size();++r){
		if(m.at(r)!=nullptr){
			_aligned_free(m.at(r));
		}
		m.at(r) = nullptr;
	}
}

template <class T> std::vector<std::vector<std::vector<T>>> tensor3(const inttype dim1, const inttype dim2, const inttype dim3){
	std::vector<std::vector<std::vector<T>>> m(dim1);
	for(inttype i=0;i<dim1;i++){
		m[i] = matrix<T>(dim2,dim3);
	}
	return m;
}


template<class T> std::vector<float> rowsums(const std::vector<std::vector<T>> &m) {
	std::vector<float> rs;
	for (msztype r = 0; r<m.size(); r++) {
		rs.push_back(std::accumulate(m[r].begin(), m.at(r).end(), 0.f));
	}
	return rs;
}

template<class T> float* aligned_rowsums(const std::vector<T*> &m, const size_t rowSize) {
	float* rs = (float*)_aligned_malloc(aligned_size(m.size(),ALIGNMENT)*sizeof(T), ALIGNMENT);
	for(msztype r = 0; r<m.size(); ++r){
		float * p = m[r];
		float sum = 0.f;
		for(size_t i=0;i<rowSize;++i){
			sum+=p[i];
		}
		rs[r] = sum;
	}
	return rs;
}

// = R::colSums(m)
template<class T> std::vector<float> colsums(const std::vector<std::vector<T>> &m) {
	std::vector<float> cs;
	for (vsztype col = 0; col<m[0].size(); col++) {
		float sum = 0.f;
		for (msztype row = 0; row<m.size(); row++) {
			sum += float(m[row][col]);
		}
		cs.push_back(sum);
	}
	return cs;
}

/*
	@param rowSize = # columns
*/
template<class T> float* aligned_colsums(const std::vector<T*> &m, const size_t rowSize) {
	float* cs = (float*)_aligned_malloc((size_t)aligned_size(rowSize)*sizeof(T), ALIGNMENT);
	for(size_t col=0;col<rowSize;++col){
		float sum = 0.f;
		for(msztype row=0;row<m.size();++row){
			sum+=m.at(row)[col];
		}
		cs[col] = sum;
	}
	return cs;
}

// = R::colsums(m[w,])
template<class T> std::vector<float> colsums(const std::vector<std::vector<T>> &m, const std::vector<size_t> &w){
	const size_t cols = m[0].size();
	std::vector<float> cs(cols, 0.f);
	
	for(size_t row : w){
		typename std::allocator_traits<std::vector<T>>::const_pointer ptr = &(m[row])[0];
		std::allocator_traits<std::vector<float>>::pointer pcs = &cs[0];
		for(size_t col=0;col<cols;col++){
			*pcs++ += *ptr++;// m[row][col];
		}
	}
	return cs;
}

template<class T> std::vector<float> colmeans(const std::vector<std::vector<T>> &m, const std::vector<size_t> &w){
	std::vector<float> cs = colsums<T>(m,w);
	for(size_t n=0;n<cs.size();n++){
		cs[n]/=float(w.size());
	}
	return cs;
}





#endif // guard