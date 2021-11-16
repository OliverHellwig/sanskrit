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

/*
the arrangement of this array is relevant (speedup ~30%).
Data that are accessed sequentially should be close in memory.
Else: cache misses, no locality of reference!
*/
#define INDEX2(a,b,B) b*B+a

typedef std::vector<std::pair<inttype,inttype>> maptype;
const float MIN_SAMPLE = 1e-5f;
#define ALIGNMENT 64

#ifdef _USE_RCPP
#define COUT Rcpp::Rcout
#else
#define COUT std::cout
#endif


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

/*
= numpy.loadtxt
*/
template<class T> std::vector<std::vector<T>> readMatrix(const std::string &path){
	std::ifstream file(path, std::ios::binary);
	if (!file.good()) {
		std::cout << "readMatrix failure: Could not open the file " << path << std::endl;
		return std::vector<std::vector<T>>();
	}
	std::string line;
	std::vector<std::vector<T>> m;
	while (std::getline(file, line)) {
		while (!line.empty() && (line.back() == 10 || line.back() == 13)) {
			line = line.substr(0, line.size() - 1);
		}
		if (!line.empty()) {
			std::stringstream s(line);
			std::vector<T> ln = numline<T>(s);
			if(!m.empty() && ln.size()!=m[0].size()){
				std::cout << "invalid matrix row" << std::endl;
				return std::vector<std::vector<T>>();
			}
			m.push_back(ln);
		}
	}
	return m;
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

/*
row-wise one-hot matrix
*/
matrixf onehot(const std::vector<inttype> &init);

/*
row-wise one-hot matrix with random assignments
*/
std::pair<matrixf, std::vector<inttype>> onehot_random(const inttype rows, const inttype cols);

std::map<inttype,int> table(const std::vector<inttype> &v);

matrixi table(const std::vector<inttype> &init, const std::vector<inttype> &groups);


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


inline std::vector<size_t> which(const std::vector<inttype> &v, const inttype &criterion){
	std::vector<size_t> w;
	std::allocator_traits<std::vector<inttype>>::const_pointer ptr = &v[0];
	for(size_t n=0;n<v.size();n++){
		if(*ptr++ == criterion){
			w.push_back(n);
		}
	}
	return w;
}

inline std::vector<size_t> which(const std::vector<inttype> &v1, const inttype &criterion1, const std::vector<inttype> &v2, const inttype &criterion2){
	std::vector<size_t> w;
	std::allocator_traits<std::vector<inttype>>::const_pointer p1 = &v1[0], p2 = &v2[0];
	for (size_t n = 0; n<v1.size(); n++) {
		if (*p1 == criterion1 && *p2==criterion2) {
			w.push_back(n);
		}
		// DO IT HERE!!
		p1++; p2++;
	}
	return w;
}

// v1 = v1+v2
template <class T> void v_add(std::vector<float> &v1, std::vector<T> &v2){
	for(size_t n=0;n<v1.size();n++){
		v1[n]+=(float)v2[n];
	}
}

template <class T> std::vector<float> v_add_r(const std::vector<float> &v1, std::vector<T> &v2){
	size_t N = v1.size();
	std::vector<float> w(N,0.f);
	for (size_t n = 0; n<v1.size(); n++) {
		w[n] = float(v1[n]+v2[n]);
	}
	return w;
}

// divide by its sum
inline void v_norm(std::vector<float> &v){
	float sum = std::accumulate(v.begin(), v.end(), 0.f);
	for(std::vector<float>::iterator it=v.begin(); it!=v.end(); it++){
		*it/=sum;
	}
}

inline std::vector<float> v_norm_r(std::vector<float> &v) {
	float sum = std::accumulate(v.begin(), v.end(), 0.f);
	std::vector<float> w(v.size(), 0.f);
	for(size_t n=0;n<v.size();n++){
		w[n] = v[n]/sum;
	}
	return w;
}


inline size_t which_max(const std::vector<float> &v){
	return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
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

/*
maximum element of a matrix
*/
template<class T> T maxElement(const std::vector<std::vector<T>> &m){
	if(m.empty() || m[0].empty()){
		std::cout << "empty matrix" << std::endl;
		return 0;
	}
	T maxe = m[0][0];
	for(size_t i=0;i<m.size();++i){
		maxe = std::max(maxe, *std::max_element(m.at(i).begin(),m.at(i).end()));
	}
	return maxe;
}

template<class T> void writeMatrix(const std::vector<std::vector<T>> &m, const std::string &path)
{
	std::cout << "Writing matrix to file ..." << std::endl;
	std::ofstream file(path, std::ios::binary);
	if (!file.good()) {
		std::cout << " FAILURE!" << std::endl;
		return;
	}
	const size_t cols = m.front().size();
	for (size_t r = 0; r<m.size(); r++) {
		for (size_t c = 0; c<cols; c++) {
			if (c>0) { file << " "; }
			file << m[r][c];
		}
		file << std::endl;
	}
	file.close();
}

/* draws N uniformly distributed integers in [0,maxval]
*/
std::vector<inttype> sample(const size_t N, const int maxval);


inline void add(std::vector<float> &dst, const float* x){
	for(size_t n=0;n<dst.size();n++){
		dst[n]+=x[n];
	}
}

inline void sub(std::vector<float> &dst, const float* x){
	for(size_t n=0;n<dst.size();n++){
		dst[n]-=x[n];
	}
}

template<class T> std::vector<T> add(std::vector<T> &in, const T value){
	std::vector<T> dst(in);
	for(size_t i=0;i<dst.size();++i){
		dst[i]+=value;
	}
	return dst;
}

/*
Create a "data frame" with one vector from @d in each column, and stores this data frame at @respath
@param d Each row (!) contains the values of one variable
*/
void write_data(const std::vector<std::vector<inttype>> &d, const std::vector<std::string> &headers, const std::string &respath, const inttype increment=1);

inline std::vector<std::pair<inttype,inttype>>::iterator vecfind(std::vector<std::pair<inttype,inttype>> &v, const inttype key){
	for(std::vector<std::pair<inttype,inttype>>::iterator it = v.begin(); it!=v.end(); ++it){
		if(it->first==key){
			return it;
		}
	}
	return v.end();
}


inline int vecfind(const std::vector<std::pair<inttype,inttype>> *v, const inttype key){
	const size_t n=v->size();
	for(size_t i=0;i<n;++i){
		if(v->at(i).first==key){
			return v->at(i).second;
		}
	}
	return 0;
}


inline void incbi(std::vector<std::vector<maptype>> & tab, const inttype tim, const inttype v1, const inttype v2, const inttype increment=1){ //, const bool warning=false){
	maptype::iterator itr = vecfind(tab[tim][v1], v2);
	if(itr==tab[tim][v1].end()){
		tab[tim][v1].push_back(std::make_pair(v2,increment));
	}else{
		itr->second+=increment;
	}
}

inline void incbi(maptype &tab, const inttype key, const inttype increment = 1){
	maptype::iterator e = tab.end();
	for(maptype::iterator it=tab.begin();it!=e;++it){
		if(it->first==key){
			it->second+=increment;
			return;
		}
	}
	// new key-value
	tab.push_back(std::make_pair(key,increment));
}

inline void decbi(std::vector<std::vector<maptype>> & tab, const inttype tim, const inttype v1, const inttype v2){
	maptype::iterator itr = vecfind(tab[tim][v1], v2);
	if(itr==tab[tim][v1].end()){
		COUT << "decbi failure with time=" << tim << ", v1=" << v1 << ", v2=" << v2 << std::endl;
	}else{
		itr->second--;
	}
}

inline void decbi(maptype & tab, const inttype key){
	maptype::iterator itr = vecfind(tab,key);
	if(itr==tab.end()){
		COUT << "decbi failure with key=" << key << std::endl;
	}else{
		itr->second--;
	}
}

/*
Tests that ...
(a) all values in m are >=0
*/
template<class T> void sanityCheckCountMatrix(const std::vector<std::vector<T>> &m, const std::string &name)
{
	size_t nLess = 0;
	for(size_t i=0;i<m.size();++i){
		for(size_t j=0;j<m.front().size();++j){
			if(m[i][j]<0){ ++nLess;}
		}
	}
	if(nLess>0){
		COUT << "matrix " << name << " has " << nLess << " entries smaller than 0." << std::endl;
	}
}

inline void sanityCheckCountMatrix(const std::vector<float*> &m, const size_t nCols, const std::string &name)
{
	size_t nLess = 0;
	for(size_t i=0;i<m.size();++i){
		for(size_t j=0;j<nCols;++j){
			if(m[i][j]<0){ ++nLess;}
		}
	}
	if(nLess>0){
		COUT << "matrix " << name << " has " << nLess << " entries smaller than 0." << std::endl;
	}
}

/*
Which data configuration should be used?
*/
std::string getPathAffix();

template<class T> T* vec2ptr(const std::vector<T> &v){
	size_t n = v.size();
	T * p = new T[n];
	for(size_t i=0;i<n;++i){
		p[i] = v[i];
	}
	return p;
}

/*
@params count Vector of raw observed values, no grouping
*/
std::pair<float,float> meanAndVarianceRaw(const std::vector<float> &v);

inline float getParam(const std::map<std::string,float> &m, const std::string &key, const float defaultValue){
	auto it = m.find(key);
	if(it==m.end()){
		return defaultValue;
	}
	return it->second;
}

template<class T> size_t ncols(const std::vector<std::vector<T>> &m){
	if(m.empty()){return 0;}
	return m.front().size();
}

std::vector<std::pair<int,int>> getAllowedTimeSlots(const matrixf &tau);
std::vector<std::vector<inttype>> getCitableTexts(const matrixi &conn);
matrixf conn2alpha(const matrixi &conn);
matrixf reestimateAlpha(const std::vector<inttype> &cits, const std::vector<inttype> &tim, const inttype D, const inttype T, const float scaleAlpha=1.f);
std::vector<float> readGoldTimeSlots();
std::vector<maptype> buildEvidenceBigrams(const std::vector<inttype> &Doc, const std::vector<inttype> &Fea);

inline bool shouldSample(const size_t itr, const size_t nItrs){
	return ( (itr>=nItrs-50 && (itr+1) % 10==0) || itr==(nItrs-1) );
}

inline bool shouldSample2(const size_t itr, const size_t nItrs){
	return ( ((itr+1) % 100==0) || itr==(nItrs-1) );
}

bool buildEvidenceBigrams(const std::vector<inttype> &Doc, const std::vector<inttype> &Fea,
	std::vector<inttype> &multiBigramIxes, std::map<std::pair<inttype,inttype>,inttype> &big2ix);

inline std::vector<float> runif(const size_t n, const float mi, const float ma)
{
	std::random_device rd;  // seed for the random number engine
	std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<float> dis(mi,ma);
	std::vector<float> v(n,0.f);
	for(size_t i=0;i<n;++i){
		v[i] = dis(gen);
	}
	return v;
};


#endif // guard