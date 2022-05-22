#include <Rcpp.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "helpers.h"

// [[Rcpp::export]]
Rcpp::IntegerMatrix unigramDistributions(Rcpp::CharacterVector deps_,
                                 Rcpp::IntegerVector control_)
{
  try
  {
    std::vector<std::string> deps = r2s(deps_);
    if(deps.size()!=control_.size()){
      throw "incongruent sizes"; 
    }
    const std::vector<int> control = r2s(control_);
    const int N = 1+*std::max_element(control.begin(),control.end());
    int maxC = 0;
    for(auto d : deps){
      std::stringstream s(d);
      auto l = numline<int>(s, 0,-1);
      maxC = std::max(maxC, *std::max_element(l.begin(),l.end()));
    }
    matrixi m = matrix<int>(maxC+1,N);
    for(size_t i=0;i<deps.size();i++)
    {
      if(i % 1000==0){ Rcpp::Rcout << i << std::endl;}
      const int c = control[i];
      // pa and de are two integer vectors giving the indices of the parent
      // and dependent atoms for one syntactic relation.
      std::stringstream s(deps[i]);
      auto l = numline<int>(s, 0,-1);
      for(auto ix : l){
        m[ix][c]++;
      }
    }
    Rcpp::Rcout<<"Matrix built" << std::endl;
    Rcpp::IntegerMatrix M(m.size(),N);
    for(int r=0;r<m.size();r++){
      for(int c=0;c<N;c++){
        M(r,c) = m[r][c];
      }
    }
    return M;
  }
  catch(const char * msg){
    Rcpp::Rcout << "Error: " << msg << std::endl;
    return Rcpp::IntegerMatrix();
  }
}



// [[Rcpp::export]]
Rcpp::DataFrame unigramLinks(Rcpp::CharacterVector deps_,
                             Rcpp::IntegerVector costs_,
                             int threshold=1)
{
  try
  {
    std::vector<std::string> deps = r2s(deps_);
    std::vector<int> costs = r2s(costs_,0);
    Rcpp::Rcout << "Preprocessing ..." << std::endl;
    int maxC = 0;
    for(size_t i=0;i<deps.size();i++)
    {
      std::stringstream s(deps[i]);
      auto l = numline<int>(s, 0,-1);
      maxC = std::max(maxC, *std::max_element(l.begin(),l.end()));
    }
    Rcpp::Rcout << "Finding sets ..." << std::endl;
    std::vector<std::set<int>> c2r(maxC+1);
    for(size_t i=0;i<deps.size();i++)
    {
      std::stringstream s(deps[i]);
      auto l = numline<int>(s, 0,-1);
      for(size_t j=0;j<l.size();j++){
        c2r[l[j]].insert(i);
      }
    }
    std::vector<std::tuple<int,int,int>> res;
    std::vector<bool> valid(c2r.size(), true);
    Rcpp::Rcout << "Finding intersections for " << c2r.size() << " constituents" << std::endl;
    for(int i=0;i<int(c2r.size())-1;i++){
      Rcpp::Rcout<<i<<std::endl;
      if(!valid[i]){continue;}
      const size_t n = c2r[i].size();
      for(int j=i+1;j<int(c2r.size());j++){
        if(!valid[j]){continue;}
        std::set<int> intersection;
        std::set_intersection(c2r[i].begin(), c2r[i].end(), c2r[j].begin(), c2r[j].end(),
                              std::inserter(intersection, intersection.begin()));
        if(intersection.size()==n && intersection.size()==c2r[j].size()){
          if(costs[i]>costs[j]){
            valid[j] = false;
          }else{
            valid[i] = false;
          }
        }
      }
      
    }
    Rcpp::Rcout << "Main iteration for " << c2r.size() << " constituents" << std::endl;
    for(int i=0;i<int(c2r.size())-1;i++){
      if(i % 50==0){ Rcpp::Rcout << i << std::endl;}
      if(valid[i]==false){
        continue;
      }
      const size_t n = c2r[i].size();
      for(int j=i+1;j<int(c2r.size());j++){
        if(valid[j]==false){
          continue;
        }
        std::set<int> intersection;
        std::set_intersection(c2r[i].begin(), c2r[i].end(), c2r[j].begin(), c2r[j].end(),
                              std::inserter(intersection, intersection.begin()));
        if(intersection.size()==n && intersection.size()==c2r[j].size()){
          Rcpp::Rcout << "Desaster!" << std::endl;
        }
        else if(intersection.size()>=threshold){
          res.push_back(std::make_tuple(i,j,int(intersection.size())));
        }
      }
      
    }
    std::vector<int> ii,jj,n;
    for(size_t i=0;i<res.size();i++){
      ii.push_back(std::get<0>(res[i]));
      jj.push_back(std::get<1>(res[i]));
      n.push_back(std::get<2>(res[i]));
    }
    return Rcpp::DataFrame::create(
      Rcpp::_["i"] = s2r(ii),
      Rcpp::_["j"] = s2r(jj),
      Rcpp::_["n"] = s2r(n,0)
    );
  }
  catch(const char * msg){
    Rcpp::Rcout << "Error: " << msg << std::endl;
    return Rcpp::DataFrame();
  }
}



#define UpdateB(xx,yy) const int zi=z[xx-1]-1, zj=z[yy-1]-1;   \
  if(xx!=ix && yy!=ix){B(zi,zj)++;if(zi!=zj){B(zj,zi)++;}}     \
  else{if(xx==ix){np(zj)++;}else{np(zi)++;}}
#define UpdateC(xx,yy) if(yy>xx){                              \
  const int zi=z[xx-1]-1, zj=z[yy-1]-1;\                       
  if(xx!=ix && yy!=ix){C(zi,zj)++;if(zi!=zj){C(zj,zi)++;}}\    
  else{if(xx==ix){nm(zj)++;}else{nm(zi)++;}}}


// [[Rcpp::export]]
Rcpp::List rebuildBC_old(const int ix, const Rcpp::IntegerVector & x, const Rcpp::IntegerVector &y,
              const Rcpp::IntegerVector &z,
              const int K){
  try
  {
    const int dim=std::max(Rcpp::max(x),Rcpp::max(y));
    if(dim!=z.size()){
      throw "inconsistent dimension for z";
    }
    if(x.size()==0 || x.size()!=y.size()){
      throw "wrong dimensions for x or y";
    }
    Rcpp::IntegerMatrix B = Rcpp::IntegerMatrix(K,K),
      C = Rcpp::IntegerMatrix(K,K);
    Rcpp::IntegerVector np(K), nm(K);
    // initialization
    const int x1 = x[0], y1 = y[0];
    int xprev = 1, yprev;
    while(xprev<=x1){
      const int limit = (xprev==x1) ? y1 : dim+1;
      for(yprev=1;yprev<limit;yprev++){
        UpdateC(xprev,yprev)
      }
      xprev++;
    }
    xprev--;
    yprev--;
    // main loop
    for(int i=0;i<x.size();i++){
      const int xi = x[i], yi = y[i];
      UpdateB(xi,yi)
      if(xi==xprev){
        if(yi>yprev+1){ // a gap in the y values
          for(int ystar=yprev+1;ystar<yi;ystar++){
            UpdateC(xi,ystar)
          }
        }
      }else{ // gap in the x values
        for(int ystar=yprev+1;ystar<=dim;ystar++){
          UpdateC(xprev,ystar)
        }
        for(int ystar=1;ystar<yi;ystar++){ // todo start at xi+1
          UpdateC(xi,ystar)
        }
        // missing x rows?
        for(int xstar=xprev+1;xstar<xi;xstar++){
          for(int ystar=1;ystar<=dim;ystar++){
            UpdateC(xstar,ystar)
          }
        }
      }
      xprev = xi;
      yprev = yi;
    }
    // last entries
    int xstar=xprev,ystar=yprev+1;
    while(xstar<=dim){
      while(ystar<=dim){
        UpdateC(xstar,ystar)
        ystar++;
      }
      ystar=1;
      xstar++;
    }
    return Rcpp::List::create(Rcpp::Named("B") = B,
                              Rcpp::Named("C") = C,
                              Rcpp::Named("np") = np,
                              Rcpp::Named("nm") = nm
    );
  }
  catch(const char * msg){
    Rcpp::Rcout << "Error: " << msg << std::endl;
    return Rcpp::List();
  }
} 
  
  
// [[Rcpp::export]]
Rcpp::NumericMatrix muMl(const int ix, const Rcpp::NumericMatrix &mat,
                     const Rcpp::IntegerVector &z,
                     const int K){
  try
  {
    std::vector<int> A(K);
    const int M = mat.cols();
    Rcpp::NumericMatrix mu(K,M);
    for(int i=0;i<z.size();i++){
      if(i!=ix){
        const int zi = z[i]-1;
        A[zi]++;
        for(int j=0;j<M;j++){
          mu(zi,j)+=mat(i,j);
        }
      }
    }
    for(int i=0;i<K;i++){
      if(A[i]>0){
        for(int j=0;j<M;j++){
          mu(i,j)/=float(A[i]);
        }
      }
    }
    return mu;
  }
  catch(const char * msg){
    Rcpp::Rcout << "Error: " << msg << std::endl;
    return Rcpp::NumericMatrix(1,1);
  }
} 