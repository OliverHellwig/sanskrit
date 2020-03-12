#include <Rcpp.h>
#include "helpers.h"

/**
 * Perplexity of the time model
 */
double _ppl_time(const std::vector<inttype> &T, // sampled times
               const std::vector<inttype> &F, // feature types
               const std::vector<inttype> &Docs, // doc numbers
               const matrixi &A, // doc -> time
               const matrixi &D, // time -> feature
               const matrixf &tau,
               const std::vector<float> &delta) // priors for time -> feature
{
  matrixi d = transpose(D);
  matrixf omega = zeros_like<float,inttype>(A),
    theta = zeros_like<float,inttype>(d);
  float eps = 1e-10;
  for(size_t i=0;i<A.size();i++){// for each document ...
    inttype sum = std::accumulate(A.at(i).begin(), A.at(i).end(), 0);
    float tausum = std::accumulate(tau.at(i).begin(), tau.at(i).end(), 0.f);
    for(size_t j=0;j<A.at(0).size();j++){
      omega[i][j] = ( float(A[i][j]) + tau[i][j])/( float(sum) + tausum );
      if(omega[i][j]==0){
        omega[i][j] = eps;
      }
    }
  }
  float deltasum = std::accumulate(delta.begin(), delta.end(), 0.f);
  for(size_t i=0;i<d.size();i++){// for each time bin ...
    inttype sum = std::accumulate(d.at(i).begin(), d.at(i).end(), 0);
    
    for(size_t j=0;j<d.at(0).size();j++){
      theta[i][j] = (float(d[i][j])+delta[j])/(float(sum)+deltasum);
      if(theta[i][j]==0){
        theta[i][j] = eps;
      }
    }
  }
  double lp = 0;
  for(size_t n=0;n<T.size();n++){
    const inttype tn=T[n],fn=F[n],dn=Docs[n];
    lp-=log(omega[dn][tn]*theta[tn][fn]);
  }
  return lp/double(T.size());
}

// [[Rcpp::export]]
void testxx(Rcpp::IntegerMatrix u){
  matrixi uu = r2s(u);
}

// [[Rcpp::export]]
double ppl_time(Rcpp::IntegerVector T, // sampled times
                Rcpp::IntegerVector F, // feature types
                Rcpp::IntegerVector Docs, // doc numbers
                 Rcpp::IntegerMatrix A, // doc -> time
                 Rcpp::IntegerMatrix D, // time -> feature
                 Rcpp::NumericMatrix tau,
                 Rcpp::NumericVector delta) // 
{
  return _ppl_time(r2s(T), r2s(F), r2s(Docs),
                   r2s(A), r2s(D), 
                   r2s(tau), r2s(delta));
}



/**
 * "Inner" routine
 */
void _gibbs_time(std::vector<inttype> &T, std::vector<inttype> &F, 
                std::vector<inttype> &Docs, 
                matrixi &A,
                matrixf &tau, 
                matrixi &D, std::vector<float> &delta,
                std::vector<inttype> &T_all, std::vector<inttype> &F_all, std::vector<inttype> &Doc_all,
                const inttype I, const inttype K, const inttype nitrs)
{
  const inttype end_burnin = nitrs - 50, post_burnin_itr = 10;
  // random number generator
  std::random_device rd;  // seed for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  // number of documents
  const inttype dd = (inttype)tau.size();
  // count matrices and parameters
  A = matrix<inttype>(dd, I); // document -> time;
  std::vector<float> alpha(I,0.1f);
  bool inference = false;
  if(D.size()==0){
    delta = std::vector<float>(K, 0.01f);
    D = matrix<inttype>(K,I);  // time -> feature; transposed for faster access
  }else{
    // use pre-trained data
    Rcpp::Rcout << "Using pre-trained A,D,delta" << std::endl;
    //D = r2s(D_in);
    //delta = r2s(delta_in);
    inference = true;
  }
  float delta_sum = std::accumulate(delta.begin(),delta.end(),0.f);
  for(size_t i = 0; i<F.size(); i++){
    const inttype ti = T[i], di = Docs[i], fi = F[i];
    A[di][ti]++;
    D[fi][ti]++;
  }
  //print(A);
  // row sums
  std::vector<float> Dr = colsums<inttype>(D);
  
  std::vector<float> p(I,0.f),pcs(I, 0.0);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  
  std::vector<inttype> ixes(F.size());
  std::iota(std::begin(ixes), std::end(ixes), 0);
  auto start_time = std::chrono::steady_clock::now();
  int total_duration = 0;
  
  for(inttype itr = 0; itr < nitrs; itr++)
  {
    if(itr >= end_burnin && itr%post_burnin_itr==0){
      std::cout << "Post-burn-in recording ..." << std::endl;
      T_all.insert(T_all.end(), T.begin(), T.end());
      F_all.insert(F_all.end(), F.begin(), F.end());
      Doc_all.insert(Doc_all.end(), Docs.begin(), Docs.end());
      //std::vector<inttype> iv(T.size(), itr);
      //Itr_all.insert(Itr_all.end(), iv.begin(), iv.end());
    }
    int tch = 0;
    double tabsdiff = 0.0;
    std::random_shuffle(ixes.begin(), ixes.end());
    for(size_t n : ixes)
    {
      const inttype dn = Docs[n], tn = T[n], fn = F[n];
      A[dn][tn]--;
      D[fn][tn]--;
      Dr[tn]--;
      std::fill(p.begin(), p.end(), 0.f);
      std::allocator_traits<std::vector<float>>::pointer p_ptr = &p[0];
      // when performing inference, don't use prior information
      std::allocator_traits<std::vector<float>>::const_pointer p_tau = inference ? &alpha[0] : &tau[dn][0];
      std::allocator_traits<std::vector<float>>::const_pointer pDr = &Dr[0];
      std::allocator_traits<std::vector<inttype>>::const_pointer pA = &A[dn][0], pD = &D[fn][0];
      const float delta_fn = delta[fn];
      for(inttype i=0;i<I;i++){
        //const float a = (float(A[dn][i]) + tau[dn][i]);
        const float a = (float(*pA++) + *p_tau++);
        //const float d = (float(D[fn][i]) + delta) / (Dr[i] + float(kk)*delta);
        const float d = (float(*pD++) + delta_fn) / (*pDr++ + delta_sum);
        *p_ptr++ = a * d;
      }
      
      // sampling the new time
      std::partial_sum(p.begin(), p.end(), pcs.begin(), std::plus<float>());// cumsum
      float r = dis(gen), highest = pcs.back();
      std::allocator_traits<std::vector<float>>::const_pointer cpcs = &pcs[0];
      for(inttype w = 0; w<I; w++){
        float v = (*cpcs++) / highest;
        if(v >= r)
        {
          const inttype tnew = w;
          A[dn][tnew]++;
          D[fn][tnew]++;
          Dr[tnew]++;
          T[n] = tnew;
          if(tnew!=tn){tch++;}
          tabsdiff+=double(std::abs(int( tnew-tn) ))/double(I);
          break;
        }
      }
    }
    if(itr < end_burnin){
      if(!inference){
        delta = hyperparameter_optimization_minka_LOO(delta, transpose(D), n_reest_features);
        delta_sum = std::accumulate(delta.begin(),delta.end(),0.f);
      }else{
        
        std::vector<float> _tau = hyperparameter_optimization_minka_LOO(tau[0], A, n_reest_features);
        for(size_t u=0;u<tau.size();u++){
          tau.at(u) = _tau;
        }
      }
    }
    // duration
    if(itr>0 && itr%10==0){
      auto end_time = std::chrono::steady_clock::now();
      int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      start_time = end_time;
      total_duration+=ms;
      int dur = int(double(total_duration)/double(itr+1));
      Rcpp::Rcout << itr << ", duration: " << dur << "ms [" << ms << "], t-changes: " << tch << "[abs: " << tabsdiff << "]" << std::endl;
    }
  }
}

/**
 * This is a standard LDA model with the subjective temporal prior.
 */
// [[Rcpp::export]]
Rcpp::List gibbs_time(Rcpp::IntegerVector T_in, Rcpp::IntegerVector F_in, 
                 Rcpp::IntegerVector Docs_in,
                 Rcpp::NumericMatrix Tau, Rcpp::List params, 
                 Rcpp::IntegerMatrix D_in, Rcpp::NumericVector delta_in)
{
  // convert the parameters to std
  std::vector<inttype> T = r2s(T_in),
    F = r2s(F_in),
    Docs= r2s(Docs_in);
  matrixf tau = r2s(Tau);
  /*const inttype nitrs = Rcpp::as<inttype>(params["iterations"]);
  const inttype ii = Rcpp::as<inttype>(params["I"]),
    kk = Rcpp::as<inttype>(params["K"]);*/
  matrixi D = r2s(D_in), A;
  std::vector<float> delta = r2s(delta_in);
  
  std::vector<inttype> T_all, F_all, Doc_all; // post-burn-in results
  
  _gibbs_time(T, F, Docs, A,
              tau, D, delta,
              T_all, F_all, Doc_all,
              Rcpp::as<inttype>(params["I"]), Rcpp::as<inttype>(params["K"]), Rcpp::as<inttype>(params["iterations"]));
  
  // --- Return the data
  return Rcpp::List::create(Rcpp::Named("result") = Rcpp::DataFrame::create(Rcpp::_["T"] = s2r(T_all),
                                        Rcpp::_["F"] = s2r(F_all), Rcpp::_["Doc"] = s2r(Doc_all)),
                                        Rcpp::Named("A") = s2r(A),
                                        Rcpp::Named("D") = s2r(D),
                                        Rcpp::Named("delta") = delta,
                                        Rcpp::Named("tau") = s2r(tau)
  );
}
