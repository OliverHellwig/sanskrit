#include <Rcpp.h>
#include "helpers.h"

/**
 * With trainable hyperparameters
 */
// [[Rcpp::export]]
Rcpp::List gibbs_time_or_background(Rcpp::IntegerVector T_in, Rcpp::IntegerVector S_in, Rcpp::IntegerVector F_in, 
                 Rcpp::IntegerVector Docs_in,
                 Rcpp::NumericMatrix Tau, Rcpp::List params,
                 Rcpp::IntegerMatrix C_in, Rcpp::IntegerMatrix D_in, Rcpp::IntegerMatrix E_in,
                 Rcpp::NumericVector beta_in, Rcpp::NumericVector gamma_in, Rcpp::NumericVector delta_in, Rcpp::NumericVector epsilon_in)
{
  // convert the parameters to std
  std::vector<inttype> T = r2s(T_in),
    S = r2s(S_in),
    F = r2s(F_in),
    Docs= r2s(Docs_in);
  matrixf tau = r2s(Tau);
  const inttype nitrs = Rcpp::as<inttype>(params["iterations"]);
  const inttype ii = Rcpp::as<inttype>(params["I"]),
    jj = Rcpp::as<inttype>(params["J"]),
    kk = Rcpp::as<inttype>(params["K"]);
  
  // -------- start of the algorithm ----------------
  const inttype end_burnin = nitrs - 50, post_burnin_itr = 10;
  // random number generator
  std::random_device rd;  // seed for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  // number of documents
  const inttype dd = (inttype)tau.size();
  // count matrices and parameter vectors
  std::vector<float> alpha(ii,0.1f), // inference only
    beta(jj,1.1f), gamma(2, 20.f), delta(kk,0.01f), epsilon(kk,0.01f);
  bool inference = false;
  matrixi A = matrix<inttype>(dd, ii), // document -> time
    B = matrix<inttype>(dd, jj), // document -> genre
    C,D,E;
  if(C_in.nrow()==0){
    C = matrix<inttype>(kk, 2),  // feature -> time or genre
    D = matrix<inttype>(kk,ii),  // time -> feature; transposed for faster access
    E = matrix<inttype>(kk,jj);  // genre -> feature; transposed for faster access
  }else{
    inference = true;
    Rcpp::Rcout << "Inference" << std::endl;
    C = r2s(C_in); D = r2s(D_in); E = r2s(E_in);
    beta = r2s(beta_in); gamma = r2s(gamma_in); delta = r2s(delta_in); epsilon = r2s(epsilon_in);
  }
  std::vector<inttype> G(F.size()); // hidden assignment: time or topic?
  std::uniform_int_distribution<int> dis_g(0, 1);
  for(size_t i = 0; i<F.size(); i++){
    const inttype si = S[i], ti = T[i], di = Docs[i], fi = F[i];
    const int gn = dis_g(gen);
    C[fi][gn]++;
    G[i] = gn;
    if(gn==0){ // time
      A[di][ti]++;
      D[fi][ti]++;
    }else{
      B[di][si]++;
      E[fi][si]++;
    }
  }
  // row sums
  std::vector<float> Dr = colsums<inttype>(D), Er = colsums<inttype>(E);
  
  const inttype M = ii+jj;
  std::vector<float> p(M,0.f), pcs(M, 0.f);
  float delta_sum = std::accumulate(delta.begin(),delta.end(),0.f),
    epsilon_sum = std::accumulate(epsilon.begin(),epsilon.end(),0.f);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  std::vector<inttype> T_all, S_all, G_all, F_all, Doc_all, Itr_all;
  std::vector<inttype> ixes(F.size());
  std::iota(std::begin(ixes), std::end(ixes), 0);
  auto start_time = std::chrono::steady_clock::now();
  int total_duration = 0;
  //const float kk_delta = float(kk)*delta, kk_epsilon = float(kk)*epsilon;
  for(inttype itr = 0; itr < nitrs; itr++)
  {
    if(itr >= end_burnin && itr%post_burnin_itr==0){
      std::cout << "Post-burn-in recording ..." << std::endl;
      T_all.insert(T_all.end(), T.begin(), T.end());
      S_all.insert(S_all.end(), S.begin(), S.end());
      G_all.insert(G_all.end(), G.begin(), G.end());
      F_all.insert(F_all.end(), F.begin(), F.end());
      Doc_all.insert(Doc_all.end(), Docs.begin(), Docs.end());
      std::vector<inttype> iv(T.size(), itr);
      Itr_all.insert(Itr_all.end(), iv.begin(), iv.end());
    }
    int tch = 0, sch = 0, gch = 0;
    double tabsdiff = 0.0;
    std::random_shuffle(ixes.begin(), ixes.end());
    for(size_t n : ixes)
    {
      const inttype dn = Docs[n], tn = T[n], sn = S[n], fn = F[n], gn = G[n];
      C[fn][gn]--;
      if(gn==0){// time
        A[dn][tn]--;
        D[fn][tn]--;
        Dr[tn]--;
      }else{
        B[dn][sn]--;
        E[fn][sn]--;
        Er[sn]--;
      }
      std::fill(p.begin(), p.end(), 0.f);
      // group probabilities for t(ime) and g(enre)
      const float pGt = float(C[fn][0]) + gamma[0], 
        pGg = float(C[fn][1]) + gamma[1];
      std::allocator_traits<std::vector<float>>::pointer p_ptr = &p[0];
      std::allocator_traits<std::vector<float>>::const_pointer p_tau = inference ? &alpha[0] : &tau[dn][0];
      std::allocator_traits<std::vector<float>>::const_pointer pDr = &Dr[0], pEr = &Er[0],
                                                              p_beta = &beta[0];
      std::allocator_traits<std::vector<inttype>>::const_pointer pA = &A[dn][0], 
                                                    pB = &B[dn][0], pD = &D[fn][0], pE = &E[fn][0];
      const float delta_fn = delta[fn], epsilon_fn = epsilon[fn];
      for(inttype i=0;i<ii;i++){
        //const float a = (float(A[dn][i]) + tau[dn][i]);
        const float a = (float(*pA++) + *p_tau++);
        //const float d = (float(D[fn][i]) + delta) / (Dr[i] + float(kk)*delta);
        const float d = (float(*pD++) + delta_fn) / (*pDr++ + delta_sum);
        *p_ptr++ = pGt * a * d;
      }
      for(inttype j=0;j<jj;j++){
        //const float b = (float(B[dn][j]) + beta);
        const float b = (float(*pB++) + *p_beta++);
        //const float e = (float(E[fn][j]) + epsilon) / (Er[j] + float(kk)*epsilon);
        const float e = (float(*pE++) + epsilon_fn) / (*pEr++ + epsilon_sum);
        *p_ptr++ = pGg * b * e;
      }
      
      // evaluation
      std::partial_sum(p.begin(), p.end(), pcs.begin(), std::plus<float>());// cumsum
      float r = dis(gen), highest = pcs.back();
      std::allocator_traits<std::vector<float>>::const_pointer cpcs = &pcs[0];
      for(inttype w = 0; w<M; w++){
        float v = (*cpcs++) / highest;
        if(v >= r)
        {
          inttype ix = w, gnew = 0;
          if(w>=ii){
            gnew = 1;
            ix = w-ii;
          }
          inttype tnew = tn, snew = sn;
          G[n] = gnew;
          C[fn][gnew]++;
          if(gnew==0){ // time
            tnew = ix;
            A[dn][tnew]++;
            D[fn][tnew]++;
            Dr[tnew]++;
          }else{
            snew = ix;
            B[dn][snew]++;
            E[fn][snew]++;
            Er[snew]++;
          }
          T[n] = tnew;
          S[n] = snew;
          
          if(gnew==0 && tnew!=tn){tch++;}
          if(gnew==1 && snew!=sn){sch++;}
          if(gnew!=gn){gch++;}
          tabsdiff+=double(std::abs(int( tnew-tn) ))/double(ii);
          break;
        }
      }
    }
    // re-estimate the hyperparameters
    // Minka 2003, eq. 65
    if(itr < end_burnin){
      if(!inference){
        beta  = hyperparameter_optimization_minka_LOO(beta, B, n_reest_features);
        // no gamma optimization? else, too early convergence on time/background
        //gamma = hyperparameter_optimization_minka_LOO(gamma,C, n_reest_features);
        delta = hyperparameter_optimization_minka_LOO(delta, transpose(D), n_reest_features);
        epsilon = hyperparameter_optimization_minka_LOO(epsilon, transpose(E), n_reest_features);
        delta_sum = std::accumulate(delta.begin(),delta.end(),0.f);
        epsilon_sum = std::accumulate(epsilon.begin(),epsilon.end(),0.f);
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
      Rcpp::Rcout << itr << ", duration: " << dur << "ms [" << ms << "], t-changes: " << tch << "[abs: " << tabsdiff << "], s-changes: " << sch << ", g-changes: " << gch << std::endl;
    }
  }
  
  // --- Return the data
  return Rcpp::List::create(Rcpp::Named("result") = Rcpp::DataFrame::create(Rcpp::_["T"] = s2r(T_all), Rcpp::_["S"] = s2r(S_all), Rcpp::_["G"] = s2r(G_all),
                                        Rcpp::_["F"] = s2r(F_all), Rcpp::_["Doc"] = s2r(Doc_all)),
                                        Rcpp::Named("A") = s2r(A),
                                        Rcpp::Named("B") = s2r(B),
                                        Rcpp::Named("C") = s2r(C),
                                        Rcpp::Named("D") = s2r(D),
                                        Rcpp::Named("E") = s2r(E),
                                        Rcpp::Named("beta") = s2r(beta),
                                        Rcpp::Named("gamma") = s2r(gamma),
                                        Rcpp::Named("delta") = s2r(delta),
                                        Rcpp::Named("epsilon") = s2r(epsilon),
                                        Rcpp::Named("tau") = s2r(tau)
  );
}
