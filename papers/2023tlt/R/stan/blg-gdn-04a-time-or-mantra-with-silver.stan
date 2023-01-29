data {
	int<lower=1> N; // number of records
	vector[N] time; // time slots as integers, 1 = RV etc.
	int<lower=0> P; // position diff. levels
	int<lower=1,upper=5> timeInt[N];
	int<lower=0> V; // number of head POS types
	int<lower=0> H; // number of head label types
	int<lower=0> I; // number of label types
	int<lower=0> nCorrPred; // number of records which are not in the VTB, but have gold annotation
	vector[N] mantra; // 1 = mantra level, 0 = other level
	int<lower=0,upper=1> y[N]; // 1 = NOUN, 0 = other POS
	int positionDiff[N]; // position(iva) - position(head), as factor
	int<lower=0,upper=1> gold[N]; // 0 = silver annotation, 1 = gold annotation
	int<lower=0,upper=1> correct[N]; // 0 = wrong (from silver), 1 = annotation is correct
	vector<lower=0,upper=1>[N] fromVTB; // 1 = from VTB, 0 = manual annotation of a small set of the silver data
	int<lower=1,upper=V> headPOS[N];
	int<lower=1,upper=H> headLabel[N];
	int<lower=1,upper=H> label[N]; // label of each record
}

transformed data{
	vector[N] timeZ;
	for(n in 1:N){
		timeZ[n] = (time[n] - 3.0)/2.0;
	}
}


parameters{
	real a; // intercept
	real A;
	real b; // slope
	real B;
	real c; // mantra level
	real C;
	real D[V]; // head POS
	real E[H]; // head label
	real F[P];
	real G; // (not) from VTB
	real J[I]; // label
	matrix[5,V] HT;
	matrix[5,P] FT; // positionDiff:timeInt
	matrix[5,I] JT; // label:timeInt
}


model{
	real std = 2.0;
	a ~ std_normal();
	b ~ std_normal();
	c ~ normal(0,5); // see the note in time-or-mantra
	
	// error model
	A ~ normal(0,std);
	B ~ normal(0,std);
	C ~ normal(0,std); 
	D ~ normal(0,std); 
	E ~ normal(0,std);
	F ~ normal(0,std); 
	G ~ normal(0,std);
	J ~ normal(0,std);
	for(t in 1:5){
		HT[t,] ~ normal(0,std);
		FT[t,] ~ normal(0,std);
		JT[t,] ~ normal(0,std);
	}
	for(n in 1:N){
		real xc = A + B*timeZ[n]+ G*(1-fromVTB[n]) + C*mantra[n] + E[headLabel[n]] + F[positionDiff[n]] + FT[timeInt[n],positionDiff[n]] + J[label[n]] + JT[timeInt[n],label[n]] + D[headPOS[n]] + HT[timeInt[n],headPOS[n]]; //  is this record correct?
		real xw;
		if(time[n] <= 2){
			xw = c;
		}else{
			xw = a + b*timeZ[n];
		}
		if(gold[n]==1){
			target += bernoulli_logit_lpmf(correct[n] | xc); // correct?
			if(correct[n]==1){
				target += bernoulli_logit_lpmf( y[n] | xw ); // if yes, is a noun preferred?
			}
		}
		else
		{
			// no gold annotation for this record
			target += log_sum_exp(
				bernoulli_logit_lpmf( 1 | xc) + bernoulli_logit_lpmf( y[n] | xw ),
				bernoulli_logit_lpmf( 0 | xc)
			);
		}
	}
}

generated quantities{
	vector[N] log_lik;
	int corr_pred[nCorrPred];
	vector[5] y_rep = rep_vector(0,5);
	vector[N] p;
	vector[N] pred_corr;
	int j = 1;
	for(n in 1:N){
		real xc = A + B*timeZ[n]+ G*(1-fromVTB[n]) + C*mantra[n] + E[headLabel[n]] + F[positionDiff[n]] + FT[timeInt[n],positionDiff[n]] + J[label[n]] + JT[timeInt[n],label[n]] + D[headPOS[n]] + HT[timeInt[n],headPOS[n]]; //  is this record correct?
		real xw = (time[n] <= 2) ? c : a + b*timeZ[n];
		if(gold[n]==1){
			log_lik[n] = bernoulli_logit_lpmf(correct[n] | xc);
			if(correct[n]==1){
				log_lik[n] += bernoulli_logit_lpmf( y[n] | xw );
			}
			if(fromVTB[n]==0){
				corr_pred[j] = bernoulli_rng( inv_logit(xc) );
				j += 1;
			}else{
				// for calculating beta (model comparison)
				if(bernoulli_rng( inv_logit(xc) ) == 1){
					y_rep[timeInt[n]] += bernoulli_rng( inv_logit(xw) );
				}
			}
		}
		else
		{
			// no gold annotation for this record
			log_lik[n] = log_sum_exp(
				bernoulli_logit_lpmf( 1 | xc) + bernoulli_logit_lpmf( y[n] | xw ),
				bernoulli_logit_lpmf( 0 | xc)
			);
		}
		p[n] = inv_logit( xw );
		pred_corr[n] = bernoulli_rng( inv_logit(xc) );
	}
}
