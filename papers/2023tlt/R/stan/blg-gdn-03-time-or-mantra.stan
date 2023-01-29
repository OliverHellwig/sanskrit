data {
	int noun[5]; // number of nouns per time slot
	int total[5]; // total number of iva=discourse per time slot
}

transformed data {
	vector[5] tz;
	for(t in 1:5){
		tz[t] = (t-3.0)/2.0;
	}
}

parameters{
	real a; // intercept
	real b; // coefficient for time
	real c; // influence of the register
}


model{
	a ~ std_normal();
	b ~ std_normal();
	c ~ normal(0,5); // this is a constant term, and we expect a small number of outcomes, so allow for high variability.
	for(t in 1:5){
		real v;
		if(t <= 2){
			v = c;
		}else{
			v = a + b * tz[t];
		}
		target += binomial_logit_lpmf( noun[t] | total[t], v );
	}
}

generated quantities{
	vector[5] log_lik;
	vector[5] y_rep;
	vector[5] p;
	for(t in 1:5){
		real v;
		if(t <= 2){
			v = c;
		}else{
			v = a + b * tz[t];
		}
		log_lik[t] = binomial_logit_lpmf( noun[t] | total[t], v);
		y_rep[t] = binomial_rng(total[t], inv_logit(v) );
		p[t] = inv_logit(v);
	}
}
