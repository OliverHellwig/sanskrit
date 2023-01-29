data {
	int noun[5]; // number of nominal heads per time slot
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
	real b; // slope
}


model{
	a ~ std_normal();
	b ~ std_normal();
	for(t in 1:5){
		noun[t] ~ binomial_logit(total[t], a + b * tz[t] );
	}
}

generated quantities{
	vector[5] log_lik;
	vector[5] y_rep;
	vector[5] p;
	for(t in 1:5){
		real v = a + b*tz[t];
		log_lik[t] = binomial_logit_lpmf( noun[t] | total[t], v );
		y_rep[t] = binomial_rng( total[t], inv_logit(v) );
		p[t] = inv_logit(v);
	}
}
