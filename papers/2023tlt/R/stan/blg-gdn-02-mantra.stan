data {
	int noun[5]; // number of nouns per time slot
	int total[5]; // total number of iva=discourse per time slot
}

parameters{
	real a; // intercept
	real c; // slope
}


model{
	//real nu = 4;
	//real sigma = 0.5;
	a ~ std_normal();
	c ~ std_normal();
	for(t in 1:5){
		real m = t<=2 ? 1 : 0;
		noun[t] ~ binomial_logit(total[t], a + c * m );
	}
}

generated quantities{
	vector[5] log_lik;
	vector[5] y_rep;
	vector[5] p;
	for(t in 1:5){
		real m = t<=2 ? 1 : 0;
		real v = a + c * m;
		log_lik[t] = binomial_logit_lpmf( noun[t] | total[t], v); 
		y_rep[t] = binomial_rng( total[t], inv_logit(v) );
		p[t] = inv_logit(v);
	}
}
