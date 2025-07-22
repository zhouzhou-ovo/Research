// function to calculate the integral of rmst
functions {
  real lnorm_surv_integrand(
    real t,
    real xc,
    array[] real theta,
    array[] real x_r,
    array[] int x_i
  ) {
    real mu = theta[1];
    real sigma = theta[2];
    return 1 - normal_cdf(log(t)|mu, sqrt(sigma));
  }

  real rmst_lnorm(real mu, real sigma, real tau) {
    array[2] real theta;
    theta[1] = mu;
    theta[2] = sigma;

    return integrate_1d(
      lnorm_surv_integrand,
      0.0,
      tau,
      theta,
      rep_array(0.0, 0),   
      rep_array(0, 0)      
    );
  }
}

data {
  // Data
  int<lower=0> N_interval; // number of interval samples
  int<lower=0> N_exact; // number of exact times
  vector<lower=0>[N_interval] L; // left intervals
  vector<lower=0>[N_interval] R; // right intervals
  vector<lower=0>[N_exact] t_exact; // exact times
  
  // Data
  int<lower=1> K; // truncated level
  
  // Hyperprior parameters for base distribution (log-normal) Gamma distributions
  real<lower=0> mu0;
  real<lower=0> lambda0;
  real<lower=0> c0;
  real<lower=0> d0;
  
  // RMST calculation setting
  real<lower=0> tau; // truncated time
  
  int<lower=0> N_grid;
}

parameters {
  // parameters of stick-breaking process
  vector<lower=0, upper=1>[K-1] v;
  
  // parameters of log-normal distribution for each components (K truncated levels)
  vector<lower=0>[K] mu;
  vector<lower=0>[K] sigma2;

  // latent exact times for interval-censored data (follows log-normal)
  vector<lower=0,upper=1>[N_interval] t_interval;
  
  real<lower=0> alpha; // concentration parameter
}

transformed parameters {
  // compute the weight vector pi
  vector[K] pi;
  { // local block for efficiency
    real remaining_stick = 1.0;
    for (k in 1:(K-1)) {
      pi[k] = v[k] * remaining_stick;
      remaining_stick *= (1 - v[k]);
    }
    pi[K] = remaining_stick;
  }
  vector[N_interval] T_interval;
  for (i in 1:N_interval)
    T_interval[i] = L[i] + t_interval[i] * (R[i] - L[i]);
}

model {
  vector[K] log_pi = log(pi); // log weight
  
  // concentration parameter prior
  alpha ~ gamma(1,1);

  //  Stick-breaking parameter prior
  v ~ beta(1, alpha);
  
  // Normal-Gamma prior for (mu, sigma2)
  for (k in 1:K) {
    sigma2[k] ~ inv_gamma(c0, d0);
    mu[k] ~ normal(mu0, sqrt(sigma2[k]/lambda0));
  }
  
  // pdf of exact time
  for (i in 1:N_exact) {
    vector[K] lp_ln;
    for (k in 1:K) {
      lp_ln[k]  = log(pi[k]) + lognormal_lpdf(t_exact[i] | mu[k], sqrt(sigma2[k]));
    }
    target += log_sum_exp(lp_ln);
  }
  
  // pdf of interval
  for (i in 1:N_interval) {
    vector[K] lp_ln;
    for (k in 1:K) {
      lp_ln[k]  = log(pi[k]) + lognormal_lpdf(T_interval[i] | mu[k], sqrt(sigma2[k]));
      // // constrain to [L[i], R[i]]
      // T_interval[i] ~ uniform(L[i], R[i]);
    }
    target += log_sum_exp(lp_ln);
    // target += uniform_lpdf(T_interval[i] | L[i], R[i]);
  }

}

generated quantities {
  real rmst_pred;
  {
    real mu_new;
    real sigma2_new;

    vector[K + 1] prob_vector;
    real sum_pi = sum(pi); 

    for (k in 1:K) {
      prob_vector[k] = pi[k];
    }
    prob_vector[K + 1] = alpha; 
    prob_vector = prob_vector / (sum_pi + alpha);

    int z = categorical_rng(prob_vector);

    if (z <= K) {
      mu_new = mu[z];
      sigma2_new = sigma2[z];
    } else {
      sigma2_new = inv_gamma_rng(c0, d0);
      mu_new = normal_rng(mu0, sqrt(sigma2_new / lambda0));
    }

    rmst_pred = rmst_lnorm(mu_new, sigma2_new, tau);
  }

  real rmst_mean;
  {
    vector[K] rmst_ln;
    for (k in 1:K) {
      rmst_ln[k] = rmst_lnorm(mu[k], sigma2[k], tau);
    }
    rmst_mean = dot_product(rmst_ln, pi);
  }

  vector[N_grid] time_grid;
  vector[N_grid] surv_mix_ln;
  for (n in 1:N_grid) {
    time_grid[n] = tau * n / N_grid;
    real surv_n = 0;
    for (k in 1:K) {
      surv_n += pi[k] * (1 - normal_cdf(log(time_grid[n]) | mu[k], sqrt(sigma2[k])));
    }
    surv_mix_ln[n] = surv_n;
  }
}
