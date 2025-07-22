// function to calculate the integral of rmst
functions {
  real exp_surv_integrand(
    real t,
    real xc,  
    array[] real theta,
    array[] real x_r,
    array[] int x_i
  ) {
    real lambda = theta[1];
    return 1 - exponential_cdf(t|lambda);
  }

  real rmst_exp(real lambda,real tau) {
    array[1] real theta;
    theta[1] = lambda;

    return integrate_1d(
     exp_surv_integrand,
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
  
  // Hyperprior parameters for base distribution (exponential) - Gamma distributions
  real<lower=0> a0;
  real<lower=0> b0;
  
  // RMST calculation setting
  real<lower=0> tau; // truncated time
  int<lower=0> N_grid;
}

parameters {
  // parameters of stick-breaking process
  vector<lower=0, upper=1>[K-1] v;
  
  // parameters of exponentail distribution for each components (K truncated levels)
  vector<lower=0>[K] lambda;

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
  
  
  // Normal-Gamma prior for lambda
  for (k in 1:K) {
    lambda[k] ~ gamma(a0,b0);
  }
  
  // pdf of exact time
  for (i in 1:N_exact) {
    vector[K] lp_exp;
    for (k in 1:K) {
      lp_exp[k]  = log(pi[k]) + exponential_lpdf(t_exact[i] | lambda[k]);
    }
    target += log_sum_exp(lp_exp);
  }
  
  // pdf of interval
  for (i in 1:N_interval) {
    vector[K] lp_exp;
    for (k in 1:K) {
      lp_exp[k]  = log(pi[k]) + exponential_lpdf(T_interval[i] |lambda[k]);
    }
    target += log_sum_exp(lp_exp);
    // target += uniform_lpdf(T_interval[i] | L[i], R[i]);
  }

}

// generated quantities {
//   vector[K] rmst_expo;
// 
//   for (k in 1:K) {
//     rmst_expo[k]  = rmst_exp(lambda[k], tau);
//   }
//   
//   real rmst_mix_exp  = dot_product(rmst_expo, pi);
//   
//     // Posterior survival curve
//   vector[N_grid] time_grid;
//   vector[N_grid] surv_mix_exp;
// 
//   for (n in 1:N_grid) {
//     time_grid[n] = tau * n / N_grid;
//     real surv_n = 0;
//     for (k in 1:K)
//       surv_n += pi[k] * (1 - exponential_cdf(time_grid[n]|lambda[k]));
//     surv_mix_exp[n] = surv_n;
//   }
// } 

generated quantities {
  real rmst_pred;
  {
    real lambda_new;

    // 1. Construct probability vector to choose between existing clusters or a new one
    vector[K + 1] prob_vector;
    real sum_pi = sum(pi); 
    for (k in 1:K) {
      prob_vector[k] = pi[k];
    }
    prob_vector[K + 1] = alpha; 
    prob_vector = prob_vector / (sum_pi + alpha);

    // 2. Draw an index z
    int z = categorical_rng(prob_vector);

    // 3. If z is an existing cluster, use its lambda. If not, draw a new lambda from the prior.
    if (z <= K) {
      // CORRECT: Use lambda[z], not mu[z]
      lambda_new = lambda[z];
    } else {
      // CORRECT: Use gamma_rng, not ~
      lambda_new = gamma_rng(a0, b0);
    }

    // 4. Calculate RMST with the correct function and variable
    // CORRECT: Use rmst_exp and lambda_new
    rmst_pred = rmst_exp(lambda_new, tau);
  }

  // Calculate the mean RMST of the mixture
  real rmst_mean;
  {
    vector[K] rmst_expo;
    for (k in 1:K) {
      // CORRECT: Use lambda[k] and rmst_exp
      rmst_expo[k] = rmst_exp(lambda[k], tau);
    }
    rmst_mean = dot_product(rmst_expo, pi);
  }

  // Calculate the posterior survival curve
  vector[N_grid] time_grid;
  vector[N_grid] surv_mix_exp; // CORRECT: Renamed for clarity
  for (n in 1:N_grid) {
    time_grid[n] = tau * n / N_grid;
    real surv_n = 0;
    for (k in 1:K) {
      // CORRECT: Removed the incorrect log() transform
      surv_n += pi[k] * (1 - exponential_cdf(time_grid[n] | lambda[k]));
    }
    surv_mix_exp[n] = surv_n;
  }
}
