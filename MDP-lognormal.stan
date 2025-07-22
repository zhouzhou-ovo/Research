functions {
  // Custom implementation of log(exp(a) - exp(b))
  real log_sub_exp_custom(real a, real b) {
    return a + log1m_exp(a - b);
  }

  // Calculates log(S(L) - S(R)) for a Log-Normal distribution
  real interval_lognormal_logpdf(real L, real R, real mu, real sigma2) {
    return log_sub_exp_custom(
      lognormal_lccdf(L | mu, sqrt(sigma2)),
      lognormal_lccdf(R | mu, sqrt(sigma2))
    );
  }
 
  // Custom implementation of the inverse normal CDF (probit function)
  real normal_icdf_custom(real p) {
    return inv_Phi(p);
  }

  // Inverse CDF for a Log-Normal distribution, now using the custom function
  real lognormal_cdf_inverse(real p, real mu, real sigma2) {
    return exp(normal_icdf_custom(p) * sqrt(sigma2) + mu);
  }
}

data {
  int<lower=0> N_exact;
  vector<lower=0>[N_exact] t_exact;
  int<lower=0> N_interval;
  vector<lower=0>[N_interval] L;
  vector<lower=0>[N_interval] R;
  int<lower=1> K;
  int<lower=1> L_grid;
  real<lower=0> M;
  real<lower=0> tau;
}

parameters {
  vector[K] mu_k;
  vector<lower=1e-9>[K] sigma2_k; // Parameter is variance
  real mu0;
  real<lower=1e-9> sigma2_0;       // Parameter is variance
  real<lower=1e-9> a_sigma;
  real<lower=1e-9> b_sigma;
  vector<lower=0, upper=1>[K-1] v;
}

transformed parameters {
  vector[K] pi;
  {
    real remaining_stick = 1.0;
    for (k in 1:(K-1)) {
      pi[k] = v[k] * remaining_stick;
      remaining_stick *= (1 - v[k]);
    }
    pi[K] = remaining_stick;
  }
}

model {
  // Hyperpriors
  mu0 ~ normal(0, 10);
  sigma2_0 ~ inv_gamma(2, 10);
  a_sigma ~ gamma(2, 1);
  b_sigma ~ gamma(2, 1);
  
  // Priors on Component Parameters
  mu_k ~ normal(mu0, sqrt(sigma2_0)); // FIXED: pass sd
  sigma2_k ~ inv_gamma(a_sigma, b_sigma);
  
  // Stick-breaking prior
  v ~ beta(1, M);

  // Likelihood using Log-Normal distributions
  for (i in 1:N_exact) {
    vector[K] lp_exact;
    for (k in 1:K) {
      lp_exact[k] = log(pi[k]) + lognormal_lpdf(t_exact[i] | mu_k[k], sqrt(sigma2_k[k])); // FIXED: pass sd
    }
    target += log_sum_exp(lp_exact);
  }
  for (i in 1:N_interval) {
    vector[K] lp_interval;
    for (k in 1:K) {
      lp_interval[k] = log(pi[k]) + interval_lognormal_logpdf(L[i], R[i], mu_k[k], sigma2_k[k]);
    }
    target += log_sum_exp(lp_interval);
  }
}

generated quantities {
  vector[N_interval] T_i;
  vector[L_grid] F_curve;
  real rmst_from_F;

  {
    vector[N_exact + N_interval] T_all;
    vector[L_grid + 1] p_j;
    vector[L_grid + 1] F_probs;

    // Part 1: Generate latent T_i
    for (i in 1:N_interval) {
      vector[K] log_probs;
      for (k in 1:K) {
        log_probs[k] = log(pi[k]) + interval_lognormal_logpdf(L[i], R[i], mu_k[k], sigma2_k[k]);
      }
      int z = categorical_logit_rng(log_probs);
      real cdf_L = lognormal_cdf(L[i] | mu_k[z], sqrt(sigma2_k[z])); // FIXED: pass sd
      real cdf_R = lognormal_cdf(R[i] | mu_k[z], sqrt(sigma2_k[z])); // FIXED: pass sd
      real u = uniform_rng(cdf_L, cdf_R);
      T_i[i] = lognormal_cdf_inverse(u, mu_k[z], sigma2_k[z]);
    }
    
    T_all = append_row(t_exact, T_i);

    // Part 2: Generate p_j and F
    vector[L_grid] tau_grid = linspaced_vector(L_grid, 0, tau);
    vector[L_grid + 1] bin_counts = rep_vector(0, L_grid + 1);
    
    for (t in T_all) {
      if (t >= tau) {
        bin_counts[L_grid + 1] += 1;
      } else {
        for (j in 1:L_grid) {
          if (t < tau_grid[j]) {
            bin_counts[j] += 1;
            break;
          }
        }
      }
    }
    
    real prev_cdf = 0;
    for (j in 1:L_grid) {
      real current_cdf_mix = 0;
      for (k in 1:K) {
        current_cdf_mix += pi[k] * lognormal_cdf(tau_grid[j] | mu_k[k], sqrt(sigma2_k[k])); // FIXED: pass sd
      }
      p_j[j] = M * (current_cdf_mix - prev_cdf) + bin_counts[j];
      prev_cdf = current_cdf_mix;
    }
    p_j[L_grid + 1] = M * (1 - prev_cdf) + bin_counts[L_grid + 1];
    
    F_probs = dirichlet_rng(p_j);
    
    real cumulative_prob = 0;
    for (j in 1:L_grid) {
      cumulative_prob += F_probs[j];
      F_curve[j] = 1 - cumulative_prob;
    }
    
    // Part 3: Calculate RMST
    rmst_from_F = 0;
    rmst_from_F += tau_grid[1] * 1.0;
    for(j in 2:L_grid){
      rmst_from_F += (tau_grid[j] - tau_grid[j-1]) * F_curve[j-1];
    }
  }
}
