functions {
  // Custom implementation of log(exp(a) - exp(b)) for stability
  real log_sub_exp_custom(real a, real b) {
    // This assumes a > b, which holds true for valid intervals (L < R)
    return a + log1m_exp(a - b);
  }

  // Calculates log(S(L) - S(R)) using the custom function
  real interval_weibull_logpdf(real L, real R, real alpha, real sigma) {
    real log_S_L = -pow(L / sigma, alpha);
    real log_S_R = -pow(R / sigma, alpha);
    return log_sub_exp_custom(log_S_L, log_S_R);
  }
 
  // Inverse CDF for a Weibull distribution
  real weibull_cdf_inverse(real p, real alpha, real sigma) {
    return sigma * pow(-log(1.0 - p), 1.0 / alpha);
  }
}

data {
  // Data Inputs
  int<lower=0> N_exact;
  vector<lower=0>[N_exact] t_exact;
  int<lower=0> N_interval;
  vector<lower=0>[N_interval] L;
  vector<lower=0>[N_interval] R;

  // Model Settings
  int<lower=1> K;
  real<lower=0> M; // DP concentration parameter
  
  // Settings for generated quantities
  int<lower=1> L_grid; // Number of bins for discrete F
  real<lower=0> tau;
}

parameters {
  // Model Parameters
  vector<lower=1e-5>[K] shape;
  vector<lower=1e-5>[K] scale;

  // MDP Hyperparameters
  real<lower=1e-5> a1;
  real<lower=1e-5> b1;
  real<lower=1e-5> a2;
  real<lower=1e-5> b2;
  
  vector<lower=0, upper=1>[K-1] v; // Stick-breaking proportions
}

transformed parameters {
  vector[K] pi; // The mixture weights pi are now transformed parameters
  { // Local block for efficiency
    real remaining_stick = 1.0;
    for (k in 1:(K-1)) {
      pi[k] = v[k] * remaining_stick;
      remaining_stick *= (1 - v[k]);
    }
    pi[K] = remaining_stick;
  }
}

model {
  // --- Priors and Hyperpriors ---
  a1~gamma(2,1);
  a2~gamma(2,1);
  b1~gamma(2,1);
  b2~gamma(2,1);
  
  shape~gamma(a1,b1);
  scale~gamma(a2,b2);
  
  // stick-breaking prior
  v ~ beta(1, M);

  // Likelihood using marginalization
  for (i in 1:N_exact) {
    vector[K] lp_exact;
    for (k in 1:K) {
      lp_exact[k] = log(pi[k]) + weibull_lpdf(t_exact[i] | shape[k], scale[k]);
    }
    target += log_sum_exp(lp_exact);
  }
  for (i in 1:N_interval) {
    vector[K] lp_interval;
    for (k in 1:K) {
      lp_interval[k] = log(pi[k]) + interval_weibull_logpdf(R[i], L[i], shape[k], scale[k]);
    }
    target += log_sum_exp(lp_interval);
  }
}

generated quantities {
  // --- Simulate one step of the Gibbs Algorithm ---
  vector[N_interval] T_i;       // Compute latent times
  vector[L_grid] F_curve;       // The generated survival curve F
  real rmst_from_F;             // RMST calculated from the generated F

  { // Use a local block to keep intermediate variables private
    vector[N_exact + N_interval] T_all;
    vector[L_grid + 1] p_j;
    vector[L_grid + 1] F_probs;

    // Part 1: Generate latent T_i for interval-censored data
    for (i in 1:N_interval) {
      vector[K] log_probs;
      for (k in 1:K) {
        log_probs[k] = log(pi[k]) + interval_weibull_logpdf(R[i], L[i], shape[k], scale[k]);
      }
      int z = categorical_logit_rng(log_probs);
      real cdf_L = weibull_cdf(L[i] | shape[z], scale[z]);
      real cdf_R = weibull_cdf(R[i] | shape[z], scale[z]);
      real u = uniform_rng(cdf_L, cdf_R);
      T_i[i] = weibull_cdf_inverse(u, shape[z], scale[z]);
    }
    
    T_all = append_row(t_exact, T_i);

    // Part 2: Generate p_j and then F
    vector[L_grid] tau_grid = linspaced_vector(L_grid, 0, tau);
    vector[L_grid + 1] bin_counts = rep_vector(0, L_grid + 1);
    
    for (t in T_all) {
      if (t >= tau) { // Use >= to match the last bin
        bin_counts[L_grid + 1] += 1;
      } else {
        for (j in 1:L_grid) {
          if (t < tau_grid[j]) { // Use < to ensure it falls in the right bin
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
        current_cdf_mix += pi[k] * weibull_cdf(tau_grid[j] | shape[k], scale[k]);
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
    
    // --- Part 3: Calculate RMST from the generated F_curve ---
    // This is the area under the step-function survival curve
    rmst_from_F = 0;
    rmst_from_F += tau_grid[1] * 1.0; // Area of first rectangle, S(0)=1
    for(j in 2:L_grid){
        // Add area of rectangle: width * height
        // Height is the survival probability at the start of the interval
        rmst_from_F += (tau_grid[j] - tau_grid[j-1]) * F_curve[j-1];
    }
  }
}
