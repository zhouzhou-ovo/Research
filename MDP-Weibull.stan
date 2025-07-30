functions {
  // Custom implementation of log(exp(a) - exp(b)) for stability
  real log_sub_exp_custom(real a, real b) {
    // This assumes a > b, which holds true for valid intervals (L < R)
    return a + log1m_exp(a - b);
  }

  // Calculates log(S(L) - S(R)) using the custom function
  real interval_weibull_logcdf(real L, real R, real alpha, real sigma) {
    real log_S_L = -pow(L / sigma, alpha);
    real log_S_R = -pow(R / sigma, alpha);
    return log_sub_exp_custom(log_S_R, log_S_L);
  }
 
  // Inverse CDF for a Weibull distribution
  real weibull_cdf_inverse(real p, real alpha, real sigma) {
    return sigma * pow(-log(1.0 - p), 1.0 / alpha);
  }
}

data {
  // Data Inputs
  int<lower=0> N; // Total number of observations
  vector<lower=0>[N] L; // Left point of exact time and interval-censored
  vector<lower=0>[N] R; // Right point of exact time and interval-censored
  array[N] int is_censored; // 1 if interval-censored, 0 otherwise

  // Model Settings
  int<lower=1> K;  // Truncated level
  real<lower=0> M; // DP concentration parameter
  
  // Settings for generated quantities
  int<lower=1> L_grid; // Number of bins for discrete F
  real<lower=0> tau; // Max limited time
  
  // Hyperprior setting
  real<lower=1e-5> a;
  real<lower=1e-5> b;
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
  // --- Priors and Hyperpriors (non-informative)---
  a1~gamma(a,b);
  a2~gamma(a,b);
  b1~gamma(a,b);
  b2~gamma(a,b);
  
  shape~gamma(a1,b1);
  scale~gamma(a2,b2);
  
  // stick-breaking prior with fixed concentration parameter
  v ~ beta(1, M);

  // Likelihood using marginalization
  for (i in 1:N) {
    vector[K] lp_components;
    for (k in 1:K) {
      if (is_censored[i] == 0) { // Exact time
        lp_components[k] = log(pi[k]) + weibull_lpdf(L[i] | shape[k], scale[k]);
      } else { // Interval-censored
        lp_components[k] = log(pi[k]) + interval_weibull_logcdf(L[i], R[i], shape[k], scale[k]);
      }
    }
    target += log_sum_exp(lp_components);
  }
}

generated quantities {
  // Generation of latent exact time Ti and pj, Fj
  vector[N] T_all;       // Contain all time (including exact time and latent time)
  vector[L_grid] F_curve;       // The generated survival curve F
  real rmst_from_F;             // RMST calculated from the generated F

  {
    vector[L_grid + 1] p_j; // probability of each components
    vector[L_grid + 1] F_probs; // target distribution
    array[K + N] real shape_vec; // new cluster's shape
    array[K + N] real scale_vec; // new cluster's scale
    array[K + N] int count_vec; // number of samples in clustering
    int K_cluster = K;
    
    // Part 1: Generate latent T_i for interval-censored data
    // Initialize the parameters
    for (k in 1:K) {
      shape_vec[k] = shape[k];
      scale_vec[k] = scale[k];
      count_vec[k] = 0;
    }
    // Iterate over every interval-censored data point
    for (i in 1:N) {
      int K_curr = K_cluster;
      vector[K_curr + 1] log_weights;

      // Calculate weight for known clusters
      for (k in 1:K_curr) {
        if (is_censored[i] == 0) { // Exact time
          log_weights[k] = log(count_vec[k] + 1e-9) + weibull_lpdf(L[i]|shape_vec[k],scale_vec[k]);
        } else { // Interval-censored
          log_weights[k] = log(count_vec[k] + 1e-9) + interval_weibull_logcdf(L[i],R[i],shape_vec[k],scale_vec[k]);
        }
    }

    // new clusters with its weight
    real shape_new = gamma_rng(a1, b1);
    real scale_new = gamma_rng(a2, b2);
    if (is_censored[i]==0) {
      log_weights[K_curr + 1] = log(M) + weibull_lpdf(L[i]|shape_new,scale_new);
    } else {
      log_weights[K_curr + 1] = log(M) + interval_weibull_logcdf(L[i],R[i],shape_new,scale_new);
    }
    
    // cluster assignment
    int z_i = categorical_logit_rng(log_weights);

    // combine new cluster and existing cluster
    if (z_i == K_curr + 1) {
      shape_vec[K_cluster + 1] = shape_new;
      scale_vec[K_cluster + 1] = scale_new;
      count_vec[K_cluster + 1] = 1;
      count_vec[z_i] = 0;
      K_cluster += 1;
    } else {
      count_vec[z_i] += 1;
    }

    // Assign or simulate the exact time Ti
    if (is_censored[i] == 0) { // For known exact time
      T_all[i] = L[i];
    } else {
      real u = uniform_rng(weibull_cdf(L[i] | shape_vec[z_i], scale_vec[z_i]),
                         weibull_cdf(R[i] | shape_vec[z_i], scale_vec[z_i]));
      T_all[i] = weibull_cdf_inverse(u,shape_vec[z_i],scale_vec[z_i]);
    }
  }

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
      real current_cdf = 0;
      for (k in 1:K) {
        // current_cdf_mix += pi[k] * weibull_cdf(tau_grid[j] | shape[k], scale[k]);
        current_cdf = weibull_cdf(tau_grid[j]|shape[k],scale[k]);
      }
      p_j[j] = M * (current_cdf - prev_cdf) + bin_counts[j];
      prev_cdf = current_cdf;
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
