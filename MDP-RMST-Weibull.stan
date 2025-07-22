data {
  // --- Count ---
  int<lower=1> N; // number of samples
  int<lower=1> K; // truncated level
  int<lower=1> J; // number of groups
  
  // --- Data --- 
  vector[N] L; // observations of left intervals
  vector[N] R; // observations of right intervals
  array[N] int<lower=1, upper=J> group_idx; // group index per observation
  
  // --- MDP Hyperparameters ---
  // prior H on Weibull parameters
  real<lower=0> a_shape;
  real<lower=0> b_shape;
  real<lower=0> a_scale;
  real<lower=0> b_scale;
  // concentration parameter
  real<lower=0> M;
  
  int<lower=1> T;
  vector[T] eval_times;
}

parameters {
  //   // Stick-breaking weights
  // vector<lower=0, upper=1>[K-1] v;

  // Atoms: each mixture component has its own Weibull shape and scale
  vector<lower=0>[K] shape;
  vector<lower=0>[K] scale;
  
  array[J] simplex[K] pi;            // Group-specific mixture weights
}

model {
  // Priors on Weibull parameters (hyperprior from H)
  shape ~ gamma(a_shape, b_shape);
  scale ~ gamma(a_scale, b_scale);

  // Dirichlet prior for each group's mixture weights
  for (g in 1:J) {
    pi[g] ~ dirichlet(rep_vector(M, K));
  }

  // Likelihood
  for (i in 1:N) {
    vector[K] llik;
    for (k in 1:K) {
      real S_l = weibull_cdf(L[i]|shape[k], scale[k]);   // CDF(left)
      real S_r = weibull_cdf(R[i]|shape[k], scale[k]);  // CDF(right)
      llik[k] = log(pi[group_idx[i], k]) + log(S_r - S_l + 1e-10);
    }
    target += log_sum_exp(llik);
  }
}

generated quantities {
  array[G] matrix[T, K] S_component;  // 每个 group、每个 time、每个 component 的生存函数值
  array[G] matrix[T, 1] S_total;      // 每个 group、每个 time 的总生存函数值（加权和）

  for (g in 1:G) {
    for (t in 1:T) {
      real time = eval_times[t];
      real S_sum = 0;

      for (k in 1:K) {
        // Weibull survival function: S(t) = exp(-(t / scale)^shape)
        real S = exp(-pow(time / scale[g, k], shape[g, k]));
        S_component[g, t, k] = S;
        S_sum += pi[g, k] * S;
      }

      S_total[g, t, 1] = S_sum;
    }
  }
}
