data {
  // --- Count ---
  int<lower=1> N;       // Number of observations
  int<lower=1> J;       // Number of groups
  int<lower=1> K;       // Number of global components (clusters)

  // --- Data ---
  vector[N] y;                         // Observations
  array[N] int<lower=1, upper=J> group_idx; // Group index per observation

  // --- HDP hyperparameters ---
  real<lower=0> alpha;  // Concentration for group-level DPs
  real<lower=0> gamma;  // Concentration for global DP

  // --- Hyperparameters for prior H over mu0, tau0, nu0, sigma0 ---
  real mu0_prior_mean;
  real<lower=0> mu0_prior_sd;

  real<lower=0> tau0_prior_shape;
  real<lower=0> tau0_prior_rate;

  real<lower=0> nu0_prior_shape;
  real<lower=0> nu0_prior_rate;

  real<lower=0> sigma0_prior_shape;
  real<lower=0> sigma0_prior_rate;
}

parameters {
  // --- Hyperprior H parameters ---
  real mu0;                             // Mean of mu_k
  real<lower=0> tau0;                   // Std dev of mu_k
  real<lower=0> nu0;                    // Shape for sigma prior
  real<lower=0> sigma0;                 // Scale for sigma prior

  // --- Global stick-breaking weights ---
  vector<lower=0, upper=1>[K-1] v;      // Stick-breaking proportions

  // --- Global atoms (G_0 components) ---
  ordered[K] mu;                        // Means of components
  vector<lower=0>[K] sigma;             // Standard deviations

  // --- Group-specific weights ---
  array[J] simplex[K] pi;              // Group-level DPs
}

transformed parameters {
  vector[K] beta_raw;
  vector[K] beta;
  real epsilon = 1e-6;

  beta_raw[1] = v[1];
  for (k in 2:(K-1)) {
    beta_raw[k] = v[k] * prod(1 - v[1:(k-1)]);
  }
  beta_raw[K] = 1 - sum(beta_raw[1:(K - 1)]);
  beta = beta_raw + rep_vector(epsilon, K);
  beta = beta / sum(beta);
}

model {
  // --- Priors on hyperparameters (H) ---
  mu0 ~ normal(mu0_prior_mean, mu0_prior_sd);
  tau0 ~ inv_gamma(tau0_prior_shape, tau0_prior_rate);
  nu0 ~ gamma(nu0_prior_shape, nu0_prior_rate);
  sigma0 ~ gamma(sigma0_prior_shape, sigma0_prior_rate);

  // --- HDP stick-breaking priors ---
  v ~ beta(1, gamma);
  for (j in 1:J) {
    pi[j] ~ dirichlet(alpha * beta);
  }

  // --- G_0: base distribution over components ---
  mu ~ normal(mu0, tau0);
  sigma ~ lognormal(log(sigma0), nu0);

  // --- Likelihood ---
  for (i in 1:N) {
    vector[K] log_lik_components;
    for (k in 1:K) {
      log_lik_components[k] = log(pi[group_idx[i], k]) +
                              normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    target += log_sum_exp(log_lik_components);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  array[N] int<lower=1, upper=K> z;
  vector[K] beta_out = beta;
  array[J] vector[K] pi_out;

  for (j in 1:J) {
    pi_out[j] = pi[j];
  }

  for (i in 1:N) {
    vector[K] log_lik_components;
    vector[K] probs;
    for (k in 1:K) {
      log_lik_components[k] = log(pi[group_idx[i], k]) +
                              normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    log_lik[i] = log_sum_exp(log_lik_components);
    probs = softmax(log_lik_components);
    z[i] = categorical_rng(probs);
    y_rep[i] = normal_rng(mu[z[i]], sigma[z[i]]);
  }
}