// The basic stan model of GMM with MDP prior
data {
  // --- Count ---
  int<lower=1> N;       // Total number of observations
  int<lower=1> J;       // Number of groups
  int<lower=1> K;       // Max number of global mixture components (clusters)

  // --- Data ---
  vector[N] y;          // The observations
  array[N] int<lower=1, upper=J> group_idx; // Index indicating the group for each observation

  // --- Hyperparameters ---
  // For the HDP
  real<lower=0> alpha;  // Group-level concentration parameter
  real<lower=0> gamma;  // Global-level concentration parameter

  // For the base measure H (priors on mu and sigma)
  real mu0;             // Prior mean for the component means
  real<lower=0> tau0;   // Prior scale for the component means
  real<lower=0> nu0;    // Prior df for the component variances
  real<lower=0> sigma0; // Prior scale for the component variances
}

parameters {
  // --- Global-level (G_0) Parameters ---
  // Stick-breaking proportions for the global mixture weights (beta)
  vector<lower=0, upper=1>[K-1] v;
  // Component parameters (the atoms of the global measure)
  ordered[K] mu;            // Component means (ordered for identifiability)
  vector<lower=0>[K] sigma; // Component standard deviations

  // --- Group-level (G_j) Parameters ---
  // Group-specific mixture weights (simplex = sums to 1)
  array[J] simplex[K] pi;
}

transformed parameters {
  // Construct the global mixture weights (beta) from the stick-breaking proportions v
  real epsilon = 1e-6;
  vector[K] beta;
  vector[K] beta_raw;
  beta_raw[1] = v[1];
  for (k in 2:(K-1)) {
    beta_raw[k] = v[k] * prod(1 - v[1:(k-1)]);
  }
  beta_raw[K] = 1 - sum(beta_raw[1:(K-1)]);
  beta = beta_raw + rep_vector(epsilon,K);
  beta = beta/sum(beta);
}

model {
  // --- Priors ---

  // HDP Priors
  v ~ beta(1, gamma); // Stick-breaking proportions for G_0
  for (j in 1:J) {
    pi[j] ~ dirichlet(alpha * beta); // Group-specific weights G_j ~ DP(alpha, G_0)
  }

  // Base Measure (H) Priors for GMM components
  mu ~ normal(mu0, tau0);
  sigma ~ lognormal(log(sigma0), nu0); // Using lognormal for stability

  // --- Likelihood ---
  // We marginalize out the discrete component assignments for each observation
  for (i in 1:N) {
    vector[K] log_lik_components;
    for (k in 1:K) {
      // Calculate log-likelihood of y[i] belonging to component k
      log_lik_components[k] = log(pi[group_idx[i], k]) +
                                normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    // Sum over all components
    target += log_sum_exp(log_lik_components);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  array[N] int<lower=1, upper=K> z; // Inferred cluster assignment for each data point

  for (i in 1:N) {
    vector[K] log_lik_components;
    vector[K] probs;

    for (k in 1:K) {
      log_lik_components[k] = log(pi[group_idx[i], k]) +
                                normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    // Log-likelihood for observation i (for model comparison, e.g., LOO)
    log_lik[i] = log_sum_exp(log_lik_components);

    // Posterior probabilities for component assignment
    probs = softmax(log_lik_components);
    // Sample a component assignment for observation i
    z[i] = categorical_rng(probs);
    // Generate a replicated data point
    y_rep[i] = normal_rng(mu[z[i]], sigma[z[i]]);
  }
}

