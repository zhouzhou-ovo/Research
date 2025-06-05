//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;          // number of samples
  array[N] real Y;               // observed data
  int<lower=1> L;          // truncated level (20,30)
  
  // hyperparameters of base distribution P_0
  real mu0;                // prior mean of mu
  real<lower=0> tau0;    // prior variance of mu
  real<lower=0> sigma0;    // prior scale of sigma
  real<lower=0> nu0;       // prior freedom of sigma
  real<lower=0> alpha;    // scaling parameter of DP
}

parameters {
  // parameters of stick-breaking process
  vector<lower=0, upper=1>[L-1] v;
  // simplex[L] pi;
  // parameters of GMM
  vector[L] mu;             // mean
  vector<lower=1e-5>[L] sigma; // variance
}

transformed parameters {
  // compute the weight vector pi
  vector[L] pi;
  { // local block for efficiency
    real remaining_stick = 1.0;
    for (l in 1:(L-1)) {
      pi[l] = v[l] * remaining_stick;
      remaining_stick *= (1 - v[l]);
    }
    pi[L] = remaining_stick;
  }
}

model {
  vector[L] log_pi = log(pi); // log weight

  //  Stick-breaking parameter prior
  v ~ beta(1, alpha);
  // pi ~ dirichlet(rep_vector(alpha / L, L));
  
  // prior of normal parameters
  for (l in 1:L) {
    // sigma prior: InverseGamma(nu0/2, nu0*sigma0^2/2)
    sigma[l] ~ inv_gamma(nu0/2, nu0 * sigma0 / 2);
    
    // mu prior: Normal(mu0, tau0)
    mu[l] ~ normal(mu0, sqrt(tau0));
  }
  
  // log likelihood
  { // local block for efficiency
    // vector[L] log_pi = log(pi);
    vector[L] log_likelihoods;
    for (n in 1:N) {
      for (l in 1:L) {
        log_likelihoods[l] = normal_lpdf(Y[n] | mu[l], sqrt(sigma[l]));
      }
      target += log_sum_exp(log_pi + log_likelihoods);
    }
  }
}

generated quantities {
  // predicited samples
  vector[N] y_pre;
  
  for (n in 1:N) {
    // randomly choose the component
    int z = categorical_rng(pi);
    
    // generate samples from the corresponding component distribution
    y_pre[n] = normal_rng(mu[z], sqrt(sigma[z]));
  }
}

