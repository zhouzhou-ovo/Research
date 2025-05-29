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
  int<lower=1> N; // number of samples
  int<lower=1> J; // number of groups
  array[N] int<lower=1, upper=J> group; // group index of samples
  vector[N] y;
  
  // hyperparameters
  real<lower=0> nu0;
  real<lower=0> sigma0;
  real<lower=0> eta0;
  real<lower=0> tau0;
  real mu0;
  real<lower=0> gamma0;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real mu; // total mean
  real<lower=0> sigma; // total variance
  real<lower=0> tau; // between-group variance
  vector[J] theta; // group mean
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // Hyperpriors
  sigma ~ inv_gamma(nu0 / 2, nu0 * sigma0 / 2);
  tau ~ inv_gamma(eta0 / 2, eta0 * tau0 / 2);
  mu ~ normal(mu0, sqrt(gamma0));

  // Group-level parameters
  theta ~ normal(mu, sqrt(tau));

  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(theta[group[n]], sqrt(sigma));
  }
}
