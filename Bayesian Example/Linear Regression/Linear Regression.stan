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
  int<lower=0> N; // number of samples
  int<lower=1> d; // dimension size
  matrix[N,d] X; // design matrix
  vector[N] y; // response vector
  
  // prior parameters
  vector[d] beta0;
  cov_matrix[d] Sigma0;
  real<lower=0> nu0;
  real<lower=0> sigma0;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[d] beta;
  real<lower=0> sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // prior
  beta ~ multi_normal(beta0,Sigma0);
  sigma ~ inv_gamma(nu0/2,nu0*sigma0/2);
  
  // sampling model
  y ~ normal(X*beta, sigma);
}

