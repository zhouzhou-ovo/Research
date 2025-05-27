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
  int<lower=0> N;
  int<lower=1> d;
  matrix[N,d] Y;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[d] mu;
  cov_matrix[d] Sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
// 正常使用covariance matrix
model {
  // Prior
  mu ~ normal(0, 100);
  Sigma ~ inv_wishart(d + 1, diag_matrix(rep_vector(1, d)));  // Weakly informative prior

  // Likelihood
  for (n in 1:N) {
    Y[n] ~ multi_normal(mu, Sigma);
}
}