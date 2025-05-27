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

data {
  int<lower=0> N;
  vector<lower=0>[N] y;  // Exponential 只能为正，加入 lower bound
}

parameters {
  real<lower=0> lambda;  // lambda 是 rate 参数，必须大于 0
}

model {
  lambda ~ gamma(1, 1);        // prior
  y ~ exponential(lambda);     // likelihood
}