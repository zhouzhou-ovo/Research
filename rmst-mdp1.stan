functions {
  // Weibull Survival Function
  real weibull_survival(real t, real shape, real scale) {
    return exp(-pow(t / scale, shape));
  }
}

data {
  int<lower=1> N;                   // Number of intervals
  //int<lower=1> J;                   // Number of Groups/treatments
  vector<lower=0>[N] L;             // Left interval
  vector<lower=0>[N] R;             // Right interval
  // array[N] int<lower=1, upper=J> group_idx;  // Group index

  // parameters of hyperprior
  real<lower=0> a_shape;
  real<lower=0> b_shape;
  real<lower=0> a_scale;
  real<lower=0> b_scale;

  // RMST calculation setting
  real<lower=0> tau; // max limited time
  int<lower=1> L_grid; // number of points for trapezoid rule
}

transformed data {
  vector[L_grid + 1] tau_grid;
  for (j in 0:L_grid)
    tau_grid[j + 1] = j * tau / L_grid;
}

parameters {
  // vector<lower=0>[J] shape;
  // vector<lower=0>[J] scale;
  real<lower=0> shape;
  real<lower=0> scale;
}

model {
  shape ~ gamma(a_shape, b_shape);
  scale ~ gamma(a_scale, b_scale);

  for (i in 1:N) {
    // int g = group_idx[i];
    real cdf_right = weibull_cdf(R[i]| shape, scale);
    real cdf_left = weibull_cdf(L[i]| shape, scale);
    target += log(cdf_right - cdf_left + 1e-10);
  }
}

generated quantities {
  //array[J] vector[L_grid + 1] S_curve;
  //array[J] real rmst;
  vector[L_grid + 1] S_curve;
  real rmst;
  
  // for (j in 1:J) {
  //   vector[L_grid + 1] S_vec;
  //   real rmst_val = 0;
  // 
  //   for (t in 1:(L_grid + 1)) {
  //     S_vec[t] = weibull_survival(tau_grid[t], shape[j], scale[j]);
  //     if (t < L_grid) {
  //       real dt = tau_grid[t + 1] - tau_grid[t];
  //       rmst_val += 0.5 * dt * (S_vec[t] + S_vec[t + 1]);
  //     }
  //   }
  //   S_curve[j] = S_vec;
  //   rmst[j] = rmst_val;
  // }
  vector[L_grid + 1] S_vec;
  real rmst_val = 0;

  for (t in 1:(L_grid + 1)) {
      S_vec[t] = weibull_survival(tau_grid[t], shape, scale);
      if (t < L_grid) {
        real dt = tau_grid[t + 1] - tau_grid[t];
        rmst_val += 0.5 * dt * (S_vec[t] + S_vec[t + 1]);
      }
    }
  S_curve = S_vec;
  rmst = rmst_val;
}
