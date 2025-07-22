functions {
  // Weibull Survival Function
  real weibull_survival(real t, real shape, real scale) {
    return exp(-(t / scale) ^ shape);
  }
}

data {
  int<lower=1> N;                   // Number of Observations
  int<lower=1> J;                   // Number of Group
  vector<lower=0>[N] L;            // Left Interval
  vector<lower=0>[N] R;            // Right Interval
  array[N] int<lower=1, upper=J> group_idx;  // Group Index of each sample

  // Hyperprior Parameters
  real<lower=0> a_shape;
  real<lower=0> b_shape;
  real<lower=0> a_scale;
  real<lower=0> b_scale;

  // MDP Concentration Parameter
  real<lower=0> M;

  // RMST Caculation Setting
  real<lower=0> tau; // Max Limited Time (not truncated levels)
  // ############# whether we need to introduce truncated level? If we use truncated level, we need to use
  // ############# stick-breaking to generate DP.
  int<lower=1> L_grid; // Number of Points for the Trapezoid Rule
}

transformed data {
  // RMST Time Grid 0=tau0<tau1<...<tauL=tau<tau_{L+1}=infty
  vector[L_grid + 1] tau_grid;

  for (j in 0:L_grid) {
    tau_grid[j + 1] = j * tau / L_grid;
  }
}

parameters {
  // Parameters of Base Distribution: Weibull Distribution
  real<lower=0> shape;
  real<lower=0> scale;
  vector<lower=0>[N] T_latent;
}

transformed parameters{
  // save last iteration values
  vector[N] T_latent_prev = T_latent;
}

model {
  // 超先验
  shape ~ gamma(a_shape, b_shape);
  scale ~ gamma(a_scale, b_scale);
  
  // ===========================================================================
  // 核心: Polya Urn Scheme 潜在生存时间更新
  // ===========================================================================
  for (i in 1:N) {
    real log_prob;
    
    // 计算基分布质量 - 添加数值稳定性
    real total_mass = M * (weibull_cdf(R[i]|shape, scale) - 
                      weibull_cdf(L[i]|shape, scale))+1e-5;
    
    // 使用当前迭代已更新的点 (1 到 i-1)
    for (j in 1:(i-1)) {
      if (T_latent[j] >= L[i] && T_latent[j] <= R[i]) {
        total_mass += 1;
      }
    }
    
    // 使用上一次迭代的点 (i+1 到 N)
    for (j in (i+1):N) {
      if (T_latent_prev[j] >= L[i] && T_latent_prev[j] <= R[i]) {
        total_mass += 1;
      }
    }
    
    // 添加小常数避免 total_mass = 0
    total_mass += 1e-5;
    
    // 计算概率
    log_prob = log(total_mass);
    
    // 目标函数 - 对数概率
    target += log_prob;
    
    // 约束T在区间内 - 添加边界检查
    
    T_latent[i] ~ uniform(L[i], R[i]+1e-5);
  }
}

generated quantities {
  // 存储每个组的RMST
  array[J] real rmst;
  
  // 存储生存函数曲线
  array[J] vector[L_grid + 1] S_curve;
  
  // ===========================================================================
  // 核心: p_vector 和 F_vector 计算 (按组别处理)
  // ===========================================================================
  for (j in 1:J) {
    // 步骤10: 计算每个区间的质量 (重命名为 p_vector)
    vector[L_grid + 1] p_vector = rep_vector(0.0, L_grid + 1);
    
    // 步骤11: 从Dirichlet分布生成区间概率 (重命名为 F_vector)
    vector[L_grid + 1] F_vector;
    
    // 只考虑当前组的样本
    for (t in 1:(L_grid + 1)) {
      real lower_bound = (t == 1) ? 0 : tau_grid[t];
      real upper_bound = (t == L_grid + 1) ? tau : tau_grid[t + 1];
      
      // 基分布部分: M * G_θ(区间)
      p_vector[t] = M * (weibull_cdf(upper_bound|shape, scale) - 
                     weibull_cdf(lower_bound|shape, scale));
      
      // 数据部分: ∑I(T_i ∈ 区间) 只针对当前组
      for (i in 1:N) {
        if (group_idx[i] == j) {
          if (T_latent[i] > lower_bound && T_latent[i] <= upper_bound) {
            p_vector[t] += 1;
          }
        }
      }
    }
    
    // 步骤11: 从Dirichlet分布生成区间概率
    F_vector = dirichlet_rng(p_vector);
    
    // 计算生存函数 S(t) = 1 - F(t)
    vector[L_grid + 2] S_vec;
    S_vec[1] = 1.0;  // S(0) = 1
    real F_cum = 0.0;
    
    // 累积概率计算生存函数
    for (t in 1:(L_grid + 1)) {
      F_cum += F_vector[t];
      S_vec[t + 1] = 1.0 - F_cum;
    }
    
    // 保存生存曲线 (0时刻到τ)
    S_curve[j] = S_vec[1:(L_grid + 1)];
    
    // =======================================================================
    // 核心: RMST计算
    // =======================================================================
    rmst[j] = 0;
    for (t in 1:(L_grid + 1)) {
      real t_start = (t == 1) ? 0 : tau_grid[t];
      real t_end = (t == L_grid + 1) ? tau : tau_grid[t + 1];
      real delta = t_end - t_start;
      
      rmst[j] += delta * (S_vec[t] + S_vec[t + 1]) / 2.0;
    }
  }
}
