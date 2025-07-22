functions {
  // Weibull 生存函数 S(t) = P(T > t)
  real weibull_survival(real t, real shape, real scale) {
    return exp(-pow(t / scale, shape));
  }
}

data {
  // 观测数据
  int<lower=1> N;         // 观测总数
  vector<lower=0>[N] L;   // 区间左端点 (Left)
  vector<lower=0>[N] R;   // 区间右端点 (Right)

  // DP 混合模型设置
  int<lower=1> K;         // 截断的混合成分最大数量 (Truncation level)
  real<lower=0> M;        // DP 集中度参数 (Concentration parameter, a.k.a. alpha)

  // 超先验参数 (Hyperpriors for base distribution)
  real<lower=0> a_shape;  // shape 的 Gamma 先验参数 a
  real<lower=0> b_shape;  // shape 的 Gamma 先验参数 b
  real<lower=0> a_scale;  // scale 的 Gamma 先验参数 a
  real<lower=0> b_scale;  // scale 的 Gamma 先验参数 b

  // RMST 计算设置
  real<lower=0> tau;      // 限制时间点 tau
  int<lower=1> L_grid;    // RMST 计算的网格点数量
}

transformed data {
  // RMST 计算的时间网格
  vector[L_grid + 1] tau_grid;
  for (j in 0:L_grid) {
    tau_grid[j + 1] = j * tau / L_grid;
  }
}

parameters {
  // Stick-breaking 过程的参数
  // 每个 eta[k] 代表在剩余的 "stick" 中掰下的比例
  vector<lower=0, upper=1>[K - 1] eta;

  // K 个混合成分的参数 (对应论文中的 G_0)
  vector<lower=0>[K] shape; // 每个成分的 Weibull shape
  vector<lower=0>[K] scale; // 每个成分的 Weibull scale
}

transformed parameters {
  // 从 eta 计算出每个成分的权重 w
  vector[K] w; // Mixture weights
  vector[K] log_w; // Log of mixture weights
  {
    // 修正: log1m_eta 的长度应该是 K-1，与 eta 匹配
    vector[K - 1] log1m_eta; 
    log1m_eta = log1m(eta);
    
    // stick-breaking 过程
    // log(w_1) = log(eta_1)
    // log(w_2) = log(eta_2) + log(1 - eta_1)
    // ...
    // log(w_k) = log(eta_k) + sum_{j=1}^{k-1} log(1 - eta_j)
    log_w[1] = log(eta[1]);
    if (K > 2) { // 增加一个判断，避免 K=2 时出现问题
      for (k in 2:(K-1)) {
        log_w[k] = log(eta[k]) + sum(log1m_eta[1:(k-1)]);
      }
    }
    // 最后一个成分的权重是 "stick" 的剩余部分
    log_w[K] = sum(log1m_eta); // sum(log1m_eta) 的长度是 K-1
    
    // 转换为普通权重 (用于 generated quantities)
    w = exp(log_w);
  }
}

model {
  // 先验分布 (Priors)
  // stick-breaking 过程的先验
  eta ~ beta(1, M);
  
  // K 个成分参数的超先验 (Hyperpriors on mixture components)
  shape ~ gamma(a_shape, b_shape);
  scale ~ gamma(a_scale, b_scale);

  // 似然函数 (Likelihood)
  // 对每个观测值，我们计算其在 K 个成分下的混合似然
  for (i in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      real log_prob_k;
      
      // --- **关键改动 2: 在似然计算中区分两种数据类型** ---
      if (L[i] == R[i]) {
        // 情况 A: 精确事件时间 (L == R)
        // 似然由概率密度函数 (PDF) 的对数给出
        log_prob_k = weibull_lpdf(L[i] | shape[k], scale[k]);
      } else {
        // 情况 B: 区间删失事件 (L < R)
        // 似然由 CDF 的差值给出
        log_prob_k = log_diff_exp(
          weibull_lcdf(R[i] | shape[k], scale[k]),
          weibull_lcdf(L[i] | shape[k], scale[k])
        );
      }
      
      lps[k] = log_w[k] + log_prob_k;
    }
    // 使用 log_sum_exp 将所有成分的对数似然加权求和
    // 这等价于对离散的聚类分配进行了边缘化，这是 Stan 的标准做法
    target += log_sum_exp(lps);
  }
}

generated quantities {
  // 计算后验的生存曲线和 RMST
  vector[L_grid + 1] S_curve; // 混合模型的生存曲线
  real rmst;                  // 最终的 RMST

  // 混合生存曲线是各成分生存曲线的加权平均
  // S_mix(t) = sum_{k=1 to K} w_k * S_k(t)
  for (t_idx in 1:(L_grid + 1)) {
    real current_t = tau_grid[t_idx];
    real s_val = 0;
    for (k in 1:K) {
      s_val += w[k] * weibull_survival(current_t, shape[k], scale[k]);
    }
    S_curve[t_idx] = s_val;
  }
  
  // 使用梯形法则计算 RMST (生存曲线下的面积)
  rmst = 0;
  for (t_idx in 1:L_grid) {
      real dt = tau_grid[t_idx + 1] - tau_grid[t_idx];
      rmst += 0.5 * dt * (S_curve[t_idx] + S_curve[t_idx+1]);
  }
}
