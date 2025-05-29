####### Linear Regression
library(cmdstanr)
library(posterior)
library(ggplot2)
library(bayesplot)

# stan model input
mod_LR <- cmdstan_model("Linear Regression.stan")
mod_LR$print()

# prior and samples setting
d <- 1
N <- 100
set.seed(123)
X <- matrix(rnorm(N * d, mean = 0, sd = 1), nrow = N, ncol = d)
beta_true <- rnorm(d, mean = 0, sd = 1)
sigma_true <- 10
y <- X%*%beta_true+rnorm(N,mean=0,sd=sqrt(sigma_true))
beta0 <- rep(1, d)
Sigma0 <- 10*diag(1, d)
nu0 <- 2
sigma0 <- 1

data_LR <- list(
  N = N,
  d = d,
  X = X,
  y = as.vector(y),
  beta0 = beta0,
  Sigma0 = Sigma0,
  nu0 = nu0,
  sigma0 = sigma0
)

# model fitting
fit_LR <- mod_LR$sample(
  data = data_LR,chains = 4,
  seed = 123,
  parallel_chains = 4,
  refresh = 500
)
fit_LR$summary()
draws_LR <- fit_LR$draws(format = 'df')

mcmc_hist(fit_LR$draws("sigma")) +
  vline_at(mean(draws_LR$sigma), size = 1.5,color="red")+
  vline_at(sigma_true, size = 1.5,color="red")+
  ggtitle(expression("Posterior of " * sigma^2 * " with MLE")) +
  xlab(expression(sigma^2)) +
  ylab("Density")

# model diagnostic
fit_LR$diagnostic_summary()

library(tidyverse)

# 获取 posterior 样本均值作为估计的 beta
beta_est <- colMeans(draws_LR %>% select(starts_with("beta")))

# 手动计算预测值
y_hat <- X %*% beta_est

# 可视化：使用 X 的第一个维度与 y 的散点图 + 拟合曲线
df_plot <- data.frame(
  x = X[, 1],
  y = y,
  y_hat = y_hat
)

df_plot_sorted <- df_plot %>% arrange(x)

ggplot(df_plot_sorted, aes(x = x, y = y)) +
  geom_point(alpha = 0.6, color = "black") +
  geom_line(aes(y = y_hat), color = "blue", size = 1.2) +
  labs(
    title = "Scatter Plot of y vs X with Fitted Regression Line",
    x = "X[,1]",
    y = "y / y_hat"
  ) +
  theme_minimal()