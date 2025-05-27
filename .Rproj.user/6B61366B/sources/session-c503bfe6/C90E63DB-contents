library(cmdstanr)
library(posterior)
library(bayesplot)
library(mvtnorm)
library(MASS)       # for kde2d
library(ggplot2)
file <- file.path(cmdstan_path(), "examples", "bernoulli", "bernoulli.stan")
mod <- cmdstan_model(file)

mod$print() # model的结构

mod$exe_file() # model执行文件路径

# names correspond to the data block in the Stan program
data_list <- list(N = 10, y = c(0,1,0,0,0,0,0,0,0,1))

fit <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)

fit$summary()
fit$summary(variables = c("theta", "lp__"), "mean", "sd")

# use a formula to summarize arbitrary functions, e.g. Pr(theta <= 0.5)
fit$summary("theta", pr_lt_half = ~ mean(. <= 0.5))

# summarise all variables with default and additional summary measures
fit$summary(
  variables = NULL,
  posterior::default_summary_measures(),
  extra_quantiles = ~posterior::quantile2(., probs = c(.0275, .975))
)

# default is a 3-D draws_array object from the posterior package
# iterations x chains x variables
draws_arr <- fit$draws() # or format="array"
str(draws_arr)

# draws x variables data frame
draws_df <- fit$draws(format = "df")
str(draws_df)

mcmc_hist(fit$draws("theta"))

# sample diagnostics
# this is a draws_array object from the posterior package
str(fit$sampler_diagnostics())
# this is a draws_df object from the posterior package
str(fit$sampler_diagnostics(format = "df"))

fit$diagnostic_summary()

fit_mle <- mod$optimize(data = data_list, seed = 123)
fit_mle$print() # includes lp__ (log prob calculated by Stan program)

mcmc_hist(fit$draws("theta")) +
  vline_at(fit_mle$mle("theta"), size = 1.5)

fit_map <- mod$optimize(
  data = data_list,
  jacobian = TRUE,
  seed = 123
)

# Normal Model
mod <- cmdstan_model("Normal Model.stan")
mod$print() # model的结构

data_list <- list(N = 100, y = rnorm(100,mean=10,sd=5))
fit_normal <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 10,
  parallel_chains = 5,
  refresh = 500 # print update every 500 iters
)

fit_normal$summary()

# Exponential Model
mod_exp <- cmdstan_model("Exponential Model.stan")
mod_exp$print()

data_exp <- list(N=200,y=rexp(200,rate=5))
fit_exp <- mod_exp$sample(
  data = data_exp,
  seed = 66,
  chains = 4,
  parallel_chains = 4,
  refresh = 500
)
fit_exp$summary()
mcmc_hist(fit_exp$draws("lambda"))
str(fit_exp$sampler_diagnostics())
draws_df <- fit_exp$draws(format = "df")
str(draws_df)
fit_exp$diagnostic_summary()

# Multivariate Normal Model
mod_mvn <- cmdstan_model("MVN.stan")
mod_mvn$print()

set.seed(66)
d <- 2
mu_true <- rnorm(d,mean=0,sd=4)
Sigma_true <- matrix(rnorm(d^2),d,d)
Sigma_true <- crossprod(Sigma_true)
data_list <- list(N=100,d = 2,Y=rmvnorm(n = 100, mean = mu_true, sigma = Sigma_true))
fit_mvn <- mod_mvn$sample(
  data = data_list,
  seed = 66,
  chains = 4,
  parallel_chains = 4,
  refresh = 500
)
fit_mvn$summary()
fit_exp$diagnostic_summary()
# 获取 mu 的后验样本
draws <- fit_mvn$draws(format = "df")
mu_post <- c(mean(draws$`mu[1]`),mean(draws$`mu[2]`))
Sigma_post <- matrix(data = c(mean(draws$`Sigma[1,1]`),
                              mean(draws$`Sigma[1,2]`),
                              mean(draws$`Sigma[2,1]`),
                              mean(draws$`Sigma[2,2]`)),nrow = 2,ncol = 2, byrow = TRUE)

# 生成真实分布和后验分布样本
truth_samples <- rmvnorm(n = 1000, mean = mu_true, sigma = Sigma_true)
post_samples <- rmvnorm(n = 1000, mean = mu_post, sigma = Sigma_post)

# 转为数据框
truth_df <- as.data.frame(truth_samples)
post_df <- as.data.frame(post_samples)
colnames(truth_df) <- c("x", "y")
colnames(post_df) <- c("x", "y")

# 绘图
ggplot() +
  stat_density_2d(data = truth_df, aes(x = x, y = y, fill = ..level..), geom = "polygon", alpha = 0.4, color = NA) +
  stat_density_2d(data = post_df, aes(x = x, y = y, color = ..level..), geom = "density2d") +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  scale_color_gradient(low = "pink", high = "red") +
  labs(title = "True (fill) vs Posterior (contour) Density",
       x = expression(X[1]),
       y = expression(X[2])) +
  theme_minimal()