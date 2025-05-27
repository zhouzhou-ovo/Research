######## one parameter model
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)

### Binomial model
mod_binom <- cmdstan_model("Binomial Model.stan")
print(mod_binom)

# Generate samples
N <- 129
set.seed(123)  # 可选：为了结果可复现
y_biom <- sample(c(rep(1, 118), rep(0, 11)))
data_binom <- list(N=N,y=y_binom)

# model fitting
fit_binom <- mod_binom$sample(
  data = data_binom,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)

# model summary
fit_binom$summary()
fit_binom$summary(variables = c("theta", "lp__"), "mean", "sd")

# store the posterior samples
draws_df_binom <- fit_binom$draws(format = "df")
str(draws_df_binom)

# posterior density plot with posterior mean
mcmc_hist(fit_binom$draws("theta")) +
  vline_at(mean(draws_df_binom$theta), size = 1.5,color="red")+
  ggtitle(expression("Posterior of " * theta * " with MLE")) +
  xlab(expression(theta)) +
  ylab("Density")

fit_binom$diagnostic_summary()

### Poisson Model
mod_poi <- cmdstan_model("Poisson model.stan")
print(mod_poi)

# generate samples
n <- 111
sum <- 217
set.seed(123)  # for reproducibility
generate_sample <- function(n, sum_target) {
  x <- rmultinom(1, size = sum_target, prob = rep(1, n))
  as.vector(x)
}
y_poi <- generate_sample(n, sum)
data_poi <- list(N=n,y=y_poi)

# model fitting
fit_poi <- mod_poi$sample(
  data=data_poi,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)

# model summary
fit_poi$summary()
fit_poi$summary(variables = c("lambda", "lp__"), "mean", "sd")

# store the posterior samples
draws_df_poi <- fit_poi$draws(format = "df")
str(draws_df_poi)

# posterior density plot with posterior mean
mcmc_hist(fit_poi$draws("lambda")) +
  vline_at(mean(draws_df_poi$lambda), size = 1.5,color="red")+
  ggtitle(expression("Posterior of " * lambda * " with MLE")) +
  xlab(expression(lambda)) +
  ylab("Density")

fit_poi$diagnostic_summary()