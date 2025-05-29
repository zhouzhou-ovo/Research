######### Normal Model
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
# semi-conjugate prior
mod_normal <- cmdstan_model("Normal Model.stan")
print(mod_normal)

# generate normal samples
N <- 100
mu <- 10
sigma <- 5
data_normal <- list(N = N, y = rnorm(N,mean=mu,sd=sigma))

# model fitting
fit_normal <- mod_normal$sample(
  data = data_normal,
  seed = 123,
  chains = 10,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)

# model summary
fit_normal$summary()
fit_normal$summary(variables = c("mu","sigma", "lp__"), "mean", "sd")

# store the posterior samples
draws_df_normal <- fit_normal$draws(format = "df")
str(draws_df_normal)

# posterior density plot with posterior mean
mcmc_hist(fit_normal$draws("mu")) +
  vline_at(mean(draws_df_normal$mu), size = 1.5,color="red")+
  ggtitle(expression("Posterior of " * mu * " with MLE(normal)")) +
  xlab(expression(mu)) +
  ylab("Density")

mcmc_hist(fit_normal$draws("sigma")^2) +
  vline_at(mean(draws_df_normal$sigma^2), size = 1.5,color="red")+
  ggtitle(expression("Posterior of " * sigma^2 * " with MLE(inverse-gamma)")) +
  xlab(expression(sigma^2)) +
  ylab("Density")

# diagnostic
fit_normal$diagnostic_summary()