############ MVN model
library(mvtnorm)
library(cmdstanr)
library(bayesplot)
library(posterior)
library(ggplot2)

mod_mvn <- cmdstan_model("MVN.stan")
mod_mvn$print()

# generate mvn samples
set.seed(123)
d <- 2
N <- 100
mu_true <- rnorm(d,mean=0,sd=4)
Sigma_true <- matrix(rnorm(d^2),d,d)
Sigma_true <- crossprod(Sigma_true)
Y <- rmvnorm(n = 100, mean = mu_true, sigma = Sigma_true)

# Data list
data_mvn <- list(N = N, d = d, Y = Y)
fit_mvn <- mod_mvn$sample(
  data = data_mvn,
  seed = 123,
  chains = 4,
  parallel_chain  = 4,
  refresh = 500
)

# model summary
fit_mvn$summary()
draws_mvn <- fit_mvn$draws(format = "df")
fit_mvn$summary("mu")$rhat
fit_mvn$summary("Sigma")$rhat
fit_mvn$summary(c("mu", "Sigma"))$ess_bulk

# model diagnostic
fit_mvn$diagnostic_summary()

mu_post <- c(mean(draws_mvn$`mu[1]`),mean(draws_mvn$`mu[2]`))
Sigma_post <- matrix(data = c(mean(draws_mvn$`Sigma[1,1]`),
                              mean(draws_mvn$`Sigma[1,2]`),
                              mean(draws_mvn$`Sigma[2,1]`),
                              mean(draws_mvn$`Sigma[2,2]`)),nrow = 2,ncol = 2, byrow = TRUE)

# generate true samples and posterior samples
set.seed(123)
truth_samples <- rmvnorm(n = 1000, mean = mu_true, sigma = Sigma_true)
post_samples <- rmvnorm(n = 1000, mean = mu_post, sigma = Sigma_post)

# transfer to data.frame
truth_df <- as.data.frame(truth_samples)
post_df <- as.data.frame(post_samples)
colnames(truth_df) <- c("x", "y")
colnames(post_df) <- c("x", "y")

# comparison plot
ggplot() +
  stat_density_2d(data = truth_df, aes(x = x, y = y, fill = ..level..), geom = "polygon", alpha = 0.4, color = NA) +
  stat_density_2d(data = post_df, aes(x = x, y = y, color = ..level..), geom = "density2d") +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  scale_color_gradient(low = "pink", high = "red") +
  labs(title = "True (fill) vs Posterior (contour) Density",
       x = expression(X[1]),
       y = expression(X[2])) +
  theme_minimal()