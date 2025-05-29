############### Hierarchical Normal Model
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(MCMCpack)

mod_HNM <- cmdstan_model("Hierarchical Normal Model.stan")
mod_HNM$print()

# prior and samples setting
mu0 <- 50
gamma0_2 <- 625
eta0 <- 1
tau0_2 <- 100 
sigma0_2 <-100
nu0 <- 1
N <- 500
J <- 10
# true hyperparameters
mu_true <- 20
sigma2_true <- 10
tau2_true <- 10
# generate samples
set.seed(123)
theta_true <- rnorm(J,mean = mu_true,sd=sqrt(tau2_true))
group <- rep(1:J,each=50)
y <- numeric(N)
for (i in 1:N) {
  j <- group[i]
  y[i] <- rnorm(1, mean = theta_true[j], sd = sqrt(sigma2_true))
}

data_HNM <- list(N=N,J=J,group=group,y=y,nu0=nu0,sigma0=sigma0_2,eta0=eta0,
                 tau0=tau0_2,mu0=mu0,gamma0=gamma0_2)

# model fitting
fit_HNM <- mod_HNM$sample(
  data = data_HNM,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)
fit_HNM$summary()
fit_HNM$summary(variables = c("mu","sigma","tau", "lp__"), "mean", "sd")
draws_HNM <- fit_HNM$draws(format = 'df')

mcmc_hist(fit_HNM$draws("mu")) +
  vline_at(mean(draws_HNM$mu), size = 1.5,color="red")+
  vline_at(mu_true, size = 1.5,color="green")+
  ggtitle(expression("Posterior of " * mu * " with MLE")) +
  xlab(expression(mu)) +
  ylab("Density")
  
mcmc_hist(fit_HNM$draws("tau")) +
  vline_at(mean(draws_HNM$tau), size = 1.5,color="red")+
  vline_at(tau2_true, size = 1.5,color="green")+
  ggtitle(expression("Posterior of " * tau^2 * " with MLE")) +
  xlab(expression(tau^2)) +
  ylab("Density")

mcmc_hist(fit_HNM$draws("sigma")) +
  vline_at(mean(draws_HNM$sigma), size = 1.5,color="red")+
  vline_at(sigma2_true, size = 1.5,color="green")+
  ggtitle(expression("Posterior of " * sigma^2 * " with MLE")) +
  xlab(expression(sigma^2)) +
  ylab("Density")

# model diagnostic
fit_HNM$diagnostic_summary()