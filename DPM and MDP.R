##### DPM and MDP model
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(tidyr)

# true parameters of  mixture gaussian distribution
set.seed(66)
N <- 300
true_K <- 3
true_means <- c(-3, 0, 4)
true_sds <- c(0.5, 0.8, 0.6)
true_weights <- c(0.3, 0.5, 0.2)

# sample component assignments
z <- sample(1:true_K, size = N, replace = TRUE, prob = true_weights)
y <- rnorm(N, mean = true_means[z], sd = true_sds[z])
hist(y, breaks = 30, col = "gray", main = "Simulated Data from True GMM")

ggplot(data.frame(y = y), aes(x = y)) +
  geom_histogram(bins = 30, fill = "gray70", color = "white") +
  labs(
    title = "Histogram of Simulated GMM Data",
    x = "y",
    y = "Count"
  ) +
  theme_minimal()

df <- data.frame(
  y = y,
  cluster = as.factor(z),
  jitter_y = jitter(rep(0, N), amount = 0.04)
)

ggplot(df, aes(x = y, y = jitter_y, color = cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(
    title = "True Cluster Assignment (Scatter Plot)",
    x = "y",
    y = "Artificial Jitter (for display)",
    color = "Cluster"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

############# DPM prior ################
# load the DPM-GMM stan model
mod_DPM_GMM <- cmdstan_model("DPM-GMM.stan")
mod_DPM_GMM$print()

# prior parameters setting
L <- 30
mu0 <- mean(y)
tau0 <- 10
alpha <- 5
sigma0 <- var(y)
nu0 <- 5

data_DPM_GMM <- list(N=N,Y=y,L=L,
                     mu0=mu0,tau0=tau0,sigma0=sigma0,
                     nu0=nu0,alpha=alpha)

# model fitting
fit_DPM_GMM <- mod_DPM_GMM$sample(
  data=data_DPM_GMM,
  chains = 4,
  seed = 66,
  iter_warmup = 1000,
  iter_sampling = 2000,
  parallel_chains = 4,
  refresh = 500
)
summary_DPM_GMM <- fit_DPM_GMM$summary()

fit_DPM_GMM$diagnostic_summary()

y_pre_draws <- fit_DPM_GMM$draws("y_pre")
y_pre_mean <- colMeans(as_draws_matrix(y_pre_draws))
df <- data.frame(
  Y = c(y, y_pre_mean),
  type = rep(c("True Data", "Posterior Predictive Mean"), c(length(y), length(y_pre_mean)))
)
ggplot(df, aes(x = Y, fill = type, color = type)) +
  geom_density(alpha = 0.4) +
  labs(
    title = "Posterior Predictive (Mean) vs True Data Density",
    x = "y",
    y = "Density"
  ) +
  theme_minimal()

# Extract the posterior predictive draws into a matrix format
y_pre_DPM_matrix <- as_draws_matrix(fit_DPM_GMM$draws("y_pre"))
# Each row is a draw from the posterior, each column corresponds to a data point y_i

# Sample about 100-200 draws to plot
set.seed(123)
plot_indices <- sample(1:nrow(y_pre_DPM_matrix), size = 50)
y_pre_DPM_subset <- y_pre_DPM_matrix[plot_indices, ]

# Combine into a long-format data frame for ggplot
df_pred_DPM <- data.frame(
  y_true = y,
  y_pred_reps = t(y_pre_DPM_subset) # Transpose to make it easier to work with
)

# Reshape to long format
df_long_DPM <- pivot_longer(df_pred, 
                        cols = starts_with("y_pred_reps"), 
                        names_to = "rep", 
                        values_to = "y_pred_value")

# Plot the densities
ggplot() +
  # Plot each of the 100 posterior predictive replications as a faint line
  geom_density(data = df_long_DPM, aes(x = y_pred_value, group = rep),
               color = "skyblue", alpha = 0.2) +
  # Overlay the density of the true observed data
  geom_density(data = data.frame(y = y), aes(x = y), 
               color = "black", linewidth = 1.2) +
  labs(
    title = "Posterior Predictive Checks",
    subtitle = "Black: Observed Data | Blue: 50 Posterior Predictive Replications",
    x = "y",
    y = "Density"
  ) +
  theme_minimal()


################ MDP prior ##################
group_idx <- rep(1,N)

# Visualize the data with its artificial grouping
sim_data_df <- data.frame(y = y, group = paste("Group", group_idx))
ggplot(sim_data_df, aes(x = y, fill = group)) +
  geom_histogram(bins = 30, color = "white", alpha = 0.8) +
  facet_wrap(~group, ncol = 1) +
  labs(title = "Data with Artificial Groupings", x = "y", y = "Count") +
  theme_minimal()

#---------------------------------------------------------------------
# 3. FIT THE HDP-GMM (MDP) MODEL
#---------------------------------------------------------------------
mod_HDP_GMM <- cmdstan_model("MDP-GMM.stan")

# Prior parameters setting
K_max <- 20
J <- 1
mu0 <- mean(y)
tau0 <- 5
alpha <- 1.0
gamma <- 1.0
sigma0 <- var(y)
nu0 <- 1

# Create the data list for the HDP model, including group info
data_HDP_GMM <- list(
  N = N, J = J, K = K_max,
  y = y, group_idx = group_idx,
  mu0 = mu0, tau0 = tau0,
  sigma0 = sigma0, nu0 = nu0,
  alpha = alpha, gamma = gamma
)

# Model fitting
fit_HDP_GMM <- mod_HDP_GMM$sample(
  data = data_HDP_GMM,
  chains = 4,
  seed = 66,
  iter_warmup = 500,
  iter_sampling = 1000,
  parallel_chains = 4,
  refresh = 500,
  adapt_delta = 0.9
)

#---------------------------------------------------------------------
# 4. VISUALIZE RESULTS (FACETED BY ARTIFICIAL GROUP)
#---------------------------------------------------------------------
y_rep_matrix <- as_draws_matrix(fit_HDP_GMM$draws("y_rep"))

set.seed(123)
plot_indices <- sample(1:nrow(y_rep_matrix), size = 50)
y_rep_subset <- y_rep_matrix[plot_indices, ]

# Create a long-format data frame for ggplot
df_pred_long <- as.data.frame(t(y_rep_subset)) %>%
  mutate(obs_id = 1:N, y_true = y, group = paste("Group", group_idx)) %>%
  pivot_longer(
    # Select all columns EXCEPT the ones we just added by name
    cols = -c(obs_id, y_true, group), 
    names_to = "rep", 
    values_to = "y_rep_value"
  )

# Plot posterior predictive checks, faceted by group
ggplot(df_pred_long, aes(x = y_rep_value)) +
  geom_density(aes(group = rep), color = "skyblue", alpha = 0.2) +
  geom_density(data = . %>% distinct(obs_id, .keep_all = TRUE), aes(x = y_true), color = "black", linewidth = 1.2) +
  facet_wrap(~group, ncol = 1) +
  labs(
    title = "Posterior Predictive Checks (MDP on Artificial Groups)",
    subtitle = "Black: Observed Data | Blue: 50 Posterior Predictive Replications",
    x = "y", y = "Density"
  ) +
  theme_minimal()

# Check the inferred group proportions 'pi'
# **EXPECTATION**: Since the groups are artificial, the inferred proportions pi[1,],
# pi[2,], and pi[3,] should all be very similar to each other.
fit_HDP_GMM$summary("pi")