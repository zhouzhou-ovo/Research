library(frailtypack)
library(jsonlite)
library(survival)
library(dplyr)
library(tidyr)
library(broom)
library(ggplot2)
library(tidyverse)
library(survival)
library(posterior)
library(patchwork)
library(cmdstanr)

# load data
data(bcos)
head(bcos)
bcos$left <- as.numeric(bcos$left)
bcos$right <- as.numeric(bcos$right)
bcos$treatment <- recode(bcos$treatment,
                         "RadChem" = "RCT",
                         "Rad" = "RT")
group_id <- as.integer(bcos$treatment)

###### the frequency NPMLE
# define Surv object; "interval2" used to estimate interval-censored data
bcos$SurvObj <- with(bcos, Surv(time = left, time2 = right, type = "interval2"))

# fit Kaplan-Meier Curve (group by treatment)
fit <- survfit(SurvObj ~ treatment, data = bcos)
fit_df <- tidy(fit)

# plot
plot(fit, xlab = "Time", ylab = "Survival probability", col = c("tomato", "green"),
     lty = 1, main = "Interval-Censored Survival Curve by Treatment (bcos)")
legend("bottomleft", legend = levels(bcos$treatment),
       col = c("tomato", "green"), lty = 1)

ggplot(fit_df, aes(x = time, y = estimate, color = strata)) +
  geom_step() +
  labs(
    title = "Interval-Censored Survival Curve by Treatment (bcos)",
    x = "Time",
    y = "Survival Probability",
    color = "Treatment",
    fill = "Treatment"
  ) +
  theme_minimal()

# add initial point t=0
fit_df_fixed <- fit_df %>%
  group_by(strata) %>%
  arrange(time) %>%
  mutate(time = ifelse(row_number() == 1 & time > 0, 0, time),
         estimate = ifelse(row_number() == 1 & time > 0, 1, estimate)) %>%
  ungroup()

# calculate RMST（Trapezoidal Rule）
tau <- max(fit_df$time, na.rm = TRUE)
rmst_result <- fit_df_fixed %>%
  group_by(strata) %>%
  filter(time <= 46) %>%
  arrange(time) %>%
  summarise(
    RMST = sum(diff(time) * head(estimate, -1), na.rm = TRUE)
  )

print(rmst_result)

# function to split interval and exact time
split_interval_exact <- function(L, R, tol = 1e-8) {
  stopifnot(length(L) == length(R))
  
  # determine exact time (L==R)
  is_exact <- abs(L - R) < tol
  is_interval <- L < R - tol
  
  # get corresponding subsets
  t_exact <- L[is_exact]
  L_interval <- L[is_interval]
  R_interval <- R[is_interval]
  
  # number of samples
  n_exact <- length(t_exact)
  n_interval <- length(L_interval)
  
  # return list results
  list(
    n_exact = n_exact,
    n_interval = n_interval,
    t_exact = t_exact,
    L_interval = L_interval,
    R_interval = R_interval,
    is_exact = is_exact,
    is_interval = is_interval
  )
}

# split exact time and interval-censor data in 2 treatments 
bcos_RT <- split_interval_exact(bcos$left[bcos$treatment=="RT"],bcos$right[bcos$treatment=="RT"])
bcos_RCT <- split_interval_exact(bcos$left[bcos$treatment=="RCT"],bcos$right[bcos$treatment=="RCT"])

# load stan model
MDP_weibull <- cmdstan_model("MDP-weibull.stan")

# stan data
RT_weibull <- list(
  N_interval = bcos_RT$n_interval,
  N_exact = bcos_RT$n_exact,
  N_grid = 100,
  t_exact = bcos_RT$t_exact,
  L = bcos_RT$L_interval,
  R = bcos_RT$R_interval,
  K = 20,
  tau = 46,
  L_grid = 50,
  M = 1
)

RCT_weibull <- list(
  N_interval = bcos_RCT$n_interval,
  N_exact = bcos_RCT$n_exact,
  N_grid = 100,
  t_exact = bcos_RCT$t_exact,
  L = bcos_RCT$L_interval,
  R = bcos_RCT$R_interval,
  K = 20,
  tau = 46,
  L_grid = 20,
  M = 1
)

# model fit
fit_RT_weibull <- MDP_weibull$sample(
  data = RT_weibull,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 1000,
  iter_warmup = 500,
  refresh = 200,
  save_warmup = FALSE
)

fit_RCT_weibull <- MDP_weibull$sample(
  data = RCT_weibull,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 1000,
  iter_warmup = 500,
  refresh = 200,
  save_warmup = FALSE
)

# get posterior results
rmst_RT_weibull <- fit_RT_weibull$draws("rmst_from_F") %>%
  as_draws_df() %>%
  drop_na(rmst_from_F)

rmst_RCT_weibull <- fit_RCT_weibull$draws("rmst_from_F") %>%
  as_draws_df() %>%
  drop_na(rmst_from_F)

ggplot(rmst_RT_weibull[1:1000,], aes(x = rmst_from_F)) +
  geom_histogram(color = "black", fill = "skyblue", bins = 50) +
  labs(
    title = "Posterior Distribution of RMST (RT Group)",
    x = "RMST",
    y = "Frequency"
  ) +
  theme_minimal()

ggplot(rmst_RCT_weibull[1:1000,], aes(x = rmst_from_F)) +
  geom_histogram(color = "black", fill = "skyblue", bins = 50) +
  labs(
    title = "Posterior Distribution of RMST (RT Group)",
    x = "RMST",
    y = "Frequency"
  ) +
  theme_minimal()

# survival curve
draws_surv_RCT_weibull <- fit_RCT_weibull$draws(c("time_grid", "F_curve"))
draws_df_RCT_weibull <- drop_na(as_draws_df(draws_surv_RCT_weibull))
draws_surv_RT_weibull <- fit_RT_weibull$draws(c("time_grid", "F_curve"))
draws_df_RT_weibull <- drop_na(as_draws_df(draws_surv_RT_weibull))

surv_df_RCT_weibull <- tibble(
  time = colMeans(dplyr::select(draws_df_RCT_weibull, starts_with("time_grid"))),
  surv = colMeans(dplyr::select(draws_df_RCT_weibull, starts_with("F_curve")))
)
surv_df_RT_weibull <- tibble(
  time = colMeans(dplyr::select(draws_df_RT_weibull, starts_with("time_grid"))),
  surv = colMeans(dplyr::select(draws_df_RT_weibull, starts_with("F_curve")))
)

# choose RCT, draw coparison plot of KM curve and posterior survival curve
fit_df_RCT_weibull <- fit_df %>% filter(strata == "treatment=RCT")
fit_df_RT_weibull <- fit_df %>% filter(strata == "treatment=RT")

km_plot_df_RCT_weibull <- fit_df_RCT_weibull %>%
  dplyr::select(time, estimate) %>%
  rename(surv = estimate) %>%
  mutate(source = "KM")

km_plot_df_RT_weibull <- fit_df_RT_weibull %>%
  dplyr::select(time, estimate) %>%
  rename(surv = estimate) %>%
  mutate(source = "KM")

posterior_plot_df_RCT_weibull <- surv_df_RCT_weibull %>%
  mutate(source = "Posterior")

posterior_plot_df_RT_weibull <- surv_df_RT_weibull %>%
  mutate(source = "Posterior")

compare_df_RCT_weibull <- bind_rows(km_plot_df_RCT_weibull, posterior_plot_df_RCT_weibull)
compare_df_RT_weibull <- bind_rows(km_plot_df_RT_weibull, posterior_plot_df_RT_weibull)

ggplot(compare_df_RCT_weibull, aes(x = time, y = surv, color = source)) +
  geom_step(data = subset(compare_df_RCT_weibull, source == "KM"), size = 1.2) +
  geom_line(data = subset(compare_df_RCT_weibull, source == "Posterior"), size = 1.2, linetype = "dashed") +
  labs(
    title = "KM vs Posterior Survival Curve (RCT Group)",
    x = "Time",
    y = "Survival Probability",
    color = "Curve Source"
  ) +
  theme_minimal()

ggplot(compare_df_RT_weibull, aes(x = time, y = surv, color = source)) +
  geom_step(data = subset(compare_df_RT_weibull, source == "KM"), size = 1.2) +
  geom_line(data = subset(compare_df_RT_weibull, source == "Posterior"), size = 1.2, linetype = "dashed") +
  labs(
    title = "KM vs Posterior Survival Curve (RT Group)",
    x = "Time",
    y = "Survival Probability",
    color = "Curve Source"
  ) +
  theme_minimal()