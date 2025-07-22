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
library(bayesplot)

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
  geom_step(linewidth=1.2) +
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

# ggplot(fit_df_fixed, aes(x = time, y = estimate, color = strata)) +
#   geom_step(linewidth=1.2) +
#   labs(
#     title = "Interval-Censored Survival Curve by Treatment (bcos)",
#     x = "Time",
#     y = "Survival Probability",
#     color = "Treatment",
#     fill = "Treatment"
#   ) +
#   theme_minimal()

# calculate RMST（Trapezoidal Rule）
tau <- max(fit_df$time, na.rm = TRUE)
rmst_result <- fit_df_fixed %>%
  group_by(strata) %>%
  filter(time <= tau) %>%
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
DPM_ln <- cmdstan_model("DPM-lognormal.stan")

# stan data
RT_ln <- list(
  N_interval = bcos_RT$n_interval,
  N_exact = bcos_RT$n_exact,
  N_grid = 100,
  t_exact = bcos_RT$t_exact,
  L = bcos_RT$L_interval,
  R = bcos_RT$R_interval,
  K = 30,
  mu0 = log(32),
  lambda0 = 1,
  c0 = 1,
  d0 = 1,
  tau = 46
)

RCT_ln <- list(
  N_interval = bcos_RCT$n_interval,
  N_exact = bcos_RCT$n_exact,
  N_grid = 100,
  t_exact = bcos_RCT$t_exact,
  L = bcos_RCT$L_interval,
  R = bcos_RCT$R_interval,
  K = 20,
  mu0 = log(18),
  lambda0 = 1,
  c0 = 1,
  d0 = 1,
  tau = 46
)

# inference
fit_RT_ln <- DPM_ln$sample(
  data = RT_ln,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 2000,
  iter_warmup = 500,
  refresh = 500,
  save_warmup = FALSE
)

fit_RCT_ln <- DPM_ln$sample(
  data = RCT_ln,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 2000,
  iter_warmup = 500,
  refresh = 500,
  save_warmup = FALSE
)

# 提取后验
rmst_RT_ln <- fit_RT_ln$draws("rmst_pred") %>%
  as_draws_df() %>%
  drop_na(rmst_pred)

ggplot(rmst_RT_ln[1:1000,], aes(x = rmst_pred)) +
  geom_histogram(color = "black", fill = "skyblue", bins = 50) +
  labs(
    title = "Posterior Distribution of RMST (RT Group)",
    x = "RMST",
    y = "Frequency"
  ) +
  theme_minimal()

rmst_RCT_ln <- fit_RCT_ln$draws("rmst_pred") %>%
  as_draws_df() %>%
  drop_na(rmst_pred)

ggplot(rmst_RCT_ln, aes(x = rmst_pred)) +
  geom_histogram(color = "black", fill = "skyblue", bins = 50) +
  labs(
    title = "Posterior Distribution of RMST (RCT Group)",
    x = "RMST",
    y = "Frequency"
  ) +
  theme_minimal()

# survival curve
draws_surv_RCT_ln <- fit_RCT_ln$draws(c("time_grid", "surv_mix_ln"))
draws_df_RCT_ln <- drop_na(as_draws_df(draws_surv_RCT_ln))
draws_surv_RT_ln <- fit_RT_ln$draws(c("time_grid", "surv_mix_ln"))
draws_df_RT_ln <- drop_na(as_draws_df(draws_surv_RT_ln))

surv_df_RCT_ln <- tibble(
  time = colMeans(dplyr::select(draws_df_RCT_ln, starts_with("time_grid"))),
  surv = colMeans(dplyr::select(draws_df_RCT_ln, starts_with("surv_mix_ln")))
)
surv_df_RT_ln <- tibble(
  time = colMeans(dplyr::select(draws_df_RT_ln, starts_with("time_grid"))),
  surv = colMeans(dplyr::select(draws_df_RT_ln, starts_with("surv_mix_ln")))
)

# choose RCT, draw coparison plot of KM curve and posterior survival curve
fit_df_RCT_ln <- fit_df %>% filter(strata == "treatment=RCT")
fit_df_RT_ln <- fit_df %>% filter(strata == "treatment=RT")

km_plot_df_RCT_ln <- fit_df_RCT_ln %>%
  dplyr::select(time, estimate) %>%
  rename(surv = estimate) %>%
  mutate(source = "KM")

km_plot_df_RT_ln <- fit_df_RT_ln %>%
  dplyr::select(time, estimate) %>%
  rename(surv = estimate) %>%
  mutate(source = "KM")

posterior_plot_df_RCT_ln <- surv_df_RCT_ln %>%
  mutate(source = "Posterior")

posterior_plot_df_RT_ln <- surv_df_RT_ln %>%
  mutate(source = "Posterior")

compare_df_RCT_ln <- bind_rows(km_plot_df_RCT_ln, posterior_plot_df_RCT_ln)
compare_df_RT_ln <- bind_rows(km_plot_df_RT_ln, posterior_plot_df_RT_ln)

ggplot(compare_df_RCT_ln, aes(x = time, y = surv, color = source)) +
  geom_step(data = subset(compare_df_RCT_ln, source == "KM"), size = 1.2) +
  geom_line(data = subset(compare_df_RCT_ln, source == "Posterior"), size = 1.2, linetype = "dashed") +
  labs(
    title = "KM vs Posterior Survival Curve (RCT Group)",
    x = "Time",
    y = "Survival Probability",
    color = "Curve Source"
  ) +
  theme_minimal()

ggplot(compare_df_RT_ln, aes(x = time, y = surv, color = source)) +
  geom_step(data = subset(compare_df_RT_ln, source == "KM"), size = 1.2) +
  geom_line(data = subset(compare_df_RT_ln, source == "Posterior"), size = 1.2, linetype = "dashed") +
  labs(
    title = "KM vs Posterior Survival Curve (RT Group)",
    x = "Time",
    y = "Survival Probability",
    color = "Curve Source"
  ) +
  theme_minimal()