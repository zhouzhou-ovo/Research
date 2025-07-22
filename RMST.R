library(frailtypack)
library(jsonlite)
library(survival)
library(dplyr)
library(broom)
library(ggplot2)
library(tidyverse)
library(survival)
library(posterior)
library(patchwork)

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

tau <- max(fit_df$time, na.rm = TRUE)
rmst_result <- fit_df %>%
  filter(time <= tau) %>%
  group_by(strata) %>%
  arrange(time) %>%
  summarise(
    RMST = sum(diff(c(0, time)) * estimate[-1], na.rm = TRUE)
  )

print(rmst_result)


# stan_data <- list(
#   N = nrow(bcos),
#   J = 2,
#   M = 1e-6,
#   L = bcos$left,
#   R = bcos$right,
#   group_idx = group_id,
#   left = bcos$left,
#   right = bcos$right,
#   K = 20,
#   a_shape = 0.01, b_shape = 0.01,
#   a_scale = 0.01, b_scale = 0.01,
#   alpha = 1
# )
# 
# # 编译并拟合模型
# mod <- cmdstan_model("MDP-RMST-Weibull.stan")
# fit <- mod$sample(
#   data = stan_data,
#   chains = 4,
#   parallel_chains = 4,
#   iter_warmup = 500,
#   iter_sampling = 1000,
#   seed = 123
# )
# 
# # 后处理：估计生存函数
# draws <- as_draws_df(fit$draws())
# time_grid <- seq(0, max(bcos$right, na.rm = TRUE), length.out = 100)
# surv_estimates <- list()
# 
# for (g in 1:stan_data$G) {
#   surv_matrix <- matrix(0, nrow = nrow(draws), ncol = length(time_grid))
#   for (t in seq_along(time_grid)) {
#     for (s in 1:nrow(draws)) {
#       S_t <- 0
#       for (k in 1:stan_data$K) {
#         shape_k <- draws[[paste0("shape[", k, "]")]][s]
#         scale_k <- draws[[paste0("scale[", k, "]")]][s]
#         pi_k <- draws[[paste0("pi[", g, ",", k, "]")]][s]
#         S_t <- S_t + pi_k * (1 - pweibull(time_grid[t], shape_k, scale_k))
#       }
#       surv_matrix[s, t] <- S_t
#     }
#   }
#   # 每个 group 存储中位数估计和 95% CI
#   surv_estimates[[g]] <- data.frame(
#     time = time_grid,
#     median = apply(surv_matrix, 2, median),
#     lower = apply(surv_matrix, 2, quantile, 0.025),
#     upper = apply(surv_matrix, 2, quantile, 0.975),
#     group = levels(bcos$treatment)[g]
#   )
# }
# 
# # 合并所有 group
# df_surv <- bind_rows(surv_estimates)
# 
# # 可视化
# ggplot(df_surv, aes(x = time, y = median, color = group, fill = group)) +
#   geom_line() +
#   geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
#   labs(title = "Survival Curves by Treatment (MDP-Weibull)",
#        y = "Survival Probability", x = "Time") +
#   theme_minimal()

# 时间网格
T_eval <- 100
eval_times <- seq(0, max(bcos$right, na.rm = TRUE), length.out = T_eval)

# Stan data
zero_events_idx <- which(bcos$left == 0)

if (length(zero_events_idx) > 0) {
  print(paste("发现了", length(zero_events_idx), "个时间点为0的精确事件。"))
  print("为了数值稳定性，将它们替换为一个极小的正数 (1e-9)。")
  
  # 将 L 和 R 都替换为这个小数字
  bcos$left[zero_events_idx] <- 1e-9
} else {
  print("数据检查通过：不存在时间点为0的精确事件。")
}

stan_data <- list(
  N = nrow(bcos)-46,
  L = bcos$left[47:nrow(bcos)],
  R = bcos$right[47:nrow(bcos)],
  a_shape = 0.01, b_shape = 0.01,
  a_scale = 0.01, b_scale = 0.01,
  M = 1e-6,
  tau = 46,
  L_grid = 50
)

# 编译和拟合模型
mod <- cmdstan_model("rmst-mdp1.stan")
fit <- mod$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 2000,
  seed = 66,
  refresh = 500
)

rmst_samples <- extract(fit)$rmst
summary(rmst_samples)
hist(rmst_samples, main="Posterior Distribution of RMST", xlab="RMST")

# 提取并绘制后验平均生存曲线
S_curve_samples <- extract(fit)$S_curve
mean_S_curve <- colMeans(S_curve_samples)
tau_grid <- stan_data$tau * (0:stan_data$L_grid) / stan_data$L_grid

plot(tau_grid, mean_S_curve, type='l', lwd=2, col="blue",
     xlab="Time", ylab="Survival Probability", main="Posterior Mean Survival Curve")

draws <- as_draws_df(fit$draws())  # 方便数据操作
# 假如你的generated quantities中叫 S_curve，你需要对其进行重塑:

# 假如 S_curve[1]是group 1, S_curve[2]是group 2
# 假如 L_grid+1 == T
# 每条抽样每时刻都有一条样本:

J <- stan_data$J
L_grid <- stan_data$L_grid
tau_grid <- seq(0, stan_data$tau, length.out = L_grid + 1)

# 抽取出 S_curve:
S_matrix <- array(NA, dim = c(
  nrow(draws),
  J,
  L_grid + 1
))

for (j in 1:J) {
  for (t in 1:(L_grid + 1)) {
    S_matrix[, j, t] <- draws[[paste0("S_curve[", j, ",", t, "]")]]
  }
}

# 计算每时刻每组的中位数和95% CI：
df_surv <- lapply(1:J, function(j) {
  mat <- S_matrix[, j, ]
  
  data.frame(
    time = tau_grid,
    group = factor(j),
    median = apply(mat, 2, median),
    lower = apply(mat, 2, quantile, 0.025),
    upper = apply(mat, 2, quantile, 0.975)
  )
}) %>% bind_rows()


# 假如你有不同的标签：
group_labels <- levels(factor(bcos$treatment))
df_surv$group <- factor(
  df_surv$group,
  levels = 1:J,
  labels = group_labels
)


# 最后你可以用 ggplot 进行对比：
p <- ggplot() + 
  # 贝叶斯生存曲线
  geom_line(data = df_surv, 
            aes(x = time, y = median, color = group), size = 1) + 
  geom_ribbon(data = df_surv, 
              aes(x = time, ymin = lower, ymax = upper, fill = group), 
              alpha = 0.2, color = NA) + 
  # KM非参数生存曲线
  geom_step(data = fit_df, 
            aes(x = time, y = estimate, color = strata), 
            size = 1) + 
  labs(
    title = "Interval-Censored Survival Curve by Treatment (bcos)", 
    x = "Time", 
    y = "Survival Probability", 
    color = "Treatment",
    fill = "Treatment"
  ) + 
  theme_minimal()

print(p)





draws <- as_draws_df(fit$draws())

# 提取 S_curve，维度：[samples, L_grid + 1]
S_mat <- as.matrix(fit$draws("S_curve"))
n_times <- stan_data$L_grid + 1
n_samples <- nrow(S_mat)

# 时间轴
eval_times <- seq(0, stan_data$tau, length.out = n_times)

# 计算 posterior 中位数及置信区间
df_surv <- data.frame(
  time = eval_times,
  median = apply(S_mat, 2, median),
  lower = apply(S_mat, 2, quantile, 0.025),
  upper = apply(S_mat, 2, quantile, 0.975)
)

# 可视化
ggplot(df_surv, aes(x = time, y = median)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "blue") +
  labs(
    title = "Posterior Survival Function (Weibull-MDP)",
    x = "Time",
    y = "Survival Probability"
  ) +
  theme_minimal()