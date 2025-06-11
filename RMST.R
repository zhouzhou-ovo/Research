library(frailtypack)
library(jsonlite)
library(survival)
library(dplyr)
library(broom)
library(ggplot2)

# load data
data(bcos)
head(bcos)
bcos$left <- as.numeric(bcos$left)
bcos$right <- as.numeric(bcos$right)
bcos$treatment <- recode(bcos$treatment,
                         "RadChem" = "RCT",
                         "Rad" = "RT")

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

