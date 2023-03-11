# Set working dir
setwd(githubdir)
setwd("instate/ms/scripts/")

# Load libs
library(tidyverse)
library(readr)
library(tidyr)
library(purrr)
library(ggplot2)

dnn <- read_csv("../out/dnn_pred.csv")

# Skew in popularity
summary(dnn$total_freq)
quantile(dnn$total_freq, c(.90, .95, .99, .997))

# We cannot get a great estimate of the relationship in the tail so need to remove some of the 'outliers'
dnn_fin <- dnn[dnn$total_freq <= 2000, ]

# Long form for group_by loess
models <- dnn_fin %>%
            select(- c("...1", "total_freq_n", "gt_state", "female_prop")) %>%
            pivot_longer(cols = c("lstm_pred", "rnn_pred", "gru_pred"), names_to = "model", values_to = "correct_or_not") %>%
            tidyr::nest(data = -model) %>%
            dplyr::mutate(
                # Perform loess calculation on each CpG group
                m = purrr::map(data, loess,
                               formula = correct_or_not ~ total_freq, span = .75),
                # Retrieve the fitted values from each model
                fitted = purrr::map(m, `[[`, "fitted")
        )

# Apply fitted y's as a new column
results <- models %>%
        dplyr::select(-m) %>%
        tidyr::unnest(cols = c(data, fitted))

case_when(
  x %% 35 == 0 ~ "fizz buzz",
  x %% 5 == 0 ~ "fizz",
  x %% 7 == 0 ~ "buzz",
  .default = as.character(x)
)

# Plot with loess line for each group
ggplot(results, aes(x = total_freq, y = correct_or_not, group = model)) +
        geom_point(alpha = .05, size = 3) +
        geom_line(aes(y = fitted, colour = model)) + 
        theme_minimal() +
        xlab("Frequency of the name") + 
        ylab("Accuracy") + 
        theme(panel.grid.major = element_line(color="#e1e1e1",  linetype = "dotted"),
          panel.grid.minor = element_blank(),
          legend.position  ="bottom",
          legend.key      = element_blank())


ggsave("../figs/popularity_perf.pdf")
ggsave("../figs/popularity_perf.png")

## Gender Perf 
# Long form for group_by loess
models <- dnn_fin %>%
            select(- c("...1", "total_freq_n", "gt_state", "total_freq")) %>%
            pivot_longer(cols = c("lstm_pred", "rnn_pred", "gru_pred"), names_to = "model", values_to = "correct_or_not") %>%
            tidyr::nest(data = -model) %>%
            dplyr::mutate(
                # Perform loess calculation on each CpG group
                m = purrr::map(data, loess,
                               formula = correct_or_not ~ female_prop, span = .75),
                # Retrieve the fitted values from each model
                fitted = purrr::map(m, `[[`, "fitted")
        )

# Apply fitted y's as a new column
results <- models %>%
        dplyr::select(-m) %>%
        tidyr::unnest(cols = c(data, fitted))

# Plot with loess line for each group
ggplot(results, aes(x = female_prop, y = correct_or_not, group = model, colour = model)) +
        geom_point() +
        geom_line(aes(y = fitted)) + 
        theme_minimal()
ggsave("../figs/gender_perf.pdf")
ggsave("../figs/gender_perf.png")

