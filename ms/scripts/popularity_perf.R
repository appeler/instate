library(readr)
library(ggplot2)

a <- read_csv("lstm.csv")
ggplot(a, 
       aes(x = log(total_freq), 
           y = lstm_pred)) +
  geom_point(color= "steelblue") +
  geom_smooth(color = "tomato")

ggsave("popularity_perf_with_se.png")
