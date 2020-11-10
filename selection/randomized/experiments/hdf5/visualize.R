library(ggplot2)
library(tidyr)

dat.wide <- read.csv('posterior-balanced.csv')

dat <- tidyr::pivot_longer(
                       dat.wide,
                       cols = coverage_naive:length_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

dat[dat$metric == 'coverage',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

dat[dat$metric == 'length',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')
