library(ggplot2)
library(tidyr)

## atomic study
atom.wide <- read.csv('posterior-atomic.csv')

atom <- tidyr::pivot_longer(
                       atom.wide,
                       cols = coverage_naive:length_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

atom.cov <- atom[atom$metric == 'coverage',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

atom.len <- atom[atom$metric == 'length',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')

ggsave('atom-cov.png', atom.cov, width = 19.20, height = 10.80, units = 'in')
ggsave('atom-len.png', atom.len, width = 19.20, height = 10.80, units = 'in')

## balanced study
bal.wide <- read.csv('posterior-balanced.csv')

bal <- tidyr::pivot_longer(
                       bal.wide,
                       cols = coverage_naive:length_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

bal.cov <- bal[bal$metric == 'coverage',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

bal.len <- bal[bal$metric == 'length',] %>%
    ggplot(aes(x = Signal, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')

ggsave('bal-cov.png', bal.cov, width = 19.20, height = 10.80, units = 'in')
ggsave('bal-len.png', bal.len, width = 19.20, height = 10.80, units = 'in')

## het study
het.wide <- read.csv('posterior-hetero.csv')

het <- tidyr::pivot_longer(
                       het.wide,
                       cols = coverage_naive:length_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

het.cov <- het[het$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Lower, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position = position_dodge(width = 0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

het.len <- het[het$metric == 'length',] %>%
    ggplot(aes(x = Signal_Lower, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')

ggsave('het-cov.png', het.cov, width = 19.20, height = 10.80, units = 'in')
ggsave('het-len.png', het.len, width = 19.20, height = 10.80, units = 'in')

