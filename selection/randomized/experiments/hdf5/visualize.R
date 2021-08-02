library(ggplot2)
library(tidyverse)

## atomic study
atom.wide <- read.csv('posterior-atomic.csv')

atom <- tidyr::pivot_longer(
                       atom.wide,
                       cols = coverage_naive:fn_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

atom.cov <- atom[atom$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

atom.len <- atom[atom$metric == 'length',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')


atom.sigdet <- atom[atom$metric %in% c('tp','fp','tn','fn'),] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    facet_wrap(~metric, scales='free_y') +
    theme_bw(base_size = 30) +
    ylab('Count')


## ggsave('atom-cov.png', atom.cov, width = 19.20, height = 10.80, units = 'in')
## ggsave('atom-len.png', atom.len, width = 19.20, height = 10.80, units = 'in')
## ggsave('atom-sigdet.png', atom.sigdet, width = 19.20, height = 10.80, units = 'in')

## balanced study
bal.wide <- read.csv('posterior-balanced.csv')

bal <- tidyr::pivot_longer(
                       bal.wide,
                       cols = coverage_naive:fn_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

bal.cov <- bal[bal$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

bal.len <- bal[bal$metric == 'length',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')


bal.sigdet <- bal[bal$metric %in% c('tp','fp','tn','fn'),] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    facet_wrap(~metric, scales='free_y') +
    theme_bw(base_size = 30) +
    ylab('Count')


## ggsave('bal-cov.png', bal.cov, width = 19.20, height = 10.80, units = 'in')
## ggsave('bal-len.png', bal.len, width = 19.20, height = 10.80, units = 'in')
## ggsave('bal-sigdet.png', bal.sigdet, width = 19.20, height = 10.80, units = 'in')

## het study
het.wide <- read.csv('posterior-hetero.csv')

het <- tidyr::pivot_longer(
                       het.wide,
                       cols = coverage_naive:fn_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

het.cov <- het[het$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Upper, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position = position_dodge(width = 0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

het.len <- het[het$metric == 'length',] %>%
    ggplot(aes(x = Signal_Upper, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')


het.sigdet <- het[het$metric %in% c('tp','fp','tn','fn'),] %>%
    ggplot(aes(x = Signal_Upper, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position = position_dodge(width = 0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    facet_wrap(~metric, scales='free_y') +
    theme_bw(base_size = 30) +
    ylab('Count')


## ggsave('het-cov.png', het.cov, width = 19.20, height = 10.80, units = 'in')
## ggsave('het-len.png', het.len, width = 19.20, height = 10.80, units = 'in')
## ggsave('het-sigdet.png', het.sigdet, width = 19.20, height = 10.80, units = 'in')

## standardized study
std.wide <- read.csv('posterior-stdized.csv')

std <- tidyr::pivot_longer(
                       std.wide,
                       cols = coverage_naive:fn_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

std.cov <- std[std$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) + 
    ylab('Coverage')

std.len <- std[std$metric == 'length',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) + 
    ylab('Length')

std.sigdet <- std[std$metric %in% c('tp','fp','tn','fn'),] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    facet_wrap(~metric, scales='free_y') +
    theme_bw(base_size = 30) +
    ylab('Count')

## ggsave('std-cov.png', std.cov, width = 19.20, height = 10.80, units = 'in')
## ggsave('std-len.png', std.len, width = 19.20, height = 10.80, units = 'in')
## ggsave('std-sigdet.png', std.sigdet, width = 19.20, height = 10.80, units = 'in')

## og study
og.wide <- read.csv('posterior-og.csv')

og <- tidyr::pivot_longer(
                       og.wide,
                       cols = coverage_naive:fn_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

og.cov <- og[og$metric == 'coverage',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2, position=position_dodge(width=0.05)) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2, position = position_dodge(width = 0.05)) +
    theme_bw(base_size = 30) +
    ylab('Coverage')

og.len <- og[og$metric == 'length',] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    theme_bw(base_size = 30) +
    ylab('Length')


og.sigdet <- og[og$metric %in% c('tp','fp','tn','fn'),] %>%
    ggplot(aes(x = Signal_Fac, y = value, color = method, style = method)) +
    stat_summary(fun.data = mean_se, geom = 'errorbar', size = 2 ) +
    stat_summary(fun.data = mean_se, geom = 'line', size = 2) +
    facet_wrap(~metric, scales='free_y') +
    theme_bw(base_size = 30) +
    ylab('Count')


## ggsave('og-cov.png', og.cov, width = 19.20, height = 10.80, units = 'in')
## ggsave('og-len.png', og.len, width = 19.20, height = 10.80, units = 'in')
## ggsave('og-sigdet.png', og.sigdet, width = 19.20, height = 10.80, units = 'in')

## compact plots

names(het)[2] <- 'Signal_Fac'
atom$Setting <- 'Atomic'
bal$Setting <- 'Balanced'
het$Setting <- 'Heterogeneous'
std$Setting <- 'Standardized'
og$Setting <- 'Overlapping'

res <- rbind(atom,bal,het,std,og)
res$SNR <- as.factor(res$Signal_Fac)
res$Method <- recode(res$method, naive = 'Naive', posi = 'Selection-informed', split50 = 'Split (1:1)', split67= 'Split (2:1)')

snr.labels <- as_labeller(c('0.2' = 'Low SNR', '0.5' = 'Medium SNR', '1.5' = 'High SNR'))

cmp.cov.can <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'coverage') %>%
    filter(Setting %in% c('Atomic','Balanced','Heterogeneous')) %>%
    ggplot(aes(x = Setting, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    geom_hline(yintercept=0.9, linetype='dashed') +
    ylab('Coverage') +
    theme_bw(base_size = 30)

cmp.len.can <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'length') %>%
    filter(Setting %in% c('Atomic','Balanced','Heterogeneous')) %>%
    ggplot(aes(x = Setting, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    ylab('Length') +
    theme_bw(base_size = 30)


cmp.cov.ext <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'coverage') %>%
    filter(Setting %in% c('Standardized','Overlapping')) %>%
    ggplot(aes(x = Setting, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    geom_hline(yintercept=0.9, linetype='dashed') +
    ylab('Coverage') +
    theme_bw(base_size = 30)

cmp.len.ext <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'length') %>%
    filter(Setting %in% c('Standardized','Overlapping')) %>%
    ggplot(aes(x = Setting, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    ylab('Length') +
    theme_bw(base_size = 30)


ggsave('canonical-coverage.png', cmp.cov.can, width = 19.20, height = 10.80, units = 'in')
ggsave('canonical-length.png', cmp.len.can, width = 19.20, height = 10.80, units = 'in')
ggsave('extensions-coverage.png', cmp.cov.ext, width = 19.20, height = 10.80, units = 'in')
ggsave('extensions-length.png', cmp.len.ext, width = 19.20, height = 10.80, units = 'in')

## punchline plots for talks (simplified; just hetero and OG case)

cmp.cov.can.punch <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'coverage') %>%
    filter(Setting %in% c('Heterogeneous')) %>%
    ggplot(aes(x = Method, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    geom_hline(yintercept=0.9, linetype='dashed') +
    ylab('Coverage') +
    theme_bw(base_size = 30) +
    guides(fill = guide_legend(override.aes = list(size=20))) +
    theme(axis.text.x=element_blank())

cmp.len.can.punch <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == 'length') %>%
    filter(Setting %in% c('Heterogeneous')) %>%
    ggplot(aes(x = Method, y = value, color = Method)) +
    geom_boxplot() +
    facet_wrap(~ SNR, ncol = 1, labeller = snr.labels) +
    ylab('Length') +
    theme_bw(base_size = 30) +
    guides(fill = guide_legend(override.aes = list(size=20))) +
    theme(axis.text.x=element_blank())

cmp.cov.og.punch <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = Method, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  theme(axis.text.x = element_blank())

cmp.len.og.punch <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = Method, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("Length") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  theme(axis.text.x = element_blank())

ggsave('canonical-coverage-punchline.png', cmp.cov.can.punch, width = 19.20, height = 10.80, units = 'in')
ggsave('canonical-length-punchline.png', cmp.len.can.punch, width = 19.20, height = 10.80, units = 'in')
ggsave("og-coverage-punchline.png", cmp.cov.og.punch, width = 19.20, height = 10.80, units = "in")
ggsave("og-length-punchline.png", cmp.len.og.punch, width = 19.20, height = 10.80, units = "in")


## alternate style punchline plots
cmp.cov.can.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Heterogeneous")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  geom_boxplot() +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.len.can.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Heterogeneous")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  geom_boxplot() +
  ylab("Length") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.cov.og.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  geom_boxplot() +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.len.og.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  geom_boxplot() +
  ylab("Length") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

ggsave("canonical-coverage-punchline-alt.png", cmp.cov.can.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("canonical-length-punchline-alt.png", cmp.len.can.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("og-coverage-punchline-alt.png", cmp.cov.og.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("og-length-punchline-alt.png", cmp.len.og.punch.alt, width = 19.20, height = 10.80, units = "in")
