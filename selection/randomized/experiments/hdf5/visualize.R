library(ggplot2)
library(tidyverse)

## atomic study
atom.wide <- read.csv('posterior-atomic.csv')

atom <- tidyr::pivot_longer(
                       atom.wide,
                       cols = coverage_naive:fn_posi|msetarget_naive:runtime_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

## balanced study
bal.wide <- read.csv('posterior-balanced.csv')

bal <- tidyr::pivot_longer(
                       bal.wide,
                       cols = coverage_naive:fn_posi|msetarget_naive:runtime_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

## het study
het.wide <- read.csv('posterior-hetero.csv')

het <- tidyr::pivot_longer(
                       het.wide,
                       cols = coverage_naive:fn_posi|msetarget_naive:runtime_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

## standardized study
std.wide <- read.csv('posterior-stdized.csv')

std <- tidyr::pivot_longer(
                       std.wide,
                       cols = coverage_naive:fn_posi|msetarget_naive:runtime_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

## og study
og.wide <- read.csv('posterior-og.csv')

og <- tidyr::pivot_longer(
                       og.wide,
                       cols = coverage_naive:fn_posi|msetarget_naive:runtime_posi,
                       names_to = c('metric','method'),
                       names_sep = '_')

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

cmp.runtime.can <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "runtime") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("Runtime (seconds)") +
  theme_bw(base_size = 30)

cmp.msetarget.can <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "msetarget") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("MSE (Projected Target)") +
  theme_bw(base_size = 30)

cmp.msetruth.can <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "msetruth") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("MSE (True Target)") +
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

cmp.runtime.ext <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "runtime") %>%
  filter(Setting %in% c('Standardized','Overlapping')) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("Runtime (seconds)") +
  theme_bw(base_size = 30)

cmp.msetarget.ext <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "msetarget") %>%
  filter(Setting %in% c('Standardized','Overlapping')) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("MSE (Projected Target)") +
  theme_bw(base_size = 30)

cmp.msetruth.ext <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "msetruth") %>%
  filter(Setting %in% c('Standardized','Overlapping')) %>%
  ggplot(aes(x = Setting, y = value, color = Method)) +
  geom_boxplot() +
  facet_wrap(~SNR, ncol = 1, labeller = snr.labels) +
  ylab("MSE (True Target)") +
  theme_bw(base_size = 30)

ggsave('canonical-coverage.png', cmp.cov.can, width = 19.20, height = 10.80, units = 'in')
ggsave('canonical-length.png', cmp.len.can, width = 19.20, height = 10.80, units = 'in')
ggsave("canonical-runtime.png", cmp.runtime.can, width = 19.20, height = 10.80, units = "in")
ggsave("canonical-msetarget.png", cmp.msetarget.can, width = 19.20, height = 10.80, units = "in")
ggsave("canonical-msetruth.png", cmp.msetruth.can, width = 19.20, height = 10.80, units = "in")
ggsave('extensions-coverage.png', cmp.cov.ext, width = 19.20, height = 10.80, units = 'in')
ggsave('extensions-length.png', cmp.len.ext, width = 19.20, height = 10.80, units = 'in')
ggsave("extensions-runtime.png", cmp.runtime.ext, width = 19.20, height = 10.80, units = "in")
ggsave("extensions-msetarget.png", cmp.msetarget.ext, width = 19.20, height = 10.80, units = "in")
ggsave("extensions-msetruth.png", cmp.msetruth.ext, width = 19.20, height = 10.80, units = "in")

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


## alternate style punchline plots, no outliers in boxplots
## use approach from https://stackoverflow.com/a/57825639/288545

skinnybox <- function(x) {
  coef <- 1.5
  stats <- quantile(x, probs = c())
  # calculate quantiles
  stats <- quantile(x, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))
  names(stats) <- c("ymin", "lower", "middle", "upper", "ymax")
  return(stats)
}

cmp.cov.can.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Heterogeneous")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = 'dodge2') +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.len.can.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Heterogeneous")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = 'dodge2') +
  ylab("Length") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.cov.og.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = 'dodge2') +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

cmp.len.og.punch.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Overlapping")) %>%
  ggplot(aes(x = SNR, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = 'dodge2') +
  ylab("Length") +
  theme_bw(base_size = 30) +
  guides(fill = guide_legend(override.aes = list(size = 20))) +
  scale_x_discrete(labels = snr.labels) +
  xlab("")

ggsave("canonical-coverage-punchline-alt.png", cmp.cov.can.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("canonical-length-punchline-alt.png", cmp.len.can.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("og-coverage-punchline-alt.png", cmp.cov.og.punch.alt, width = 19.20, height = 10.80, units = "in")
ggsave("og-length-punchline-alt.png", cmp.len.og.punch.alt, width = 19.20, height = 10.80, units = "in")
