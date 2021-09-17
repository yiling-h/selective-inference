library(ggplot2)
library(tidyverse)

## atomic study
atom.wide <- read.csv('posterior-atomic.csv')

atom <- tidyr::pivot_longer(
                       atom.wide,
                       cols = coverage_naive:postcoverage_posi33,
                       names_to = c('metric','method'),
                       names_sep = '_')

## balanced study
bal.wide <- read.csv('posterior-balanced.csv')

bal <- tidyr::pivot_longer(
                       bal.wide,
                       cols = coverage_naive:postcoverage_posi33,
                       names_to = c('metric','method'),
                       names_sep = '_')

## het study
het.wide <- read.csv('posterior-hetero.csv')

het <- tidyr::pivot_longer(
                       het.wide,
                       cols = coverage_naive:postcoverage_posi33,
                       names_to = c('metric','method'),
                       names_sep = '_')

## standardized study
std.wide <- read.csv('posterior-stdized.csv')

std <- tidyr::pivot_longer(
                       std.wide,
                       cols = coverage_naive:postcoverage_posi33,
                       names_to = c('metric','method'),
                       names_sep = '_')

## og study
og.wide <- read.csv('posterior-og.csv')

og <- tidyr::pivot_longer(
                       og.wide,
                       cols = coverage_naive:postcoverage_posi33,
                       names_to = c('metric','method'),
                       names_sep = '_')

## compact plots

names(het)[2] <- 'Signal_Fac'
atom$Setting <- 'Atomic'
bal$Setting <- 'Balanced'
het$Setting <- 'Heterogeneous'
std$Setting <- 'Standardized'
og$Setting <- 'Overlapping'

res <- rbind(atom, bal)
res <- rbind(atom, bal, het, std, og)
res$SNR <- as.factor(res$Signal_Fac)

res$Method <- recode(res$method,
  naive = "Naive",
  posi50 = "Selection-informed",
  posi67 = "Selection-informed",
  posi33 = "Selection-informed",
  split50 = "Split",
  split67 = "Split",
  split33 = "Split"
)


res$QueryProp <- factor(recode(res$method,
  naive = "0",
  posi50 = "1:1",
  posi67 = "2:1",
  posi33 = "1:2",
  split50 = "1:1",
  split67 = "2:1",
  split33 = "1:2"
),
levels = c("1:2", "1:1", "2:1", "0")
)


snr.labels <- as_labeller(c('0.2' = 'Low SNR', '0.5' = 'Medium SNR', '1.5' = 'High SNR',
                            "Atomic" = "Atomic",
                            "Balanced" = "Balanced",
                            "Heterogeneous" = "Heterogeneous",
                            "Standardized" = "Standardized",
                            "Overlapping" = "Overlapping"
                            ))

## compute quality of query
tps <- res[res$metric == "tp", ]$value
fps <- res[res$metric == "fp", ]$value
tns <- res[res$metric == "tn", ]$value
fns <- res[res$metric == "fn", ]$value
frame <- res[res$metric == "tp", ]

acc <- frame
acc$metric <- "Query Accuracy"
acc$value <- (tps + tns) / (tps + tns + fps + fns)

res <- rbind(res, acc)

## compute quality of query with F1 score
tps <- res[res$metric == "tp", ]$value
fps <- res[res$metric == "fp", ]$value
tns <- res[res$metric == "tn", ]$value
fns <- res[res$metric == "fn", ]$value
frame <- res[res$metric == "tp", ]

acc <- frame
acc$metric <- "Query F1 Score"
acc$value <- (tps) / (tps + 1/2 * (fps + fns))

res <- rbind(res, acc)

## compute TPR and FPR
tps <- res[res$metric == "posttp", ]$value
fps <- res[res$metric == "postfp", ]$value
tns <- res[res$metric == "posttn", ]$value
fns <- res[res$metric == "postfn", ]$value
frame <- res[res$metric == "posttp", ]

tpr <- frame
tpr$metric <- "TPR"
tpr$value <- tps / (tps + fns)

fpr <- frame
fpr$metric <- "FPR"
fpr$value <- fps / (fps + tns)

fdp <- frame
fdp$metric <- "FDP"
fdp$value <- fps / max(tps + fps, 1)

res <- rbind(res, tpr, fpr, fdp)

## start making plots
res <- filter(res, Setting != "Atomic")

can.que <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "Query F1 Score") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Query F1 Score") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

can.cov <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

can.len <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Length") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.que <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "Query F1 Score") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Query F1 Score") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.cov <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.len <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  geom_boxplot(position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Length") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)


ggsave("canonical-query.png", can.que, width = 19.20, height = 10.80, units = "in")
ggsave('canonical-coverage.png', can.cov, width = 19.20, height = 10.80, units = 'in')
ggsave('canonical-length.png', can.len, width = 19.20, height = 10.80, units = 'in')
ggsave("extensions-query.png", ext.que, width = 19.20, height = 10.80, units = "in")
ggsave('extensions-coverage.png', ext.cov, width = 19.20, height = 10.80, units = 'in')
ggsave('extensions-length.png', ext.len, width = 19.20, height = 10.80, units = 'in')

## alternate boxplots (no outliers)
skinnybox <- function(x) {
  coef <- 1.5
  stats <- quantile(x, probs = c())
  # calculate quantiles
  stats <- quantile(x, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))
  names(stats) <- c("ymin", "lower", "middle", "upper", "ymax")
  return(stats)
}

can.que.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "Query F1 Score") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Query F1 Score") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

can.cov.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

can.len.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Atomic", "Balanced", "Heterogeneous")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Length") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.que.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "Query F1 Score") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Query F1 Score") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.cov.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "coverage") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  ylab("Coverage") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)

ext.len.alt <- filter(res, SNR %in% c(0.2, 0.5, 1.5) & metric == "length") %>%
  filter(Setting %in% c("Standardized", "Overlapping")) %>%
  ggplot(aes(x = QueryProp, y = value, color = Method)) +
  stat_summary(fun.data = skinnybox, geom = 'boxplot', position = position_dodge2(preserve = "single")) +
  facet_grid(rows = vars(SNR), cols = vars(Setting), labeller = snr.labels) +
  ylab("Length") +
  xlab("Level of Randomization") +
  theme_bw(base_size = 30)


ggsave("alt-canonical-query.png", can.que.alt, width = 19.20, height = 10.80, units = "in")
ggsave('alt-canonical-coverage.png', can.cov.alt, width = 19.20, height = 10.80, units = 'in')
ggsave('alt-canonical-length.png', can.len.alt, width = 19.20, height = 10.80, units = 'in')
ggsave("alt-extensions-query.png", ext.que.alt, width = 19.20, height = 10.80, units = "in")
ggsave('alt-extensions-coverage.png', ext.cov.alt, width = 19.20, height = 10.80, units = 'in')
ggsave('alt-extensions-length.png', ext.len.alt, width = 19.20, height = 10.80, units = 'in')

