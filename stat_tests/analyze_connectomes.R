#!/usr/bin/env Rscript
# =============================================================================
# analyze_connectomes.R
# =============================================================================
#
# C. elegans connectome-configuration analysis (self-contained).
#
# Inputs (one CSV per connectome configuration):
#   celegans/results_celegans_phys_contact.csv
#   celegans/results_celegans_syn_count.csv
#   celegans/results_celegans_syn_size.csv
#
# For each Null k in {1, 2, 3} and each experiment, fits:
#   outcome_adj ~ network * connectome + network * Age + network * Series
# with sign convention original − null (negative ⇒ null outperforms original).
#
# Also draws per-connectome 5×5 raw-mean age profiles (experiments × series).
#
# Run from the repo root:
#   Rscript celegans/tests_230726/analyze_connectomes.R
# =============================================================================

# --- Paths & project library -------------------------------------------------

# Script lives in celegans/tests_230726/; repo root is two levels up.
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
this_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  normalizePath(file.path("celegans", "tests_230726"))
}
repo_root <- normalizePath(file.path(this_dir, "..", ".."))
source(file.path(repo_root, "R", "setup.R"))
setup_project(repo_root)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(glmmTMB)
  library(broom.mixed)
  library(emmeans)
})

# Output root for this analysis batch
OUT_ROOT <- this_dir
PLOTS_DIR <- file.path(OUT_ROOT, "plots")
OUTPUT_DIR <- file.path(OUT_ROOT, "output")

# --- Constants ---------------------------------------------------------------

BEHAVIORAL_TASKS <- c("PerceptualDecisionMaking", "GoNogo")
FORECAST_TASKS <- c("MemoryCapacity", "henon_map", "mackey_glass")
ALL_TASKS <- c(BEHAVIORAL_TASKS, FORECAST_TASKS)

TASK_LABELS <- c(
  PerceptualDecisionMaking = "Perceptual Decision Making",
  GoNogo = "Go/No-go",
  MemoryCapacity = "Memory Capacity",
  henon_map = "Henon Map",
  mackey_glass = "Mackey-Glass"
)

# Internal connectome id → display name (replaces "Culture" in outputs)
CONNECTOME_LABELS <- c(
  phys_contact = "physical contact",
  syn_count = "synapse count",
  syn_size = "synapse size"
)
CONNECTOME_IDS <- names(CONNECTOME_LABELS)

# Raw Age code → plot label (A1a/A1b kept distinct for syn_count profiles)
AGE_PLOT_MAP <- c(
  "1" = "L1.1", "2" = "L1.2", "3" = "L1.3", "4" = "L1.4",
  "5" = "L2", "6" = "L3", "7" = "A1a", "8" = "A1b"
)

# Model Age levels: 7 and 8 both pool to A1
AGE_MODEL_LEVELS <- c("L1.1", "L1.2", "L1.3", "L1.4", "L2", "L3", "A1")

# X-axis order for profiles (A1 is one position; A1a/A1b share it)
AGE_X_LEVELS <- c("L1.1", "L1.2", "L1.3", "L1.4", "L2", "L3", "A1")

NETWORK_COLORS <- c(
  original = "#2166AC",  # blue
  `Null 1` = "#B2182B",  # red
  `Null 2` = "#E66101",  # orange
  `Null 3` = "#FDB863"   # yellow
)

# Input CSVs (relative to repo root)
INPUT_FILES <- c(
  phys_contact = file.path(repo_root, "celegans", "results_celegans_phys_contact.csv"),
  syn_count = file.path(repo_root, "celegans", "results_celegans_syn_count.csv"),
  syn_size = file.path(repo_root, "celegans", "results_celegans_syn_size.csv")
)

# --- Small helpers -----------------------------------------------------------

adjust_for_beta <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

derive_outcome <- function(experiment, metric1, metric2) {
  ifelse(experiment %in% BEHAVIORAL_TASKS, metric2, metric1^2)
}

parse_series_num <- function(series) {
  as.integer(sub(".*series_(\\d+).*", "\\1", series))
}

parse_null_variant <- function(series) {
  out <- rep(NA_integer_, length(series))
  hits <- grepl("_null_", series, fixed = TRUE)
  if (any(hits)) {
    out[hits] <- as.integer(sub(".*_null_(\\d+)_.*", "\\1", series[hits]))
  }
  out
}

# Map raw age integer/character → plot label (A1a / A1b) and model label (A1)
age_plot_label <- function(age_raw) {
  key <- as.character(as.integer(as.character(age_raw)))
  unname(AGE_PLOT_MAP[key])
}

age_model_label <- function(age_plot) {
  ifelse(age_plot %in% c("A1a", "A1b"), "A1", age_plot)
}

# X position for plotting: A1a and A1b both sit at "A1"
age_x_label <- function(age_plot) {
  ifelse(age_plot %in% c("A1a", "A1b"), "A1", age_plot)
}

format_p <- function(p) {
  ifelse(is.na(p), "—",
    ifelse(p < 0.001, "< .001", sprintf("%.3f", p)))
}

sig_stars <- function(p) {
  ifelse(is.na(p), "",
    ifelse(p < 0.001, "***",
      ifelse(p < 0.01, "**",
        ifelse(p < 0.05, "*", ""))))
}

sig_bg_color <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "#ffffff",
    p < 0.001 ~ "#a8d5a2",
    p < 0.01 ~ "#c8e6c9",
    p < 0.05 ~ "#e8f5e9",
    TRUE ~ "#ffffff"
  )
}

# --- Data loading ------------------------------------------------------------

load_all_connectomes <- function() {
  # Read each CSV, tag connectome from filename/Culture, derive columns.
  pieces <- lapply(CONNECTOME_IDS, function(cid) {
    path <- INPUT_FILES[[cid]]
    if (!file.exists(path)) stop("Missing input file: ", path)

    read_csv(path, show_col_types = FALSE) |>
      filter(Experiment %in% ALL_TASKS) |>
      mutate(
        connectome = cid,
        connectome_label = CONNECTOME_LABELS[[cid]],
        metric1 = `Metric 1`,
        metric2 = `Metric 2`,
        outcome = derive_outcome(Experiment, metric1, metric2),
        series_num = parse_series_num(Series),
        null_variant = parse_null_variant(Series),
        # Binary network for models; null_variant keeps which Null k
        network = ifelse(is.na(null_variant), "original", "null"),
        age_plot = {
          ap <- age_plot_label(Age)
          # Only syn_count keeps A1a/A1b distinct; others map both to A1
          if (cid != "syn_count") {
            ap <- ifelse(ap %in% c("A1a", "A1b"), "A1", ap)
          }
          ap
        },
        Age = age_model_label(age_plot),
        age_x = age_x_label(age_plot),
        network_plot = case_when(
          is.na(null_variant) ~ "original",
          null_variant == 1L ~ "Null 1",
          null_variant == 2L ~ "Null 2",
          null_variant == 3L ~ "Null 3"
        )
      )
  })

  bind_rows(pieces) |>
    filter(!is.na(outcome), outcome >= 0, outcome <= 1) |>
    mutate(
      connectome = factor(connectome, levels = CONNECTOME_IDS),
      Age = factor(Age, levels = AGE_MODEL_LEVELS),
      Series = factor(series_num, levels = sort(unique(series_num))),
      network = factor(network, levels = c("null", "original")),
      network_plot = factor(network_plot, levels = names(NETWORK_COLORS)),
      Experiment = factor(Experiment, levels = ALL_TASKS),
      age_x = factor(age_x, levels = AGE_X_LEVELS),
      age_plot = factor(
        age_plot,
        levels = c("L1.1", "L1.2", "L1.3", "L1.4", "L2", "L3", "A1a", "A1b", "A1")
      )
    )
}

# --- Age profile plots (descriptive) -----------------------------------------

summarise_age_profile <- function(df) {
  # Raw mean ± 95% normal CI per experiment × series × network × age_plot
  df |>
    group_by(connectome, connectome_label, Experiment, Series, network_plot, age_plot, age_x) |>
    summarise(
      n = n(),
      mean = mean(outcome),
      se = sd(outcome) / sqrt(n()),
      ci_low = mean - 1.96 * se,
      ci_high = mean + 1.96 * se,
      .groups = "drop"
    ) |>
    mutate(
      ci_low = pmax(0, ci_low),
      ci_high = pmin(1, ci_high),
      experiment_label = TASK_LABELS[as.character(Experiment)],
      series_label = paste("Series", as.character(Series))
    )
}

plot_age_profile <- function(means_df, connectome_id, plots_dir) {
  display <- CONNECTOME_LABELS[[connectome_id]]
  plot_df <- means_df |>
    filter(connectome == connectome_id) |>
    mutate(
      experiment_label = factor(experiment_label, levels = unname(TASK_LABELS[ALL_TASKS])),
      series_label = factor(series_label, levels = paste("Series", 1:5)),
      # For syn_count: encode A1a/A1b in a hue group; elsewhere all "—"
      a1_hue = case_when(
        age_plot == "A1a" ~ "A1a",
        age_plot == "A1b" ~ "A1b",
        TRUE ~ "other"
      )
    )

  # Dodge A1a/A1b slightly so both are visible at the same x
  plot_df <- plot_df |>
    mutate(
      x_num = as.numeric(age_x) + case_when(
        a1_hue == "A1a" ~ -0.12,
        a1_hue == "A1b" ~ 0.12,
        TRUE ~ 0
      )
    )

  p <- ggplot(plot_df, aes(x = x_num, y = mean, color = network_plot, group = interaction(network_plot, a1_hue))) +
    geom_line(aes(linetype = a1_hue), linewidth = 0.6) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, linewidth = 0.35) +
    geom_point(aes(shape = a1_hue), size = 1.8) +
    facet_grid(rows = vars(experiment_label), cols = vars(series_label)) +
    scale_color_manual(values = NETWORK_COLORS, name = "Network") +
    scale_x_continuous(
      breaks = seq_along(AGE_X_LEVELS),
      labels = AGE_X_LEVELS,
      limits = c(0.5, length(AGE_X_LEVELS) + 0.5)
    ) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(
      title = paste0("C. Elegans - ", display),
      subtitle = "Raw mean outcome ± 95% CI (response scale)",
      x = "Age",
      y = "Outcome (response scale)"
    ) +
    theme_bw() +
    theme(
      panel.grid.minor = element_blank(),
      strip.text = element_text(size = 8),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
      legend.position = "bottom"
    )

  if (connectome_id == "syn_count") {
    p <- p +
      scale_linetype_manual(
        values = c(other = "solid", A1a = "solid", A1b = "dashed"),
        breaks = c("A1a", "A1b"),
        name = "A1 individual"
      ) +
      scale_shape_manual(
        values = c(other = 16, A1a = 16, A1b = 17),
        breaks = c("A1a", "A1b"),
        name = "A1 individual"
      )
  } else {
    p <- p +
      scale_linetype_manual(values = c(other = "solid", A1a = "solid", A1b = "solid"), guide = "none") +
      scale_shape_manual(values = c(other = 16, A1a = 16, A1b = 16), guide = "none")
  }

  dir.create(file.path(plots_dir, connectome_id), recursive = TRUE, showWarnings = FALSE)
  ggsave(
    file.path(plots_dir, connectome_id, "age_profile.png"),
    p, width = 18, height = 14, dpi = 300, bg = "white"
  )
}

write_age_profiles <- function(df, plots_dir, output_dir) {
  means <- summarise_age_profile(df)
  for (cid in CONNECTOME_IDS) {
    dir.create(file.path(output_dir, cid), recursive = TRUE, showWarnings = FALSE)
    sub <- means |> filter(connectome == cid)
    write_csv(sub, file.path(output_dir, cid, "age_profile_means.csv"))
    plot_age_profile(means, cid, plots_dir)
    cat(sprintf("  Profile written for %s (%d rows)\n", cid, nrow(sub)))
  }
  invisible(means)
}

# --- Inferential models (option B) -------------------------------------------

prepare_model_df <- function(df) {
  df |> mutate(outcome_adj = adjust_for_beta(outcome))
}

fit_track_b <- function(df) {
  d <- prepare_model_df(df)
  # Ensure factor levels for this subset
  d$network <- factor(d$network, levels = c("null", "original"))
  d$connectome <- droplevels(d$connectome)
  d$Age <- droplevels(d$Age)
  d$Series <- droplevels(d$Series)

  model <- glmmTMB(
    outcome_adj ~ network * connectome + network * Age + network * Series,
    family = beta_family(link = "logit"),
    data = d
  )
  list(data = d, model = model)
}

lrt_one <- function(full, reduced, experiment, null_variant, term_removed) {
  cmp <- anova(reduced, full)
  tibble(
    experiment = experiment,
    experiment_label = unname(TASK_LABELS[experiment]),
    null_variant = null_variant,
    term_removed = term_removed,
    df = cmp$Df[2],
    chisq = cmp$Chisq[2],
    p.value = cmp$`Pr(>Chisq)`[2]
  )
}

run_omnibus <- function(fit, experiment, null_variant) {
  d <- fit$data
  m <- fit$model
  fam <- beta_family(link = "logit")

  # Full: network * connectome + network * Age + network * Series
  # Reduced models drop one interaction while keeping the matching main effects.
  bind_rows(
    lrt_one(
      m,
      glmmTMB(outcome_adj ~ network * connectome + Age + network * Series,
              family = fam, data = d),
      experiment, null_variant, "network:Age"
    ),
    lrt_one(
      m,
      glmmTMB(outcome_adj ~ network * connectome + network * Age + Series,
              family = fam, data = d),
      experiment, null_variant, "network:Series"
    ),
    lrt_one(
      m,
      glmmTMB(outcome_adj ~ network + connectome + network * Age + network * Series,
              family = fam, data = d),
      experiment, null_variant, "network:connectome"
    )
  )
}

contrast_original_vs_null <- function(model, by_factor, experiment, null_variant) {
  # emmeans contrast original − null at each level of by_factor
  exp_id <- experiment
  exp_label <- unname(TASK_LABELS[exp_id])
  emm <- emmeans(model, reformulate(c("network", by_factor)), type = "link")
  as.data.frame(summary(
    contrast(emm, method = "revpairwise", by = by_factor),
    infer = TRUE, adjust = "holm"
  )) |>
    mutate(
      experiment = exp_id,
      experiment_label = exp_label,
      null_variant = null_variant,
      level = as.character(.data[[by_factor]]),
      .before = 1
    ) |>
    rename(
      std.error = SE,
      statistic = z.ratio,
      conf.low = asymp.LCL,
      conf.high = asymp.UCL
    ) |>
    select(-any_of(by_factor))
}

collect_network_effect <- function(model, experiment, null_variant) {
  exp_id <- experiment
  tidy(model, effects = "fixed", conf.int = TRUE) |>
    filter(component == "cond", term == "networkoriginal") |>
    mutate(
      experiment = exp_id,
      experiment_label = unname(TASK_LABELS[exp_id]),
      null_variant = null_variant,
      .before = 1
    )
}

run_null_variant_analysis <- function(df, null_k, plots_dir, output_dir) {
  cat(sprintf("\n=== Null %d ===\n", null_k))
  out_k <- file.path(output_dir, sprintf("null_%d", null_k))
  plot_k <- file.path(plots_dir, sprintf("null_%d", null_k))
  dir.create(out_k, recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_k, recursive = TRUE, showWarnings = FALSE)

  # Original + this null only, all connectomes
  df_k <- df |>
    filter(is.na(null_variant) | null_variant == null_k) |>
    mutate(network = ifelse(is.na(null_variant), "original", "null"))

  fixed_rows <- list()
  omnibus_rows <- list()
  age_ctr <- list()
  series_ctr <- list()
  conn_ctr <- list()

  for (task in ALL_TASKS) {
    cat(sprintf("  Fitting %s ...\n", TASK_LABELS[[task]]))
    task_df <- df_k |> filter(Experiment == task)
    fit <- fit_track_b(task_df)

    fixed_rows[[task]] <- collect_network_effect(fit$model, task, null_k)
    omnibus_rows[[task]] <- run_omnibus(fit, task, null_k)
    age_ctr[[task]] <- contrast_original_vs_null(fit$model, "Age", task, null_k)
    series_ctr[[task]] <- contrast_original_vs_null(fit$model, "Series", task, null_k)
    conn_ctr[[task]] <- contrast_original_vs_null(fit$model, "connectome", task, null_k)
  }

  fixed_df <- bind_rows(fixed_rows)
  omnibus_df <- bind_rows(omnibus_rows)
  age_df <- bind_rows(age_ctr)
  series_df <- bind_rows(series_ctr)
  conn_df <- bind_rows(conn_ctr)

  # Friendly labels in CSVs
  conn_df <- conn_df |>
    mutate(
      level = ifelse(
        as.character(level) %in% names(CONNECTOME_LABELS),
        CONNECTOME_LABELS[as.character(level)],
        as.character(level)
      )
    )

  write_csv(fixed_df, file.path(out_k, "fixed_effects.csv"))
  write_csv(omnibus_df, file.path(out_k, "omnibus_interaction.csv"))
  write_csv(age_df, file.path(out_k, "div_network_contrast.csv"))
  write_csv(series_df, file.path(out_k, "series_network_contrast.csv"))
  write_csv(conn_df, file.path(out_k, "connectome_network_contrast.csv"))

  plot_contrast_forest(
    age_df, plot_k, "div_interaction_forest.png",
    sprintf("Age-specific original − Null %d", null_k),
    "Age"
  )
  plot_contrast_forest(
    series_df, plot_k, "series_interaction_forest.png",
    sprintf("Series-specific original − Null %d", null_k),
    "Series"
  )
  plot_contrast_forest(
    conn_df, plot_k, "connectome_configuration_interaction_forest.png",
    sprintf("Connectome-configuration-specific original − Null %d", null_k),
    "Connectome configuration"
  )

  summary_tbl <- build_summary_table(fixed_df, omnibus_df, age_df, series_df, conn_df)
  plot_summary_table(summary_tbl, plot_k, null_k)

  invisible(list(
    fixed = fixed_df, omnibus = omnibus_df,
    age = age_df, series = series_df, connectome = conn_df,
    summary = summary_tbl
  ))
}

# --- Forest & summary plots --------------------------------------------------

plot_contrast_forest <- function(contrast_df, plots_dir, filename, title, y_label) {
  plot_df <- contrast_df |>
    filter(!is.na(estimate)) |>
    mutate(
      experiment_label = factor(experiment_label, levels = unname(TASK_LABELS[ALL_TASKS])),
      level = factor(level, levels = unique(level)),
      significant = p.value < 0.05
    )
  if (nrow(plot_df) == 0) return(invisible(NULL))

  p <- ggplot(plot_df, aes(x = estimate, y = level, color = significant)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2, orientation = "y") +
    geom_point(size = 2) +
    facet_wrap(~experiment_label, ncol = 1, scales = "free_y") +
    scale_color_manual(
      values = c("FALSE" = "steelblue", "TRUE" = "firebrick"),
      labels = c("FALSE" = "p >= 0.05", "TRUE" = "p < 0.05"),
      name = NULL
    ) +
    labs(
      title = title,
      subtitle = "Direct emmeans contrast: original minus null (logit scale, Holm-adjusted)",
      x = "Logit-scale estimate (original − null)",
      y = y_label
    ) +
    theme_bw() +
    theme(panel.grid.minor = element_blank())

  ggsave(file.path(plots_dir, filename), p, width = 10, height = 12, dpi = 300, bg = "white")
}

build_summary_table <- function(fixed_df, omnibus_df, age_df, series_df, conn_df) {
  network <- fixed_df |>
    transmute(
      experiment, experiment_label,
      network_beta = estimate,
      network_se = std.error,
      network_p = p.value
    )

  omni_wide <- omnibus_df |>
    select(experiment, term_removed, p.value) |>
    pivot_wider(names_from = term_removed, values_from = p.value)

  n_sig <- function(ctr) {
    ctr |>
      group_by(experiment) |>
      summarise(n = sum(p.value < 0.05, na.rm = TRUE), .groups = "drop")
  }

  age_sig <- n_sig(age_df) |> rename(n_sig_age = n)
  series_sig <- n_sig(series_df) |> rename(n_sig_series = n)
  conn_sig <- n_sig(conn_df) |> rename(n_sig_connectome = n)

  network |>
    left_join(omni_wide, by = "experiment") |>
    left_join(age_sig, by = "experiment") |>
    left_join(series_sig, by = "experiment") |>
    left_join(conn_sig, by = "experiment") |>
    mutate(
      experiment_label = factor(experiment_label, levels = unname(TASK_LABELS[ALL_TASKS])),
      network_effect = sprintf("%.2f (%.2f)%s", network_beta, network_se, sig_stars(network_p)),
      age_omni = paste0(format_p(`network:Age`), sig_stars(`network:Age`)),
      series_omni = paste0(format_p(`network:Series`), sig_stars(`network:Series`)),
      connectome_omni = paste0(format_p(`network:connectome`), sig_stars(`network:connectome`)),
      sig_contrasts = sprintf(
        "%d Age / %d Series / %d connectome",
        n_sig_age, n_sig_series, n_sig_connectome
      ),
      p_network = network_p,
      p_age = `network:Age`,
      p_series = `network:Series`,
      p_connectome = `network:connectome`
    ) |>
    arrange(experiment_label)
}

plot_summary_table <- function(summary_tbl, plots_dir, null_k) {
  body <- summary_tbl |>
    mutate(
      row = as.character(experiment_label),
      col_network = network_effect,
      col_age = age_omni,
      col_series = series_omni,
      col_conn = connectome_omni,
      col_contrasts = sig_contrasts
    )

  value_cols <- c("col_network", "col_age", "col_series", "col_conn", "col_contrasts")
  col_headers <- c(
    "Network (orig−null)", "Age omnibus", "Series omnibus",
    "Connectome omnibus", "Sig. contrasts"
  )

  table_long <- bind_rows(lapply(seq_len(nrow(body)), function(i) {
    tibble(
      row = body$row[i],
      col = value_cols,
      text = as.character(unlist(body[i, value_cols])),
      p_value = c(
        body$p_network[i], body$p_age[i], body$p_series[i],
        body$p_connectome[i], NA_real_
      ),
      row_id = i,
      has_contrast = body$n_sig_age[i] + body$n_sig_series[i] + body$n_sig_connectome[i] > 0
    )
  })) |>
    mutate(
      col_id = match(col, value_cols),
      fill_color = ifelse(
        col == "col_contrasts",
        ifelse(has_contrast, "#e3f2fd", "#ffffff"),
        sig_bg_color(p_value)
      )
    )

  # Header row + experiment name column via y labels is awkward in geom_tile;
  # include experiment as first text column by expanding.
  header <- tibble(
    row = "Experiment",
    col_id = 0:(length(value_cols)),
    text = c("Experiment", col_headers),
    row_id = 0L,
    fill_color = "#ececec",
    is_header = TRUE
  )

  body_tiles <- bind_rows(
    lapply(seq_len(nrow(body)), function(i) {
      tibble(
        row = body$row[i],
        col_id = 0L,
        text = body$row[i],
        row_id = i,
        fill_color = "#ffffff",
        is_header = FALSE
      )
    }),
    table_long |>
      mutate(is_header = FALSE) |>
      select(row, col_id, text, row_id, fill_color, is_header)
  )

  plot_data <- bind_rows(header, body_tiles)

  p <- ggplot(plot_data, aes(x = col_id, y = -row_id)) +
    geom_tile(aes(fill = fill_color), color = "grey70", linewidth = 0.3) +
    geom_text(aes(label = text, fontface = ifelse(is_header, "bold", "plain")), size = 2.6) +
    scale_fill_identity() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(
      title = sprintf("Summary: original vs Null %d (across connectome configurations)", null_k),
      subtitle = "Network = logit coef original − null; omnibus = LRT p-values for interactions",
      caption = "Green: p < .05 (*), p < .01 (**), p < .001 (***). Blue: ≥1 significant contrast."
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0),
      plot.subtitle = element_text(size = 8, hjust = 0),
      plot.caption = element_text(size = 7, hjust = 0),
      plot.margin = margin(12, 12, 12, 12)
    )

  ggsave(
    file.path(plots_dir, "summary_table.png"),
    p,
    width = 14,
    height = 0.55 * (nrow(body) + 2) + 1.5,
    dpi = 300,
    bg = "white"
  )
}

# --- Main --------------------------------------------------------------------

main <- function() {
  cat("Loading connectome CSVs...\n")
  df <- load_all_connectomes()
  cat(sprintf(
    "  %d rows | connectomes: %s | ages (model): %s\n",
    nrow(df),
    paste(unique(df$connectome), collapse = ", "),
    paste(levels(df$Age), collapse = ", ")
  ))

  dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

  cat("\nWriting age profiles...\n")
  write_age_profiles(df, PLOTS_DIR, OUTPUT_DIR)

  for (k in 1:3) {
    run_null_variant_analysis(df, k, PLOTS_DIR, OUTPUT_DIR)
  }

  cat("\nDone. Outputs under:\n  ", OUT_ROOT, "\n", sep = "")
}

if (sys.nframe() == 0) {
  main()
}
