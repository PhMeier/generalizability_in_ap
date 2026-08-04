library(glmnet)
library(car)
library(caret)


pan14 <-'path_to_pan14_training_file\\train_pan_14_stratified_prepro_full_features_4.tsv'

# Read in the data
train <- read.table(file = pan14,
                    sep = '\t', header = TRUE, check.names = FALSE, quote = "",
                    fill = TRUE,
                    comment.char = "")
aggregate_features <- c("PUNCT", "BRACKETS", "QUOTATION", "SYMBOL", "ADJECTIVE",
                        "COMMON_NOUN", "PROPER_NOUN", "ADVERB", "VERB", "WH",
                        "DETERMINER", "PRONOUN", "CONJUNCTION_ADPOSITION",
                        "SPACE", "AUX_SPECIAL", "NS", "SN", "NN_",
                        # Dependency-relation aggregates
                        "modifier_relations", "adverbial_dependencies",
                        'Contrast', 'Enablement', 'Evaluation', 'textual-organization', 'Condition', 'Joint',
                        'Background', 'Temporal', 'Attribution', 'Manner-Means', 'Topic-Comment', 'Summary',
                        'Comparison', 'Elaboration', 'Explanation', 'Topic-Change', 'same-unit', 'Cause')


# Drop before scaling/pruning/glmnet -- i.e., at the very start of the pipeline
train_data_clean <- train[, !names(train) %in% aggregate_features]


table(train$gender)
# remove author id and index
train_model <- dplyr::select(train_data_clean, -1, -2) # remove column 1 and two (index and author ID)

# label prüfen, convert the label to a category
# Label differs on the dataset either '"label"' or normal label
train_model$`"label"`
train_model$label <- as.factor(train_model$label) #`"label"`)



# return everything in A that is not in B (so that is not label) and use these
# as predictor variables
x_cols <- setdiff(names(train_model), "label")

# get the numerical columns
is_num <- vapply(train_model, is.numeric, logical(1))

num_cols <- names(train_model)[is_num] # list of numeric predictor columns
num_cols <- setdiff(num_cols, "label") # removes label from num_cols
which(is.na(names(train_model)) | names(train_model) == "")


# Check if the feature columns are numeric so that scaling is possible
#num_cols <- x_cols[sapply(train_model[x_cols], is.numeric)]



zero_var <- sapply(train_model[num_cols], function(x) {
  sd(x, na.rm = TRUE) == 0
})
#zero_var
# remove the zero variance columns
zero_var_cols <- names(zero_var[zero_var])

train_model <- train_model[, !names(train_model) %in% zero_var_cols]
num_cols <- setdiff(num_cols, zero_var_cols)
stopifnot(all(num_cols %in% names(train_model)))
cor_matrix <- cor(train_model[, num_cols], use = "pairwise.complete.obs")



x_cols <- setdiff(names(train_model), "label")
num_cols <- x_cols[sapply(train_model[x_cols], is.numeric)]


train_scaled <- train_model


train_scaled[num_cols] <- scale(train_model[num_cols])
sum(is.na(train_scaled))
table(train_scaled$label, useNA = "ifany")


cor_matrix <- cor(train_scaled[, num_cols], use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)


high_cor_names <- num_cols[high_cor]
print(high_cor_names)

features_with_high_correlation <- train_scaled[, high_cor_names, drop = FALSE]
train_scaled_reduced <- train_scaled[, !names(train_scaled) %in% high_cor_names]


num_cols <- setdiff(num_cols, high_cor_names)
stopifnot(all(num_cols %in% names(train_scaled_reduced)))


features_with_high_correlation <- train_scaled[high_cor]

train_scaled_reduced <- train_scaled[, -high_cor]


x <- model.matrix(label ~ ., train_scaled_reduced)[, -1]
y <- train_scaled_reduced$label


cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = 0.5, nfolds = 10)
selected_coefs <- coef(cv_fit, s = "lambda.min")
selected_coefs

coefs <- summary(cv_fit)$coefficients
results <- data.frame(
  feature    = rownames(coefs),
  estimate   = coefs[, "Estimate"],
  odds_ratio = exp(coefs[, "Estimate"]),
  p_raw      = coefs[, "Pr(>|z|)"]
)
results$feature <- gsub("`", "", results$feature)<
results$p_adj <- p.adjust(results$p_raw, method = "BH")
results <- subset(results, feature != "(Intercept)")
results[order(results$p_adj), ]








selected_features <- setdiff(rownames(selected_coefs)[selected_coefs[, 1] != 0], "(Intercept)")
selected_features <- gsub("`", "", selected_features)       # strip any auto-added backticks
stopifnot(all(selected_features %in% names(train_scaled_reduced)))

selected_features_quoted <- paste0("`", selected_features, "`")  # re-add deliberately, once
formula_reduced <- as.formula(paste("label ~", paste(selected_features_quoted, collapse = " + ")))
print(formula_reduced)

final_model <- glm(formula_reduced, data = train_scaled_reduced, family = binomial)

aliased_terms <- names(coef(final_model))[is.na(coef(final_model))]
aliased_terms <- setdiff(aliased_terms, "(Intercept)")
aliased_terms
vif(final_model)
bic <- BIC(final_model)
bic

fitted_probs <- fitted(final_model)
sum(fitted_probs < 1e-6)
sum(fitted_probs > 1 - 1e-6)
summary(fitted_probs)

sum(fitted_probs < 1e-6)
sum(fitted_probs > 1 - 1e-6)


sum(fitted_probs == 0)
sum(fitted_probs == 1)

coefs <- summary(final_model)$coefficients
coefs[order(-abs(coefs[, "Estimate"])), ][1:10, ]

final_model$converged  # TRUE?
final_model$boundary   # FALSE?

c("Topic-Change", "NS", "SN", "NN_", "modifier_relations", "adverbial_dependencies") %in% names(train_scaled_reduced)

coefs <- summary(final_model)$coefficients
results <- data.frame(
  feature    = rownames(coefs),
  estimate   = coefs[, "Estimate"],
  odds_ratio = exp(coefs[, "Estimate"]),
  p_raw      = coefs[, "Pr(>|z|)"]
)
results$feature <- gsub("`", "", results$feature)
results$p_adj <- p.adjust(results$p_raw, method = "BH")
results <- subset(results, feature != "(Intercept)")
results[order(results$p_adj), ]

ci_walt <- confint.default(final_model_reduced)
bic <- BIC(final_model)
bic

ci <- confint(final_model)

rownames(ci) <- gsub("`", "", rownames(ci))
stopifnot(all(results$feature %in% rownames(ci)))
setdiff(results$feature, rownames(ci))

results$ci_low  <- ci[results$feature, 1]
results$ci_high <- ci[results$feature, 2]
stopifnot(!any(is.na(results$ci_low)))

results$direction <- ifelse(results$estimate > 0, "Female", "Male")  # adjust labels to your coding

plot_data <- subset(results, p_adj < 0.05)
plot_data <- plot_data[order(plot_data$estimate), ]
plot_data$feature <- factor(plot_data$feature, levels = plot_data$feature)


ggplot(plot_data, aes(x = estimate, y = feature, color = direction)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = c("Male" = "steelblue", "Female" = "indianred")) +
  labs(
    title = "PAN 14",
    x = "Coefficient estimate (log-odds)",
    y = NULL,
    color = "Associated with"
  ) +
  theme_minimal(base_size = 12)

