library(glmnet)
library(car)
library(caret)


blog <- 'path_to_blog\\blog_train_feat4.tsv'

# Read in the data
train <- read.table(file = blog,
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
train_model <- dplyr::select(train_data_clean, -1, -2,-3) # remove column 1 and two (index and author ID)
# class(train)
# dim(train)
# traceback()
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



x_cols <- setdiff(names(train_model), "label")
num_cols <- x_cols[sapply(train_model[x_cols], is.numeric)]


train_scaled <- train_model

# only scale numeric
train_scaled[num_cols] <- scale(train_model[num_cols])
sum(is.na(train_scaled))
table(train_scaled$label, useNA = "ifany")

cor_matrix <- cor(train_scaled[, num_cols], use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)

high_cor_names <- num_cols[high_cor]
print(high_cor_names)

features_with_high_correlation <- train_scaled[, high_cor_names, drop = FALSE] #train_scaled[high_cor]
train_scaled_reduced <- train_scaled[, !names(train_scaled) %in% high_cor_names]

# keep num cols in syn
num_cols <- setdiff(num_cols, high_cor_names)
stopifnot(all(num_cols %in% names(train_scaled_reduced)))
#train_scaled_reduced <- train_scaled[, -high_cor]


x <- model.matrix(label ~ ., train_scaled_reduced)[, -1]
y <- train_scaled_reduced$label
dim(x)
cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = 0.5, nfolds = 10)
selected_coefs <- coef(cv_fit, s = "lambda.min")


selected_features <- setdiff(rownames(selected_coefs)[selected_coefs[, 1] != 0], "(Intercept)")



# check
selected_features
setdiff(selected_features, names(train_scaled_reduced))  # should be empty

selected_features_clean <- gsub("`", "", selected_features)
setdiff(selected_features_clean, names(train_scaled_reduced))
selected_features_quoted <- paste0("`", selected_features_clean, "`")

formula_reduced <- as.formula(paste("label ~", paste(selected_features_quoted, collapse = " + ")))
final_model <- glm(formula_reduced, data = train_scaled_reduced, family = binomial)


vif(final_model)
bic <- BIC(final_model)
bic


coefs <- summary(final_model)$coefficients
results <- data.frame(
  feature    = rownames(coefs),
  estimate   = coefs[, "Estimate"],
  odds_ratio = exp(coefs[, "Estimate"]),
  p_raw      = coefs[, "Pr(>|z|)"]
)

results$p_adj <- p.adjust(results$p_raw, method = "BH")
results <- subset(results, feature != "(Intercept)")
results[order(results$p_adj), ]

#de fault walt test for sanity check
ci_walt <- confint.default(final_model)
bic <- BIC(final_model)
bic

# get the confidence interval
ci <- confint(final_model)
setdiff(results$feature, rownames(ci))  # should be empty
sum(is.na(results$ci_low))               # should be 0

results$ci_low  <- ci[results$feature, 1]
results$ci_high <- ci[results$feature, 2]
summary(ci)


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
    title = "Blog",
    x = "Coefficient estimate (log-odds)",
    y = NULL,
    color = "Associated with"
  ) +
  theme_minimal(base_size = 12)

