# ==================== 1. Load Required Libraries ===================
# Install packages if not already installed (uncomment if needed):
# install.packages(c("randomForest", "ggplot2", "ggthemes", "pROC", "Boruta"))

library(randomForest)   
library(ggplot2)        
library(ggthemes)       
library(pROC)          
library(Boruta)        

# Note: This script uses rfcv() from randomForest for recursive feature elimination with cross-validation.

# ==================== 2. Set Working Directory and Load Data ====================
setwd("Your own")

# Read the dataset (assumes first column is 'Group', rest are features)
data <- read.csv("your own.csv", header = TRUE)

# Inspect data dimensions to verify successful loading
cat("Data dimensions (rows × columns):", dim(data), "\n")

# ==================== 3. Data Preprocessing ====================
# Ensure the response variable 'Group' is a factor (required for classification)
# R is case-sensitive—confirm column name matches exactly (e.g., "Group", not "group")
if (!"Group" %in% names(data)) {
  stop("Column 'Group' not found in the dataset. Please check column names.")
}
data$Group <- factor(data$Group)

# Re-check dimensions after conversion (should be unchanged)
cat("Data dimensions after preprocessing:", dim(data), "\n")

# Optional: Check class distribution
print(table(data$Group))

# ==================== 4. Train Random Forest Model ====================
set.seed(999)  # Ensures reproducibility of results

# Fit Random Forest model with default parameters (500 trees)
# Importance = TRUE enables computation of variable importance measures
fit <- randomForest(Group ~ ., data = data, importance = TRUE, ntree = 500)

# Print model summary (OOB error, confusion matrix, etc.)
print(fit)

# Plot out-of-bag (OOB) error rate vs. number of trees
# Helps assess convergence and potential overfitting
plot(fit, main = "Random Forest: OOB Error Rate vs. Number of Trees")
legend("topright", colnames(fit$err.rate), col = 1:ncol(fit$err.rate), lty = 1, cex = 0.8)

# ==================== 5. Recursive Feature Elimination with Cross-Validation (rfcv) ====================
# Perform RF-based cross-validated feature selection
# At each step, removes a fraction of least important variables and evaluates CV error

set.seed(647)  # For reproducible CV folds
res <- rfcv(
  trainx = data[, !names(data) == "Group"],  # All features except 'Group'
  trainy = data$Group,                       # Response variable
  cv.fold = 5,                               # 5-fold cross-validation
  step = 0.75                                # Retain 75% of features at each step (i.e., remove 25%)
)

# Output key results
cat("Number of variables at each step:\n")
print(res$n.var)

cat("Corresponding cross-validation error rates:\n")
print(res$error.cv)

# ==================== 6. Plot Cross-Validation Error vs. Number of Features ====================
# Base R plot (clean and informative)
plot(
  res$n.var, res$error.cv,
  type = "o",
  pch = 16,
  lwd = 2,
  col = "steelblue",
  xlab = "Number of Variables",
  ylab = "Cross-Validation Error Rate",
  main = "RF-CV: Error Rate vs. Number of Features",
  xlim = rev(range(res$n.var)),  # Reverse x-axis to show decreasing features left-to-right
  ylim = c(0, max(res$error.cv) * 1.1)
)

# Enhance readability
grid(lty = 2, col = "gray80")
abline(h = min(res$error.cv), col = "red", lty = 2)  # Highlight minimum error

# Add annotation for minimum error
min_err <- min(res$error.cv)
min_nvar <- res$n.var[which.min(res$error.cv)]
text(min_nvar, min_err + 0.02, 
     labels = paste("Min Error =", round(min_err, 4)), 
     pos = 4, col = "red", font = 2)

legend("topright", legend = "CV Error", col = "steelblue", lwd = 2, bty = "n")

# ==================== 7. Extract and Visualize Top Important Features ====================
# Extract variable importance from the full RF model
imp <- importance(fit)
importance_df <- as.data.frame(imp)
importance_df$Gene <- rownames(importance_df)

# Sort by MeanDecreaseGini (measure of node impurity reduction)
# Alternative: sort by MeanDecreaseAccuracy for prediction accuracy impact
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

# Select top 8 genes (adjust number as needed)
n_top <- 8
top_genes <- head(importance_df, n_top)

cat("Top", n_top, "most important genes:\n")
print(top_genes[, c("Gene", "MeanDecreaseGini")])

# ==================== 8. Create Publication-Quality Bar Plot ====================
# Define color gradient (purple theme)
color_low  <- "#8B008B"  # Dark magenta
color_high <- "#D8BFD8"  # Light thistle

# Build horizontal bar plot using ggplot2
gene_plot <- ggplot(top_genes, aes(x = reorder(Gene, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(aes(fill = MeanDecreaseGini), width = 0.7) +
  geom_text(
    aes(label = round(MeanDecreaseGini, 2)),
    hjust = -0.1,            # Place label to the right of bars
    size = 3.5,
    color = "black"
  ) +
  scale_fill_gradient(low = color_low, high = color_high, guide = "none") +
  coord_flip() +  # Flip axes for better gene name readability
  labs(
    title = paste("Top", n_top, "Most Important Genes (Random Forest)"),
    x = "Gene",
    y = "Mean Decrease in Gini Impurity"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text.y = element_text(size = 11),
    axis.text.x = element_text(size = 10),
    panel.grid.major.y = element_blank(),  # Remove vertical grid lines
    panel.grid.minor = element_blank()
  ) +
  ylim(0, max(top_genes$MeanDecreaseGini) * 1.15)  # Extra space for labels

# Display the plot
print(gene_plot)

# Save as high-quality PDF (vector format, ideal for publications)
ggsave(
  filename = "top_genes_importance.pdf",
  plot = gene_plot,
  device = "pdf",
  width = 9,      # inches
  height = 6,
  dpi = 300
)

# Also save as PNG if needed
# ggsave("top_genes_importance.png", plot = gene_plot, width = 9, height = 6, dpi = 300)

cat("Gene importance plot saved as 'top_genes_importance.pdf'\n")
