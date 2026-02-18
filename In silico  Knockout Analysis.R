=============================================================
# scTenifoldKnk Virtual Knockout Analysis 
# =============================================================

library(Seurat)
library(scTenifoldKnk)
library(dplyr)
library(ggplot2)
library(patchwork)
library(pheatmap)
library(ggrepel)

# Set working environment
work_dir <- "C:/Users/xieru/Desktop/6PPQD/GEO10X"
setwd(work_dir)

# =============================================================
# Step 1: Load Seurat Object
# =============================================================
cat("========== Loading Seurat Object ==========\n")

if (!file.exists("merged_seurat_auto_annotated.rds")) {
  stop("Error: Input RDS file not found.")
}

merged_seurat <- readRDS("merged_seurat_auto_annotated.rds")
cat("Total Cells:", ncol(merged_seurat), "\n")
cat("Total Genes:", nrow(merged_seurat), "\n")

# =============================================================
# Step 2: Target Gene Validation
# =============================================================
cat("\n========== Validating Target Gene ==========\n")

# Standardize gene names to Uppercase
rownames(merged_seurat) <- toupper(rownames(merged_seurat))

if ("TP53" %in% rownames(merged_seurat)) {
  target_gene <- "TP53"
  cat("✓ Target gene identified:", target_gene, "\n")
} else if ("TRP53" %in% rownames(merged_seurat)) {
  target_gene <- "TRP53"
  cat("✓ Target gene identified (Mouse ortholog):", target_gene, "\n")
} else {
  stop("Critical Error: TP53/TRP53 not found in the dataset!")
}

# =============================================================
# Step 3: Feature Selection (HVGs)
# =============================================================
cat("\n========== Feature Selection (HVGs) ==========\n")

# Expand HVG count to 3000 to ensure regulatory network density
N_HVG <- 3000

if (length(VariableFeatures(merged_seurat)) == 0) {
  cat("Computing highly variable genes...\n")
  merged_seurat <- FindVariableFeatures(merged_seurat, nfeatures = N_HVG, verbose = FALSE)
}

hvgs <- VariableFeatures(merged_seurat)

# Force inclusion of TP53 in the feature list even if not highly variable
if (target_gene %in% hvgs) {
  selected_hvgs <- head(hvgs, N_HVG)
} else {
  cat("Note: Target gene not in top HVGs, manually adding to the list.\n")
  selected_hvgs <- c(head(hvgs, N_HVG - 1), target_gene)
}

# =============================================================
# Step 4: Global Virtual Knockout
# =============================================================
cat("\n========== Global TP53 Knockout Analysis ==========\n")

set.seed(2026)
# Sample 1000 cells for network stability (standard for TenifoldKnk)
n_cells_to_sample <- min(1000, ncol(merged_seurat))
global_cells <- sample(colnames(merged_seurat), n_cells_to_sample)
seurat_global <- subset(merged_seurat, cells = global_cells)

# Check expression sparsity
tp53_expr_count <- sum(GetAssayData(seurat_global, layer = "counts")[target_gene, ] > 0)
cat("Cells expressing", target_gene, "in global sample:", tp53_expr_count, "\n")

if (tp53_expr_count < 10) {
  warning("Low expression detected. Network inference may be unstable.")
}

# Construct the Matrix
expr_matrix_global <- as.matrix(GetAssayData(seurat_global[selected_hvgs,], layer = "counts"))

cat("Building Global Regulatory Network...\n")
perturbation_global <- scTenifoldKnk(
  countMatrix = expr_matrix_global,
  gKO = target_gene,
  qc = TRUE,
  nc_nNet = 10,
  nc_nCells = 500,
  nCores = parallel::detectCores()
)

write.csv(perturbation_global, "TP53_global_knockout_result.csv")

# =============================================================
# Step 5: Neuron-Specific Virtual Knockout
# =============================================================
cat("\n========== Neuron-Specific TP53 Knockout ==========\n")

# Use previously annotated cell types
if (!"cell_type_auto" %in% colnames(merged_seurat@meta.data)) {
  stop("Cell type annotations missing in metadata.")
}

neuron_cells <- WhichCells(merged_seurat, expression = cell_type_auto %in% c("Neurons", "Neuron", "Interneuron"))
cat("Total Neurons available:", length(neuron_cells), "\n")

if (length(neuron_cells) < 100) {
  stop("Insufficient neurons for robust network construction.")
}

n_neuron_sample <- min(1000, length(neuron_cells))
seurat_neuron <- subset(merged_seurat, cells = sample(neuron_cells, n_neuron_sample))
expr_matrix_neuron <- as.matrix(GetAssayData(seurat_neuron[selected_hvgs,], layer = "counts"))

cat("Building Neuron-Specific Regulatory Network...\n")
perturbation_neuron <- scTenifoldKnk(
  countMatrix = expr_matrix_neuron,
  gKO = target_gene,
  qc = TRUE,
  nc_nNet = 10,
  nc_nCells = min(300, ncol(expr_matrix_neuron)),
  nCores = parallel::detectCores()
)

write.csv(perturbation_neuron, "TP53_neuron_specific_knockout_result.csv")

# =============================================================
# Visualization Function
# =============================================================


generate_knockout_plots <- function(result_df, prefix) {
  cat("\nGenerating report for:", prefix, "\n")
  
  # Ensure consistent data frame structure
  df <- as.data.frame(result_df)
  if(!"gene" %in% colnames(df)) df$gene <- rownames(df)
  
  # Standardize metrics
  if (!"FC" %in% colnames(df)) df$FC <- if("delta" %in% colnames(df)) 2^(df$delta) else 1
  if (!"p.value" %in% colnames(df)) df$p.value <- if("zscore" %in% colnames(df)) 2 * pnorm(-abs(df$zscore)) else 1
  
  df <- df %>% filter(gene != target_gene) %>% mutate(log2FC = log2(FC))
  
  # 1. Barplot: Top 20 Perturbed Genes
  top_genes <- df %>% arrange(desc(abs(log2FC))) %>% head(20)
  
  p1 <- ggplot(top_genes, aes(x = reorder(gene, log2FC), y = log2FC, fill = log2FC > 0)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_manual(values = c("TRUE" = "#E74C3C", "FALSE" = "#3498DB")) +
    coord_flip() +
    labs(title = paste(prefix, "Top 20 Affected Genes"), subtitle = paste("KO Target:", target_gene),
         x = "Gene", y = "log2(Fold Change)") +
    theme_minimal() + theme(legend.position = "none")

  ggsave(paste0(prefix, "_Barplot.pdf"), p1, width = 8, height = 7)

  # 2. Volcano Plot
  p2 <- ggplot(df, aes(x = log2FC, y = -log10(p.value))) +
    geom_point(aes(color = p.value < 0.05), alpha = 0.5) +
    scale_color_manual(values = c("FALSE" = "grey70", "TRUE" = "#E74C3C")) +
    geom_text_repel(data = head(arrange(df, p.value), 10), aes(label = gene), size = 3) +
    labs(title = paste(prefix, "Perturbation Magnitude"), x = "log2(FC)", y = "-log10(P-value)") +
    theme_bw() + theme(legend.position = "none")

  ggsave(paste0(prefix, "_Volcano.pdf"), p2, width = 7, height = 6)
}

# =============================================================
# Run Visualization
# =============================================================
generate_knockout_plots(perturbation_global, "Global")
generate_knockout_plots(perturbation_neuron, "Neuron")

cat("\n=============================================================\n")
cat("scTenifoldKnk Analysis Complete. Reports saved to:", getwd(), "\n")
cat("=============================================================\n")
