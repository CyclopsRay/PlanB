library(zinbwave)
library(BiocParallel)
library(SingleCellExperiment)

# Use a for loop to read all the file under the path /users/lyuyang/lyy/PlanB/schiebinger2019/raw_data_origin/raw_data end in .tsv
# and store them in a list called files
# files <- list.files(path = "/users/lyuyang/lyy/PlanB/schiebinger2019/raw_data_origin/raw_data", pattern = "*.tsv", full.names = TRUE)
input_dir <- "/users/lyuyang/lyy/PlanB/schiebinger2019/raw_data_origin/raw_data/"
pattern <- ".tsv"
for (i in 1:16) {
    # concatenate the all the file along the 0th dimension and transpose the matrix
    # Cell by gene matrix
    file_name <- paste0(input_dir, "original_", i, pattern)
    if (i == 1) {
        t0 = read.delim(file_name)
        # Add one column to the matrix to store the timepoint information, which should be the same as the i
        t0$timepoint <- i
    }else{
        t1 = read.delim(file_name)
        # Add one column to the matrix to store the timepoint information, which should be the same as the i
        t1$timepoint <- i
        t0 = rbind(t0, t1)
    }
    print(file_name)
}
    print("finish reading")
    dense_mat <- t(as.matrix(t0))
    sce <- SingleCellExperiment(assays = list(counts = dense_mat))
    counts <- assay(sce, "counts")
    counts_rounded <- round(counts)
    counts_rounded[counts_rounded < 0] <- 0
    assay(sce, "counts") <- counts_rounded
    colData(sce)$timepoint <- t0$timepoint
    print("Start to zinbwave")
    sce_zinb <- zinbwave(sce, K = 2, epsilon = 1000, BPPARAM=MulticoreParam(24), X="~timepoint")
    print("finish zinbwave, start to reduceDim")
    W <- reducedDim(sce_zinb)
    print("finish reduceDim")
    # Specify the full path to the directory where you want to save the file
    output_dir <- "/users/lyuyang/lyy/PlanB/schiebinger2019/large_test/zinb-latent-exponent/"

    # Save the output as a TSV file in the specified directory
    write.table(W, file = paste0(output_dir, paste0("zinb_wave_latent_k2", ".tsv")), sep = "\t", quote = FALSE, row.names = TRUE)
