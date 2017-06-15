loadhdf5data <- function(h5File, dataFrameName) {
    # 
    require(h5) # available on CRAN

    f <- h5file(h5File)
    nblocks <- h5attr(f[dataFrameName], "nblocks")

    df <- do.call(cbind, 
                    lapply(seq_len(nblocks)-1, 
                           function(i){
                                data <- as.data.frame(f[paste0(dataFrameName, "/block", i, "_values")][])
                                colnames(data) <- f[paste0(dataFrameName, "/block", i, "_items")][]
                           }))
    h5close(f)
    return df
    
}
