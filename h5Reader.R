loadhdf5data <- function(h5File, dataName) {
    require(h5) # available on CRAN

    f <- h5file(h5File)
    nblocks <- h5attr(f[dataName], "nblocks")

    df <- do.call(cbind, 
                    lapply(seq_len(nblocks)-1, 
                           function(i){
                                data <- as.data.frame(f[paste0(dataName, "/block", i, "_values")][])
                                colnames(data) <- f[paste0(dataName, "/block", i, "_items")][]
                           }))
    return df
  h5close(f)
}
