require(h5)

getdfFromH5 <- function(i, object, f){
    # get data frame from the i-th block of h5 object
    d <- as.data.frame(f[paste0(object, "/block", i, "_values")][])
    colnames(d) <- f[paste0(object, "/block", i, "_items")][]
    return(d)
}

loadhdf5data <- function(loc, objectName) {
    # obtain data frame named object from h5 object
    f <- h5file(loc)
    nblocks <- h5attr(f[objectName], "nblocks")

    df <- do.call(cbind, 
                  lapply(seq(nblocks)-1, getdfFromH5, object = objectName, f = f))
    h5close(f)
    return(df)
}

# pheno <- loadhdf5data(loc, objectName)
