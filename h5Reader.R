# ----------------------------------------------------------------------------
# read Pandas dataframe from h5 using h5 R package
# ----------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------
# read Pandas dataframe from h5 using rhdf5
# ---------------------------------------------------------------------------------
require(rhdf5)

loadDataFrameFromH5 <- function(loc){
    h5.structure <- h5ls(loc)
    # Find all data nodes, values are stored in *_values and corresponding column
    # titles in *_items
    data_nodes <- grep("_values", h5.structure$name)
    name_nodes <- grep("_items", h5.structure$name)

    data_paths = paste(h5.structure$group[data_nodes], h5.structure$name[data_nodes], sep = "/")
    name_paths = paste(h5.structure$group[name_nodes], h5.structure$name[name_nodes], sep = "/")

    columns = list()
    for (idx in seq(data_paths)) {
      data <- as.data.frame(t(h5read(loc, data_paths[idx])))
      names <- t(h5read(loc, name_paths[idx]))
      colnames(data) <- names
      columns <- append(columns, data)
    }
    data.frame(columns)
}
