# library(doMC)
library(Metrics)
library(glmnet)
# registerDoMC(cores = 4)

# format results from Elastic Net with given alpha
optimal.var <- function(alpha, train_X, train_y, n_fold){
    
    glmnet.obj <- cv.glmnet(
        x = train_X,
        y = train_y, 
        nfolds = n_fold,
        alpha = alpha,
        family = 'binomial',     
        type.measure = "deviance", 
        standardize = TRUE, 
        parallel = TRUE, 
        keep = TRUE)
    return(list(glmnet=glmnet.obj, deviance=min(glmnet.obj$cvm)))
}

nfold = 3
alpha.vec = seq(0,1,0.1)
setwd('/exeh/exe3/zhaok/data/genePerturbation')
cmap <- read.table('cmap_differential_expression_perturbation.csv', header=TRUE)
ATC.N05A <- read.csv('/exeh/exe3/zhaok/data/N05A.txt', header=FALSE)

# obtain indications for N05A drugs
search.term <- paste(tolower(as.vector(ATC.N05A[[1]])), collapse = '|')
drug.name <- rownames(cmap)
idx <- grep(search.term, drug.name, ignore.case=TRUE)
Indication <- rep(0, length(drug.name))
Indication[idx] = 1

# add a new column of Indication to cmap
cmap$Indication <- Indication

# adjust the order of columns
col.name <- names(cmap)
new.order <- c('Indication', col.name[1:length(col.name)-1])
indicated.cmap <- cmap[, new.order]

# generate foldid for cv.glmnent function
# folds <- createFolds(factor(indicated.cmap$Indication), k = nfold, list = FALSE)

# obtain X and y from cmap data
X <- as.matrix(indicated.cmap[, -1])
y <- indicated.cmap$Indication

# run cv.glmnet
var.list <- lapply(alpha.vec, optimal.var, train_X = X, train_y = y, n_fold = nfold)
min.pos <- which.min(unlist(lapply(var.list, function(e) e$deviance)))

# print best parameters from training
print(paste0("[optimal parameters] alpha: ", min.pos/10, ', corresponding deviance: ', var.list[[min.pos]]$deviance, ', corresponding lambda: ', var.list[[min.pos]]$glmnet$lambda.min,'.'), quote=FALSE)

# compute logLoss
cv.glmnet.fit <- var.list[[min.pos]]$glmnet
fited.y <- cv.glmnet.fit$fit.preval[, min.pos]
folds <- cv.glmnet.fit$foldid

for(i in seq(1, max(folds))){
    y.pred <- fited.y[folds == i]
    y.real <- y[folds == i]
    print(paste0('log Loss for fold ', i, ' :', logLoss(y.real, y.pred), '.'), quote = FALSE)
}

# make predictions on knockdown and overexpression data
knockdown <- read.table('L1000_consensi_knockdown.csv', header=TRUE)
overexpression <- read.table('L1000_consensi_knockdown.csv', header=TRUE)
predicted.knockdown <- predict(cv.glmnet.fit, as.matrix(knockdown[,-1]), s="lambda.min", type='response')
predicted.overexpression <- predict(cv.glmnet.fit, as.matrix(overexpression[,-1]), s="lambda.min", type='response')

# save results
knockdown.res <- data.frame(perturbationID = knockdown[, 1], pred = predicted.knockdown[,1])
overexpression.res <- data.frame(perturbationID = overexpression[, 1], pred = predicted.overexpression[,1])
write.table(knockdown.res, 'knockdown_res.csv', append=TRUE , quote=FALSE, sep=',', row.names=FALSE, col.names = FALSE)
write.table(overexpression.res, 'overexpression_res.csv', append=TRUE , quote=FALSE, sep=',', row.names=FALSE, col.names = FALSE)
