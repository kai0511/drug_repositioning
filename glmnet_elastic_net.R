library(doMC)
library(glmnet)
registerDoMC(cores = 4)

# format results from Elastic Net with given alpha
optimal.var <- function(alpha, train_X, train_y, n_fold){
    
    glmnet.obj <- cv.glmnet(
        x = train_X,
        y = train_y, 
        nfolds = n_fold,
        alpha = alpha,
        family = 'binomial',     
        type.measure="deviance", 
        standardize = TRUE, 
        parallel = TRUE)
    return(list(glmnet=glmnet.obj, deviance=min(glmnet.obj$cvm)))
}

for(n in seq(3)){
    
    setwd('/exeh/exe3/zhaok/GE_result/')
    file_train = 'SCZ_indication_orig_drug_expr_train_part%s.csv' % n
    file_test = 'SCZ_indication_orig_drug_expr_test_part%s.csv' % n
    file_result = 'predicted_result_SCZ_indication_orig_drug_expr.csv'
    
    pheno_train <- read.csv(file_train,header=F)
    pheno_test <- read.csv(file_test,header=F)
    
    drug.name <- pheno_test[[1]]
    train_X <- as.matrix(pheno_train[,c(-1,-2)])
    train_y <- pheno_train[[2]]
    test_X <- as.matrix(pheno_test[,c(-1,-2)])
    actual.y <- pheno_test[[2]]

    var.list <- lapply(alpha.vec, optimal.var, train_X = train_X, train_y = train_y, n_fold = 3)
    min.pos <- which.min(unlist(lapply(var.list, function(e) e$deviance)))
    print(paste0("[optimal parameters] alpha: ", min.pos/10, ', corresponding deviance: ', var.list[[min.pos]]$deviance, ', corresponding lambda: ', var.list[[min.pos]]$glmnet$lambda.min,'.'), quote=FALSE)

    predicted.y <- predict(var.list[[min.pos]]$glmnet, test_X, s="lambda.min", type='response')

    result <- data.frame(drug.name, actual.y, predicted.y)
    write.table(result, file_result, append=TRUE , quote=FALSE, sep=',', row.names=FALSE, col.names=FALSE)
}