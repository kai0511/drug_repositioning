library(randomForest)

setwd('/exeh/exe3/zhaok/GE_result/')

for(n in seq(3)){
   
    file_train = paste0('antidepression/Cmap_differential_expression_antidepression_train_part', n, '.csv')
    file_test = paste0('antidepression/Cmap_differential_expression_antidepression_test_part', n,'.csv')

    pheno_train <- read.csv(file_train, header = FALSE)
    pheno_test <- read.csv(file_test, header = FALSE)

    drug_name <- pheno_test[[1]]
    train_indication <- as.factor(pheno_train[[2]])
    train_drug_expr <- pheno_train[,c(-1,-2)]

    test_indication <- pheno_test[[2]]
    test_drug_expr <- pheno_test[,c(-1,-2)]
    
    # parameters to tune: number of tress, max depth, number of features used
    rf_fit = randomForest(train_drug_expr, train_indication, mtry = sqrt(dim(train_drug_expr)[2]))
    res = predict(rf_fit, test_drug_expr, type = 'prob')
    
    log_loss = logLoss(as.numeric(test_indication), iris.pred[,2])
    roc = auc(as.numeric(test_indication), iris.pred[,2])
    
    print(paste('Log Loss: ', log_loss, sep = ''))
    print(paste('ROC-AUC: ', roc, sep = ''))
}

