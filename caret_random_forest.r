library(caret)
library(Metrics)

setwd('/exeh/exe3/zhaok/GE_result/')

for(n in seq(3)){
   
    file_train = paste0('antidepression/Cmap_differential_expression_antidepression_train_part', n, '.csv')
    file_test = paste0('antidepression/Cmap_differential_expression_antidepression_test_part', n,'.csv')
    
    pheno_train <- read.csv(file_train, header=F)
    pheno_test <- read.csv(file_test, header=F)
    
    drug_name <- pheno_test[[1]]
    train_indication <- pheno_train[[2]]
    train_drug_expr <- pheno_train[,c(-1,-2)]
    test_indication <- pheno_test[[2]]
    test_drug_expr <- pheno_test[,c(-1,-2)]

    train_indication = as.factor(train_indication)
    levels(train_indication)[1] = "NOTindicated"
    levels(train_indication)[2] = "Indicated"
    
    fitControl <- trainControl(method = "repeatedcv",
                               number = 3,
                               classProbs =TRUE,
                               repeats = 3,
                               allowParallel=TRUE,
                               sampling = NULL,   ## "down", "up", "smote", or "rose"
                               summaryFunction = twoClassSummary)
    
    t1 = proc.time() 
    set.seed(0) 
    rf_fit <- train(y = train_indication, 
                    x = train_drug_expr,
                    method = "rf",
                    preProcess =c("center","scale"),
                    tuneLength = 10,
                    metric = "logLoss",
                    trControl = fitControl)
                    
    proc.time()-t1
    print(rf_fit)

    rf_pred = predict(rf_fit, newdata = test_drug_expr, type = "prob")
    pred.res = data.frame(drug=drug_name, 
                          indication=as.numeric(test_indication),
                          rf = rf_pred)
    print(paste0("logLoss: ", logLoss(as.numeric(test_indication), rf_pred[,2]), '.'), quote=False)
    pred.res = arrange(pred.res,desc(rf))
    # head(pred.res,n=100)
    print(varImp(rf_fit)) 
    save(rf_fit,rf_pred,pred.res, file="RF_scz_indications.Rdata")
    