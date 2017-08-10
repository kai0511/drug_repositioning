library(doMC)
library(glmnet)
# library(Metrics)
registerDoMC(cores = 4)

optimal.var <- function(train_X, train_y, sample_weights, n_fold, alpha){
    
    glmnet.obj <- cv.glmnet(
        x = train_X,
        y = train_y, 
        weights = sample_weights
        nfolds = n_fold,
        alpha = alpha,
        family = 'binomial',     
        type.measure="deviance", 
        standardize = TRUE, 
        parallel = TRUE)
    return(list(glmnet=glmnet.obj, deviance=min(glmnet.obj$cvm)))
}

weightedLogLoss <- function(actual, predicted, sample_weights){
    score <- -(actual * log(predicted) + (1 - actual) * log(1 -predicted))
    score <- score * sample_weights
    score[actual == predicted] <- 0
    score[is.nan(score)] <- Inf
    return(mean(score))
}

alpha.vec = seq(0,1,0.1)
setwd('/exeh/exe3/zhaok/GE_result/')
coefs_df <- data.frame(p1 = numeric(), p2 = numeric(), p3 = numeric())
dir_vec = c('antidepression', 'antipsycho', 'depressionANDanxiety', 'scz')
gene_name = read.csv('/exeh/exe3/zhaok/GE_result/gene_name.csv', header=F, stringsAsFactors = F)
gene_name_vec = gene_name[[1]]
 
for(d in dir_vec){
    print(paste0('Begining compute ', d, '...'))
    
    coefs_df = data.frame(name=gene_name_vec)
    file_result = paste0('glmnet_', d, '_result.out')
    system(paste0('rm -f ', file_result))
    
    for(n in seq(3)){
        file_train = paste0(d,'/Cmap_differential_expression_', d, '_train_part', n, '.csv')
        file_test = paste0(d,'/Cmap_differential_expression_', d, '_test_part', n,'.csv')

        pheno_train <- read.csv(file_train,header=F)
        pheno_test <- read.csv(file_test,header=F)
        
        drug.name <- pheno_test[[1]]
        train_X <- as.matrix(pheno_train[,c(-1,-2)])
        train_y <- pheno_train[[2]]
        test_X <- as.matrix(pheno_test[,c(-1,-2)])
        actual.y <- pheno_test[[2]]
        
        # compute sample weights 
        obs_num <- dim(train_y)[1] + dim(actual.y)[1]
        pos_num <- sum(c(train_y, actual.y))
        neg_num <- obs_num - pos_num
        pos_weight <- obs_num / (2*pos_num)
        neg_weight <- obs_num / (2*neg_num)
        train_weights <- rep(neg_weight, dim(train_y)[1])
        train_weights[train_y == 1] <- pos_weight
        test_weights <- rep(neg_weight, dim(actual.y)[1])
        test_weights[actual.y == 1] <- pos_weight
        
        var.list <- lapply(alpha.vec, optimal.var, train_X = train_X, train_y = train_y, sample_weights = train_weights, n_fold = 3)
        min.pos <- which.min(unlist(lapply(var.list, function(e) e$deviance)))
        print(paste0("[optimal parameters] alpha: ", min.pos/10, ', corresponding deviance: ', var.list[[min.pos]]$deviance, ', corresponding lambda: ', var.list[[min.pos]]$glmnet$lambda.min,'.'), quote=FALSE)
        
        # obtain non-zero coefs and write them into a csv file
        coef_df <- as.data.frame(as.matrix(coef(var.list[[min.pos]]$glmnet, s='lambda.min')))
        coefs_df[[n+1]] <- coef_df[[1]]
        
        # make prediction using the best model
        predicted.y <- predict(var.list[[min.pos]]$glmnet, test_X, s="lambda.min", type='response')
        
        # compute negLoss
        print(paste0("Corresponding weighted logLoss: ", weightedLogLoss(actual.y, predicted.y, test_weights), '.'), quote=F)
 
        result <- data.frame(drug.name, predicted.y, actual.y)
        write.table(result, file_result, append=TRUE , quote=FALSE, sep=',', row.names=FALSE, col.names=FALSE)
    }
                                           
    dest <- paste0('/exeh/exe3/zhaok/GE_result/coefs/', d, '_coef.csv')
    colnames(coefs_df) <- c('gene_name', 'p1', 'p2', 'p3')
    coefs_df$avg_coef <- (coefs_df$p1 + coefs_df$p2 + coefs_df$p3)/3
    write.table(coefs_df, dest, sep = ",", col.names = F)
}
