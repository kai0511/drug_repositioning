#***********************************
# test for enrichment of antidepressants or anxiolytics
# for each disorder, just change the abbreviation of each disorder
#***********************************

library("ICSNP")
library(metap)
library(mppa)
library(dplyr)

setwd("/exeh/exe3/zhaok/GE_result/")
scz = read.csv("/exeh/exe3/sohc/cmap/enrichment_existing_drug/ClinicalTrials.gov_mapped_toDrugBank-DO.csv", header=TRUE)

enrichment_t.test <- function(search.terms, pred_loc){
    ind.match.scz <- grep(paste(search.terms, collapse="|"), scz$doid_name, ignore.case=TRUE)
    scz.drugs <- unique(scz$drugbank_name[ind.match.scz])
    ## KEY: remove empty space first 
    scz.drugs <- as.character(scz.drugs[scz.drugs!=""])

    CMAP <- read.csv(pred_loc, header=FALSE) 
    colnames(CMAP) <- c("Drug", "Indicated", "Prob")
    CMAP = subset(CMAP, Indicated == 0)

    ## find drugs in our results that match with drugs in clinicalTrial.org
    ind.match.trial = grep(paste(scz.drugs,collapse="|"), CMAP$Drug, ignore.case=TRUE) 
    ind.notmatch.trial = setdiff(1:nrow(CMAP), ind.match.trial)

    # two-sample t test
    zval= CMAP$Prob
    fit2 = t.test(zval[ind.match.trial], zval[ind.notmatch.trial], alternative = "greater")
    return(fit2$p.value)
}

search.terms = list(schizophrenia = c("schizo","psychosis","psychotic","paranoid"),  # schizophrenia
                    bipolar = c("bipolar"),   # bipolar disorder
                    depression = c("depress"),  # depression
                    anxiety = c("anxiety","phobia","phobic","panic")) # anxiety 
                    
# search.terms= c("schizo","psychosis","psychotic","paranoid",
#                 "dementia","Alzheimer",
#                 "anxiety",
#                 "autis",
#                 "attention deficit",
#                 "bipolar",
#                 "depress",
#                 "phobia","phobic",
#                 "panic")  ##all psychiatric disorder

# prediction file location list
# pred_loc_list = list("depressionANDanxiety"="/keras_deep_learning_depressionANDanxiety_prediction.out", 
#                      "antidepression"="/keras_deep_learning_antidepression_prediction.out",
#                      "antipsycho"="/keras_deep_learning_antipsycho_prediction.out", 
#                      "scz"="/keras_deep_learning_scz_prediction.out")
                     
desease_name = c("depressionANDanxiety", 'antidepression', 'antipsycho', 'scz')
models = c('svm', 'rf', 'bt', 'glmnet')

for(d in desease_name){
    pred_loc = paste(d, "/glmnet_result.out", sep = "")
    res <- t(as.matrix(unlist(lapply(search.terms, enrichment_t.test, pred_loc = pred_loc))))
    rownames(res) = d
    write.table(res, 'enrichment_t.test/glmnet_pval.res', sep = ",", quote = FALSE, append = TRUE)
}
