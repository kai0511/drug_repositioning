
setwd('/exeh/exe3/zhaok/data/genePerturbation')

# read cmap gene expression data
load("/exeh/exe3/sohc/cmap/Cmap_differential_expression.Rdata")
cmap = as.data.frame(t(degList$tstat))
drugExprGeneID = names(cmap)

# read knockdown gene expression data
knockDown <- read.table('/exeh/exe3/sohc/L1000/consensi-knockdown.tsv', header=TRUE)
knockDownGeneID <- colnames(knockDown)
knockDownGeneID <- unname(sapply(knockDownGeneID, function(x) substring(x, 2)))
colnames(knockDown) <- knockDownGeneID

# read overexpression gene expression data
overExpression <- read.table('/exeh/exe3/sohc/L1000/consensi-overexpression.tsv', header=TRUE)
overExpressionID <- colnames(overExpression)
overExpressionID <- unname(sapply(overExpressionID, function(x) substring(x, 2)))
colnames(overExpression) <- overExpressionID

# find the intersection between cmap gene set and knockdown gene set
intersectGeneID <- intersect(drugExprGeneID, knockDownGeneID)
intersectGeneID1 <- intersect(drugExprGeneID, overExpressionID)

# confirm whether gene set is consensus between the two perturbations 
intersectGeneID == length(intersect(intersectGeneID1, intersectGeneID))

# save
write.table(overExpression[, intersectGeneID], 'L1000_consensi_overExpression.csv')
write.table(cmap[, intersectGeneID], 'cmap_differential_expression_knockdown.csv')
write.table(knockDown[, intersectGeneID], 'L1000_consensi_knockdown.csv')
