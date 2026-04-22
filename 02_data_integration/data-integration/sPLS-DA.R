library(dplyr)
library(tidyverse)
library(stringr)
library(mixOmics)
library(logr)
library(glue)
set.seed(29043)
###TODO: Implement loging 
###      Make an environment (?)
###      dd a description for the script and the readme file on git 
#tmp <- file.path("~\\Projects\\sosall\\results\\2022-01-31\\logger.log")

root = '/home/damir/Documents/Projects/sosall/'
setwd(root)
path_out = glue(file.path(root,'results/{Sys.Date()}'))
path_in = glue(file.path(root,'raw-data'))
# Open log
#lf <- log_open(tmp)

################################################################################
# 1.Loading the files and processing the input 
################################################################################
### 1. Read in the files 
### X.lasso and X.boruta character vectors containing geneIDs selected during feature selection process 
X.lasso <- read.csv(file.path(path_in,"Union_selected_features/feature_selection/lasso_union.csv"), 
                    header = 0, 
                    col.names = c('gene_id','index'))
X.boruta <- read.csv(file.path(path_in,"Union_selected_features/feature_selection/boruta.csv"),
                     header = 0, 
                     col.names = c('gene_id','index'))

###Y - questionnaire dataset containing clinical variables and rna varaibles already scaled 
Y <- read.csv(file.path(path_in,'IntegratedData_2022-09-06.txt'), 
              sep = '\t', 
              na.strings = 'NaN')

### metadata file with sample labels 
meta <- read.csv(file.path(root,'rna_metadata.csv'))


#### extract the gene ids from LASSO and Boruta feature selection methods
lasso.id <- as.vector(X.lasso$gene_id)
boruta.id <- as.vector(X.boruta$gene_id)
gene.ids <- c(lasso.id, boruta.id)


X <- Y[,colnames(Y) %in% lasso.id]
rownames(X) = Y$X
colnames(X) = gsub(".*__","",colnames(X))

meta$treatment <- as.factor(meta$treatment)
meta$names = toupper(meta$names)
rownames(meta) = meta$names
meta = meta[order(match(rownames(X), rownames(meta))),]

# set the rownames to the PID in the clinical dataset 
rownames(Y) = Y$X 
Y$X = NULL
Y =  Y[, !colnames(Y) %in% gene.ids]
Y$diagnosis_location = NULL

# convert all character column to numeric 
Y = Y %>% mutate_if(is.character, as.numeric)

setdiff(rownames(X),rownames(Y))

# order X and Y to have them in the same order 
Y = Y[order(match(rownames(X), rownames(Y))),]
X = X[order(match(rownames(X), rownames(Y))),]

################################################################################
# 2.Data Split 
################################################################################
### Split the data into train and test 
### 75% of the sample size
smp_size <- floor(0.80 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)

### Train of the RNAseq/Questionnaire data 
### X.rna - train data with RNA features 
### X.q - train data with questionnaire features 
X.rna.train  <- X[train_ind, ]
X.q.train <- Y[train_ind, ]

### Test set 
X.rna.test <- X[-train_ind, ]
X.q.test <- Y[-train_ind, ]

### Input for the mixomics should be in the form of lists containing the dataframes 
X.train <- list(RNA = X.rna.train,
                Clinical = X.q.train)
X.test <- list(RNA = X.rna.test, 
               Clinical = X.q.test)

### Labels for the training and testing  
Y.train <- meta[train_ind, 4]
Y.test <- meta[-train_ind, 4]
Y.train <- as.factor(Y.train)
Y.test <- as.factor(Y.test)

################################################################################
# 3.Model training
################################################################################
### Cross Validation
### Tuning to find the number of components to retain
diablo.sosall <- block.splsda(X.train, 
                              Y.train, 
                              ncomp = 10)
cv.diablo.sosall <- perf(diablo.sosall, 
                         validation = 'loo', 
                         progressBar = TRUE, 
                         auc = FALSE)
plot(cv.diablo.sosall)
### inspecting the plot it turns out the ncomp is the best at 2 
ncomp = 2

### specify the design matrix 
### Fully connected design, could also be a value in the range [-0,1]
design <- matrix(1, ncol = length(X.train), 
                 nrow = length(X.train),
                 dimnames = list(names(X.train), 
                                 names(X.train)))
diag(design) <- 0 

### CV to determine the number of features to keep in every dataset 
keepX.grid <- list(RNA = c(20, 25, 30, 50, 100 ),
                   Clinical = c(3, 5, 10, 12, 15, 20, 25, 30, 40, 45))
tune.diablo.sosall <- tune.block.splsda(X.train, 
                                        Y.train, 
                                        ncomp = 2,
                                        test.keepX = keepX.grid, 
                                        validation = 'Mfold',
                                        folds = 5, 
                                        nrepeat = 200, 
                                        max.iter = 1000, 
                                        design = design,
                                        dist = 'centroids.dist',
                                        progressBar = FALSE, 
                                        BPPARAM = BiocParallel::SnowParam(workers = parallel::detectCores()-1))
### store the variable
keep.features <- tune.diablo.sosall$choice.keepX

################################################################################
# 4.Model fitting and plot inspection
################################################################################
### fit the model with defined parameters 
diablo.sosall <- block.splsda(X.train, 
                              Y.train, 
                              ncomp = 2, 
                              keepX = keep.features, 
                              design = design)
### inspect the variables that were selected in every dataset 
selectVar(diablo.sosall, 
          block = 'RNA', 
          comp = 1)
selectVar(diablo.sosall, 
          block = 'Clinical', 
          comp = 1)

### set up the output file so that all plots are saved in one pdf 
ifelse(!dir.exists(file.path(path_out,'plots')), 
       dir.create(file.path(path_out,'plots'),
                  recursive = TRUE), 
       FALSE)
pdf(file.path(path_out,'plots/test.pdf'),
    width = 20,
    height = 20)
### why though? somehow should remove the loop but then how to save efficiently
for (i in 1:1) {
        set.seed(i)
        ### plot the final model and correlation between datasets 
        plotDiablo(diablo.sosall, 
                   ncomp = 1)
        ### plot the indiv data points for clustering observation 
        plotIndiv(diablo.sosall, 
                  ind.names = FALSE, 
                  legend = TRUE, 
                  title = 'SOSALL, DIABLO comp 1-2', 
                  block = 'weighted.average',
                  ellipse = F)
        
        plotArrow(diablo.sosall, 
                  ind.names = FALSE, 
                  legend = TRUE, 
                  title = 'SOSALL, DIABLO comp 1-2')
        
        plotVar(diablo.sosall, 
                var.names = TRUE, 
                style = 'graphics', 
                legend = TRUE,
                pch = c(5, 5), 
                cex = c(0.5,0.75),
                cutoff = 0.5, 
                col = c('darkorchid','blue'),
                title = 'SOSALL, DIABLO comp 1 - 2')
        
        circosPlot(diablo.sosall,
                   cutoff = 0.5, 
                   line = F,
                   color.blocks = c('darkorchid', 'lightgreen'),
                   color.cor = c("chocolate3","grey20"), 
                   size.labels = 1,
                   size.variables = 1.2)
        
        # ploting loadings of the variables 
        plotLoadings(diablo.sosall,  
                     comp = 1, 
                     block = 'RNA',
                     contrib = 'max', 
                     method = 'mean', 
                     ndisplay = 20, 
                     plot = TRUE,
                     title = 'RNA Loadings Component 1',
                     size.name = 1)
        plotLoadings(diablo.sosall, 
                     comp = 2,
                     block = 'RNA',
                     contrib = 'max',
                     method = 'mean', 
                     ndisplay = 20, 
                     plot = TRUE,
                     title = 'RNA Loadings Component 2',
                     size.name = 1)
        
        plotLoadings(diablo.sosall,  
                     comp = 1, 
                     block = 'Clinical',
                     contrib = 'max',
                     method = 'mean',
                     ndisplay = 20,
                     plot = TRUE, 
                     study = 'global',
                     size.name = 0.7,
                     size.title = 1,
                     title = 'Clinical Loadings Component 1',
                     border = F)
        
        
        plotLoadings(diablo.sosall,  
                     comp = 2, 
                     block = 'Clinical',
                     contrib = 'max', 
                     method = 'mean', 
                     ndisplay = 20, 
                     plot = TRUE)
        cimDiablo(diablo.sosall, 
                  color.blocks = c('darkorchid', 'lightgreen'),
                  comp = c(1,2), 
                  margin=c(8,10), 
                  legend.position = "right", 
                  size.legend = 0)
}
dev.off()

#cimDiablo(diablo.sosall, 
#          color.blocks = c('darkorchid', 'lightgreen'),
#          comp = c(1,2), 
#          margin=c(8,10), 
#          legend.position = "right", 
#          size.legend = 0)

################################################################################
# 5.Model performance on the unseen data 
################################################################################
perf.diablo.sosall <- perf(diablo.sosall, 
                           validaiton = 'Mfold', 
                           folds = 5,
                           nrepeat = 100, 
                           dist = 'centroids.dist')
perf.diablo.sosall$MajorityVote.error.rate
perf.diablo.sosall$WeightedVote.error.rate

### plot the performance 
pdf(file.path(path_out,'plots/performance.pdf'),
    width = 15,
    height = 15)
for (i in 1:1) {
        set.seed(i)

        auc.diablo.sosall <- auroc(diablo.sosall, 
                                   roc.block = 'RNA',
                                   roc.comp = 2, 
                                   print = TRUE)
        auc.diablo.sosall <- auroc(diablo.sosall,
                                   roc.block = 'Clinical',
                                   roc.comp = 2, 
                                   print = TRUE)
}
dev.off()

### Predictions on the unseen data 
predict.diablo.sosall <- predict(diablo.sosall, 
                                 newdata = X.test)
confusion.mat <- get.confusion_matrix(truth = Y.test, 
                                      predicted = predict.diablo.sosall$WeightedVote$centroids.dist[,2])
View(confusion.mat)

# save the data in the session 
save.image(file = file.path(path_out,'workspace.rds'))
