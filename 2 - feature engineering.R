############################################################################
### load libraries
############################################################################
library(caret)
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

library(randomForest)
library(glmnet)
library(party)

library(dplyr)
library(ggplot2)

############################################################################
### feature selection with glmnet (did not help much)
############################################################################
# glmnet
# note: glmnet in caret is unable to let glmnet choose lambda range
x <- model.matrix(target ~., data = train[in.train, ])[, -1]
y <- train[in.train, ]$target

# find best lambda at alpha = 0.5
glmnet.cv <- cv.glmnet(x = x, y = y, 
                       alpha = 0.5, family = 'multinomial', 
                       nfold = 5, parallel = T)

glmnet.cv$lambda.min  # 0.0002909891
glmnet.cv$lambda.1se  # 0.0008886387


# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

# use lambda found from above in glmnet.grid to use caret.glmnet
glmnet.grid <- expand.grid(alpha = (1:5) * 0.2, 
                           lambda = (1:5) * 0.002)

glmnet.fit <- train(target ~., data = train[in.train, ], method = 'glmnet', 
                    metric = 'LogLoss', maximize = F, tuneGrid = glmnet.grid, trControl = ctrl)

# LogLoss was used to select the optimal model using  the smallest value.
# The final values used for the model were alpha = 0.2 and lambda = 0.002. 

# examine coefficients
coef(glmnet.fit$finalModel, glmnet.fit$bestTune$.lambda)
coef(glmnet.fit$finalModel, 0.002)  # doesn't seem like we can exclude any features


# find best lambda at alpha = 1 
glmnet.grid <- expand.grid(alpha = 1, 
                           lambda = (1:5) * 0.002)

glmnet.fit <- train(target ~., data = train[in.train, ], method = 'glmnet', 
                    metric = 'LogLoss', maximize = F, tuneGrid = glmnet.grid, trControl = ctrl)

coef(glmnet.fit$finalModel, 0.002)  # looks like a dead end

############################################################################
### rf and gbm to examine top features (cforest was unusable)
############################################################################
# small rf
rf.fit <- randomForest(target ~., data = train[in.train, ], 
                       mtry = 9, nodesize = 5, ntree = 500, 
                       keep.forest = T, importance = T)

# rf variable importance
rf.imp <- importance(rf.fit, scale = T)
varImpPlot(rf.fit, n.var = 20)  # top 20 most impt variables

# extract 20 most important rf variables
rf.imp <- as.data.frame(importance(rf.fit, type = 1))
rf.imp$Vars <- row.names(rf.imp)
rf.20 <- rf.imp[order(-rf.imp$MeanDecreaseAccuracy),][1:20,]$Vars


# small cforest (unusable)
cf.fit <- cforest(target ~., data = train[in.train, ], 
                  control = cforest_unbiased(mtry = 9, ntree = 200))

# see importance of CF variables (could not run)
cf.imp <- varimp(cf.fit, conditional = T, threshold = 0.8)


# small gbm
gbm.grid <- expand.grid(interaction.depth = 10,
                        n.trees = 50,
                        shrinkage = 0.01)

gbm.fit <- train(target ~., data = train[in.train, ], 
                 method = 'gbm', distribution = 'multinomial', 
                 metric = 'LogLoss', maximize = F, 
                 tuneGrid = gbm.grid, trControl = ctrl,
                 n.minobsinnode = 4, bag.fraction = 0.9)

# gbm variable importance
varImp(gbm.fit, scale = T)

# extract 20 most important GBM variables
gbm.imp <- data.frame(varImp(gbm.fit)$importance)
gbm.imp$Vars <- row.names(gbm.imp)
gbm.20 <- gbm.imp[order(-gbm.imp$Overall),][1:20,]$Vars

# combine top features identified by rf and gbm
top.feats <- unique(c(gbm.20, rf.20))

############################################################################
### testing ground using a small sample set
############################################################################
# take small sample for testing
set.seed(100)
feat <- train %>%
  sample_frac(0.1)

### create sum of rows, variance of rows, and number of columns filled
addAggFeatures <- function(data) {
  
  # add new features
  mutate(data, feat_sum = as.integer(rowSums(data[, 1:93])),  # count sum of features by row
         feat_var = as.integer(apply(data[, 1:93], 1, var)),  # variance of features by row
         feat_filled = as.integer(rowSums(data[, 1:93] != 0))  # count no. of non-empty features
  )
}

# add new features to feat with function
feat <- addAggFeatures(train)

# plot new variables
ggplot(data = feat, aes(x = target, y = feat_sum, col = target)) + 
  geom_boxplot()
ggplot(data = feat, aes(x = target, y = feat_var, col = target)) + 
  geom_boxplot() +
  scale_y_continuous(limits = c(0, 12))
ggplot(data = feat, aes(x = target, y = feat_filled, col = target)) + 
  geom_boxplot()


### for each row, "normalize" features by dividing by sum (not useful in LB score)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_n_', i, ' <- feat[, i] / feat$feat_sum')))
}


### create +, -, *, / features using top 20 features
# select top 20 features
feat.20 <- train %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

# create +, -, *, / features
for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
} 

### create factor for 1 when feat > 0 and 0 when feat = 0 (works poorly--do not use)
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_flag_', i, ' <- ifelse(feat[, i] == 0, 0, 1)')))
}

### features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_mean_', i, ' <- feat[, i] - mean(feat[, i])')))
}

feat <- feat[-c(1:93)]
str(feat)

### features after normalization
for (i in 1:93) {
  eval(parse(text = paste0('feat$feat_norm_', i, ' <- (feat[, i] - mean(feat[, i]))/sd(feat[, 1])')))
}

feat <- feat[-c(1:93)]
str(feat)

############################################################################
### add features to entire train data
############################################################################
### add agg features
train <- addAggFeatures(train)

### select top 20 features to add op features
feat.20 <- train %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}  # multiplication

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}  # division

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}  # addition

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
}  # subtraction

# keep only created features
feat.20 <- (feat.20[-c(1:21)])


### extra clean up for division variables (due to errors from dividing 0 or dividing by 0)
str(feat.20[, 190: 220])
sapply(feat.20, function(x) sum(is.na(x)))

# cleaning up the Nan values from dividing 0
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

feat.20[is.nan(feat.20)] <- 0

# cleaning up the Inf values from dividing by 0
feat.20[mapply(is.infinite, feat.20)] <- 0

# check the division variables
str(feat.20[, 190: 220])

### features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('train$feat_mean_', i, ' <- train[, i] - mean(train[, i])')))
}

train <- train[-c(1:93)]

### features after normalization
for (i in 1:93) {
  eval(parse(text = paste0('train$feat_norm_', i, ' <- (train[, i] - mean(train[, i]))/sd(train[, i])')))
}

### number of filled features
train$feat_filled <- as.integer(rowSums(train[, 1:93] != 0))  # number of filled variables
train$feat_filled <- (train$feat_filled - mean(train$feat_filled))/sd(train$feat_filled)  # normalize

############################################################################
### add features to entire test data
############################################################################
### add agg features
test <- addAggFeatures(test)

### select top 20 features to add op features
feat.20 <- test %>%
  select(c(feat_11, feat_60, feat_34, feat_90, feat_14,
           feat_15, feat_26, feat_40, feat_86, feat_75,
           feat_36, feat_42, feat_39, feat_69, feat_68,
           feat_67, feat_62, feat_25, feat_9, feat_24, 
           target))

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, '_x_', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] * feat.20[, j]')))
  }
}  # multiplication

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'div', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] / feat.20[, j]')))
  }
}  # division

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'plus', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] + feat.20[, j]')))
  }
}  # addition

for (i in 1:19) {
  for (j in (i + 1) : 20) {
    var.x <- colnames(feat.20)[i]
    var.y <- colnames(feat.20)[j]
    var.new <- paste0(var.x, 'min', var.y)
    eval(parse(text = paste0('feat.20$', var.new, ' <- feat.20[, i] - feat.20[, j]')))
  }
}  # subtraction

# keep only created features
feat.20 <- (feat.20[-c(1:21)])


### extra clean up for division variables (due to errors from dividing 0 or dividing by 0)
str(feat.20[, 190: 220])
sapply(feat.20, function(x) sum(is.na(x)))

# cleaning up the Nan values from dividing 0
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

feat.20[is.nan(feat.20)] <- 0

# cleaning up the Inf values from dividing by 0
feat.20[mapply(is.infinite, feat.20)] <- 0

# check the division variables
str(feat.20[, 190: 220])

### features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_mean_', i, ' <- test[, i] - mean(test[, i])')))
}

test <- test[-c(1:93)]

### features after normalization
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_norm_', i, ' <- (test[, i] - mean(test[, i]))/sd(test[, i])')))
}

### number of filled features
test$feat_filled <- as.integer(rowSums(test[, 1:93] != 0))  # number of filled variables
test$feat_filled <- (test$feat_filled - mean(test$feat_filled))/sd(test$feat_filled)  # normalize