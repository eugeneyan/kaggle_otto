############################################################################
### load libraries
############################################################################
library(xgboost)
library(methods)
library(caret)

############################################################################
### load train data and create matrices for xgb
############################################################################
train <- read.csv("train.csv", header = T)
train$id <- NULL  # remove ID
set.seed(668)
train <- train[sample(nrow(train)), ]  # shuffle data

# create target vector
train.y <- train$target
train.y <- gsub('Class_','', train.y)
train.y <- as.integer(train.y) - 1  #xgboost take features in [0, number of classes)

# create matrix of original features for train.x
train.x <- train
train.x$target <- NULL
train.x <- as.matrix(train.x)
train.x <- matrix(data = as.numeric(train.x), nrow = nrow(train.x), ncol = ncol(train.x))


############################################################################
### create useful functions (check log loss specific to xgb; requires matrix creation)
############################################################################
### create function to compute logloss on test set in one step
checkLogLoss2 <- function(model, xgbdata, traindata) {
  
  # LogLoss Function
  LogLoss <- function(actual, predicted, eps = 1e-15) {
    predicted <- pmax(pmin(predicted, 1 - eps), eps)
    -sum(actual*log(predicted))/nrow(actual)
  }
  
  # create predictions and dummy predictions and compare with fitted model
  pred <- predict(xgb.fit, newdata = xgbdata)
  pred <- t(matrix(pred, nrow = 9, ncol = length(pred)/9))  # prediction based on fitted model
  dummy.fit <- dummyVars(~ target, data = traindata, levelsOnly = T)
  truth <- predict(dummy.fit, newdata = traindata)  # ground truth
  LogLoss(truth, pred)
}


############################################################################
### try creating a small xgb model
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.1,
                 'gamma' = 1,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

# run cross validation
xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x[in.train, ], label = train.y[in.train], 
                nfold = 5, nrounds = 250)

# check best iteration
which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))

# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                     label = train.y[in.train], nrounds = 250)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  # log.loss = 0.4627816

# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                   label = train.y, nrounds = 250)


############################################################################
### check feature importance to create interaction features
# note: while this worked with the original features, it seemed to increase
# error rate when using scaled features. Thus, it was not used eventually.
############################################################################
# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = 100)

# check feature importance
xgb.importance(feature_names = names(train), model = xgb.fit)


############################################################################
### xgb using original + aggregated features
# aggregated features being row sum, row var, and no. of cols filled
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.005,
                 'gamma' = 1,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

# run cross validation
xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x[in.train, ], label = train.y[in.train], 
                     nfold = 5, nrounds = 10000)

# check best iteration
cv.min <- min(xgb.fit.cv$test.mlogloss.mean)
cv.min.rounds <- which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))  
# min = 0.469457 at nrounds 7483 7484

cv.rounds <- round(mean(which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))))


# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = cv.rounds)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  
# log.loss = 0.4517306 (improvement of 0.0006185)


# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                   label = train.y, nrounds = cv.rounds)
# LB score = 0.44085 

############################################################################
### xgb using original features (after mean-standardization) 
# gamma = 0.5, min_child = 4
############################################################################
# Set necessary parameter
xg.param <- list("objective" = "multi:softprob",
                 'eval_metric' = "mlogloss",
                 'num_class' = 9,
                 'eta' = 0.005,
                 'gamma' = 0.5,
                 'max.depth' = 10,
                 'min_child_weight' = 4,
                 'subsample' = 0.9,
                 'colsample_bytree' = 0.8,
                 'nthread' = 3)

xgb.fit.cv <- xgb.cv(param = xg.param, data = train.x, label = train.y, 
                     nfold = 5, nrounds = 10000)

# check best iteration
cv.min <- min(xgb.fit.cv$test.mlogloss.mean)
cv.min.rounds <- which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))  
# min = 0.448826 at nrounds 7563

plot(xgb.fit.cv$test.mlogloss.mean[7000:8000])
cv.rounds <- round(mean(which(xgb.fit.cv$test.mlogloss.mean == min(xgb.fit.cv$test.mlogloss.mean))))  

# fit model on training set
xgb.fit <- xgboost(param = xg.param, data = train.x[in.train, ], 
                   label = train.y[in.train], nrounds = cv.rounds)

# check log loss on validation set
checkLogLoss2(xgb.fit, train.x[-in.train, ], train[-in.train, ])  
# log.loss = 0.4489771 

# fit model on full training data
xgb.fit <- xgboost(param = xg.param, data = train.x, 
                        label = train.y, nrounds = cv.rounds)
# LB score = 0.43609


############################################################################
### create predictions on test data
############################################################################
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL

# create matrix of original features for test.x
test <- as.matrix(test)
test <- matrix(data = as.numeric(test), nrow = nrow(test), ncol = ncol(test))

# create predictions on test data 
# using original + aggregated features
xgb.pred <- predict(xgb.fit, test)
xgb.pred <- t(matrix(xgb.pred, nrow = 9, ncol = length(xgb.pred)/9))
xgb.pred <- data.frame(1:nrow(xgb.pred), xgb.pred)
names(xgb.pred) <- c('id', paste0('Class_',1:9))
write.csv(xgb.pred, file='xgb.pred.csv', quote=FALSE, row.names=FALSE)


# create predictions on test data 
# using original (after mean-standardization) features
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL

# features for difference from mean
for (i in 1:93) {
  eval(parse(text = paste0('test$feat_mean_', i, ' <- test[, i] - mean(test[, i])')))
}
test <- test[-c(1:93)]

# create matrix of original features for test.x
test <- as.matrix(test)
test <- matrix(data = as.numeric(test), nrow = nrow(test), ncol = ncol(test))

# create predictions on test data 
# using original + aggregated features
xgb.pred <- predict(xgb.fit, test)
xgb.pred <- t(matrix(xgb.pred, nrow = 9, ncol = length(xgb.pred)/9))
xgb.pred <- data.frame(1:nrow(xgb.pred), xgb.pred)
names(xgb.pred) <- c('id', paste0('Class_',1:9))
write.csv(xgb.pred, file='xgb.pred2.csv', quote=FALSE, row.names=FALSE)

