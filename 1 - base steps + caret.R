############################################################################
### load libraries
############################################################################
library(caret)
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

############################################################################
### load data sets and create train/validation split
############################################################################
# load training data
train <- read.csv("train.csv", header = T)
train$id <- NULL  # remove ID
set.seed(668)
train <- train[sample(nrow(train)), ]  # shuffle data

# load testing data
test <- read.csv("test.csv", header = T)
test.id <- test$id
test$id <- NULL  # remove ID

# partition into training and validation set
set.seed(668)
in.train <- createDataPartition(y = train$target, p = 0.80, list = F)  # use this to train model
in.train <- in.train[1:49506]  # this makes it a vector

############################################################################
### useful functions for checking logloss and creating submissions
############################################################################
# implement log loss function and test it with test case from here: 
# http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error
# note: the function only takes in matrices
LogLoss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -sum(actual*log(predicted))/nrow(actual)
}

actual <- matrix(data = c(0, 1, 0, 1, 0, 0, 0, 0, 1), nrow = 3)
pred <- matrix(data = c(0.2, 0.7, 0.1, 0.6, 0.2, 0.2, 0.6, 0.1, 0.3), 
               nrow = 3, byrow = T)

LogLoss(actual, pred)  # this should be 0.6904911 if the function is working correction


# create function to compute logloss on validation set using ground truth
checkLogLoss <- function(model, data) {
  
  # LogLoss Function
  LogLoss <- function(actual, predicted, eps = 1e-15) {
    predicted <- pmax(pmin(predicted, 1 - eps), eps)
    -sum(actual*log(predicted))/nrow(actual)
  }
  
  # create dummy predictions and compare with fitted model
  pred <- as.matrix(predict(model, newdata = data, type = 'prob'))
  dummy.fit <- dummyVars(~ target, data = data, levelsOnly = T)  
  truth <- predict(dummy.fit, newdata = data)  # predict ground truth using validation set
  LogLoss(truth, pred)
}


# create custom logloss summary function for use with caret cross validation
LogLossSummary <- function(data, lev = NULL, model = NULL) {
  
  # this is slightly different from function above as above function leads to errors
  LogLoss <- function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] <- eps
    pred[pred > 1 - eps] <- 1 - eps
    -sum(actual * log(pred)) / nrow(pred)
  }
  if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
  pred <- data[, 'pred']
  obs <- data[, 'obs']
  is.na <- is.na(pred)
  pred <- pred[!is.na]
  obs <- obs[!is.na]
  data <- data[!is.na, ]
  class <- levels(obs)
  
  if (length(obs) + length(pred) == 0) {
    out <- rep(NA, 2)
  } else {
    probs <- data[, class]
    actual <- model.matrix(~ obs - 1)
    out <- LogLoss(actual = actual, pred = probs)
  }
  names(out) <- c('LogLoss')
  
  if (any(is.nan(out))) out[is.nan(out)] <- NA
  
  out
}


# create function to create submissions
# note: file should be a string
submit <- function(model, data, file) {
  
  # create predictions and write out to csv.file
  pred <- predict(model, newdata = data, type = 'prob', na.action = NULL)
  submission <- data.frame(id = test.id, pred)
  write.csv(submission, file = file, row.names = F)
}

############################################################################
### random forest
############################################################################
# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

rf.grid <- expand.grid(mtry = c(6, 9, 12))

rf.fit <- train(target ~., data = train[in.train, ], method = 'rf', 
                metric = 'LogLoss', maximize = F,
                tuneGrid = rf.grid, trControl = ctrl, ntree = 500)

# create predictions
rf.pred <- as.matrix(predict(rf.fit, newdata = train[-in.train, ], type = 'prob'))

# compute log.loss  
checkLogLoss(rf.fit, train[-in.train, ])  # logloss = 0.56234

############################################################################
### gbm
############################################################################
# set params for fit control
ctrl <- trainControl(method = 'cv', number = 5, verboseIter = T, classProbs = T, 
                     summaryFunction = LogLossSummary)

gbm.grid <- expand.grid(interaction.depth = 10,
                        n.trees = (2:100) * 50,
                        shrinkage = 0.005)

gbm.fit <- train(target ~., data = train[in.train, ], 
                    method = 'gbm', distribution = 'multinomial', 
                    metric = 'LogLoss', maximize = F, 
                    tuneGrid = gbm.grid, trControl = ctrl,
                    n.minobsinnode = 4, bag.fraction = 0.9)

checkLogLoss(gbm.fit, train[-in.train, ])  # log.loss = 0.509993


# good practice to close connections when done
showConnections()
closeAllConnections()
showConnections()
