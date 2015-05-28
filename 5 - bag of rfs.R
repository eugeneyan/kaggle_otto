############################################################################
### load libraries
############################################################################
library(caret)
library(doParallel)
detectCores()
registerDoParallel(detectCores() - 1) 
getDoParWorkers()

# randomForest
library(randomForest)

############################################################################
### tune rf params
############################################################################
ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, verboseIter = T,
                     summaryFunction = LogLossSummary2)

rf.grid <- expand.grid(mtry = c(10, 12))

rf.fit <- train(target ~., data = train[in.train, ], method = 'rf', 
                metric = 'LogLoss', maximize = F,
                tuneGrid = rf.grid, trControl = ctrl, ntree = 180,
                nodesize = 8)

rf.fit  # 100 trees, mtry = 10, LL = 0.6382415
rf.fit  # 180 trees, mtry = 12, LL = 0.6202338
rf.fit  # 250 trees, mtry = 12, LL = 0.6093700
rf.fit  # 250 trees, mtry = 12, nodesize = 8, LL = 0.6082160
rf.fit  # 250 trees, mtry = 12, nodesize = 8, LL = 0.605719
rf.fit  # 500 trees, mtry = 12, LL = 0.5947927

############################################################################
### bagging RFs
############################################################################
### create a single rf model
# params
ntree <- 150
mtry <- 12
nodesize <- 4

# create truth
dummy.fit <- dummyVars(~ target, data = train[-in.train, ], levelsOnly = T)
truth <- predict(dummy.fit, newdata = train[-in.train, ])

# create rf.fit
rf.fit <- randomForest(target ~., data = train[in.train, ], ntree = ntree, nodesize = nodesize, mtry = mtry)
rf.pred <- as.matrix(predict(rf.fit, newdata = train[-in.train, ], type = 'prob'))

# check LogLoss for 1 RF
LogLoss(truth, rf.pred)  # LL = 0.5859206


### creating bag of 10 RFs
# set all rf.pred to 0
rf.pred[, 1:9] <- 0

# params
ntree <- 150
mtry <- 12
nodesize <- 4

# create bag of 10 RFs
for (i in 1:10) {
  print(paste0('Building RF: ', i))
  rf.fit <- randomForest(target ~., data = train[in.train, ], ntree = ntree, 
                         nodesize = nodesize, mtry = mtry)
  rf.pred <- (rf.pred + as.matrix(predict(rf.fit, newdata = train[-in.train, ], type = 'prob')))
  rf.pred_avg <- rf.pred/i
  # print log.loss of current bag
  print(paste0('LogLoss for ', i, ' RFs: ', (LogLoss(truth, rf.pred_avg))))
}

# ntree = 150, mtry = 12, nodesize = 4  (bagging 10 rf models reduces validation logloss)
# [1] "Building RF: 1"
# [1] "LogLoss for 1 RFs: 0.592349543550637"
# [1] "Building RF: 2"
# [1] "LogLoss for 2 RFs: 0.573437551141468"
# [1] "Building RF: 3"
# [1] "LogLoss for 3 RFs: 0.56990495203981"
# [1] "Building RF: 4"
# [1] "LogLoss for 4 RFs: 0.565196657543647"
# [1] "Building RF: 5"
# [1] "LogLoss for 5 RFs: 0.564934777643734"
# [1] "Building RF: 6"
# [1] "LogLoss for 6 RFs: 0.564983847844716"
# [1] "Building RF: 7"
# [1] "LogLoss for 7 RFs: 0.560068400139644"
# [1] "Building RF: 8"
# [1] "LogLoss for 8 RFs: 0.560029089264538"
# [1] "Building RF: 9"
# [1] "LogLoss for 9 RFs: 0.55975913557159"
# [1] "Building RF: 10"
# [1] "LogLoss for 10 RFs: 0.559649291973404"

############################################
### create predictions on test data
############################################
# clear predictions
rf.pred[, 1:9] <- 0

# params
ntree <- 150
mtry <- 12
nodesize <- 4

# create bag of 10 RFs
for (i in 1:10) {
  print(paste0('Building RF: ', i))
  rf.fit <- randomForest(target ~., data = train, ntree = ntree, 
                         nodesize = nodesize, mtry = mtry)
  rf.pred <- (rf.pred + as.matrix(predict(rf.fit, newdata = test, type = 'prob')))
  rf.pred.avg <- rf.pred/i
}

# write to .csv
submission <- data.frame(id = test.id, rf.pred.avg)
write.csv(submission, file = 'rf.bag.pred.csv', row.names = F)
