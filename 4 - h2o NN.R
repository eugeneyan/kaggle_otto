######################################################################
## download and install H2O
######################################################################

# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("rjson" %in% rownames(installed.packages()))) { install.packages("rjson") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o-dev/master/1179/R")))

######################################################################
## launch H2O
######################################################################
## Load h2o R module
library(h2o)

## Launch h2o on localhost, using all cores
h2oServer <- h2o.init(nthreads = -1)

## Point to directory where the Kaggle data is
# dir <- paste0(path.expand("~"), "/h2o-kaggle/otto/")

## For Spark/Hadoop/YARN/Standalone operation on a cluster, follow instructions on http://h2o.ai/download/
## Then connect to any cluster node from R

#h2oServer = h2o.init(ip="mr-0xd1",port=53322)
#dir <- "hdfs://mr-0xd6/users/arno/h2o-kaggle/otto/"

######################################################################
## load data sets and create train/validation split
######################################################################

train.hex <- h2o.importFile(paste0("C:/Users/IBM_ADMIN/Desktop/Eugene's/Otto/train.norm.csv"), destination_frame="train.hex")
test.hex <- h2o.importFile(paste0("C:/Users/IBM_ADMIN/Desktop/Eugene's/Otto/test.csv"), destination_frame="test.hex")
dim(train.hex)

## Split into 80/20 Train/Validation
train_holdout.hex <- h2o.assign(train.hex[in.train,], "train_holdout.hex")
dim(train_holdout.hex)
valid_holdout.hex <- h2o.assign(train.hex[in.valid,], "valid_holdout.hex")
dim(valid_holdout.hex)

######################################################################
## parameter tuning with random search
######################################################################
models <- c()
for (i in 1:10) {
  rand_activation <- c("RectifierWithDropout", "MaxoutWithDropout")[sample(1:2,1)]
  rand_numlayers <- sample(2:3,1)
  rand_hidden <- c(sample(93:800, rand_numlayers, T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-2)
  rand_hidden_dropout <- c(runif(rand_numlayers, 0, 0.5))
  rand_input_dropout <- runif(1, 0, 0.2)
  rand_rho <- runif(1, 0.9, 0.999)
  rand_epsilon <- runif(1, 1e-10, 1e-4)
  rand_rate <- runif(1, 0.005, 0.02)
  rand_rate_decay <- runif(1, 0, 0.66)
  rand_momentum <- runif(1, 0, 0.5)
  rand_momentum_ramp <- runif(1, 1e-7, 1e-5)
  dlmodel <- h2o.deeplearning(x = 2:94, y = 1, 
                              training_frame = train_holdout.hex, 
                              validation_frame = valid_holdout.hex,
                              rho = rand_rho, epsilon = rand_epsilon, 
                              rate = rand_rate,
                              rate_decay = rand_rate_decay, 
                              nesterov_accelerated_gradient = T, 
                              momentum_start = rand_momentum,
                              momentum_ramp = rand_momentum_ramp,
                              epochs = 10, 
                              activation = rand_activation, 
                              hidden = rand_hidden, 
                              l1 = rand_l1, 
                              l2 = rand_l2,
                              input_dropout_ratio = rand_input_dropout, 
                              hidden_dropout_ratios = rand_hidden_dropout, 
                              epochs = 20
)
  models <- c(models, dlmodel)
}
Sys.time()

## Find the best model (lowest logloss on the validation holdout set)
best_err <- 1e3 
for (i in 1:length(models)) {
  err <- h2o.logloss( h2o.performance(models[[i]], valid_holdout.hex))
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}

best_model
best_params <- best_model@parameters
best_params$activation  # RectifierWithDropout
best_params$hidden  # 605 222
best_params$l2 # 0.002619916
best_params$input_dropout_ratio # 0.004822871
best_params$hidden_dropout_ratios  # 0.4341811 0.0763970
best_params$rate  # 0.01916945
best_params$rate_decay  # 0.5107246
best_params$momentum_start  # 0.08898998
best_params$momentum_ramp  # 3.146529e-06

# check logloss on validation set
valid_perf <- h2o.performance(best_model, valid_holdout.hex)
h2o.confusionMatrix(valid_perf)
h2o.logloss(valid_perf)  # 0.5348277

######################################################################
## h2o NN with best found params
######################################################################
h2o.fit <- h2o.deeplearning(x = 2:94, y = 1, 
                            training_frame = train_holdout.hex, 
                            validation_frame = valid_holdout.hex, 
                            adaptive_rate = T,
                            nesterov_accelerated_gradient = T, 
                            momentum_start = 0.33,
                            momentum_ramp = 5e-6,
                            epochs = 50, 
                            activation = "RectifierWithDropout", 
                            hidden = c(512, 512, 388),  
                            hidden_dropout_ratios = c(0.25, 0.45, 0.33),
                            rate = 0.009,
                            rate_decay = 0.4,
                            score_training_samples = 0,
                            score_validation_samples = 0)

# check log loss on validation set
valid_perf <- h2o.performance(h2o.fit, valid_holdout.hex)
h2o.confusionMatrix(valid_perf)
h2o.logloss(valid_perf)  # 0.5283119

View(h2o.fit@model$scoring_history)

######################################################################
## create predictions
######################################################################

## Predictions: label + 9 per-class probabilities
pred <- predict(h2o.fit, test.hex)
head(pred)

## Remove label
pred <- pred[,-1]
head(pred)

## Paste the ids (first col of test set) together with the predictions
submission <- h2o.cbind(test.hex[,1], pred)
head(submission)

## Save submission to disk
h2o.exportFile(submission, paste0(dir, "h2o.pred.csv"))