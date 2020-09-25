---
## Load packages

library(dplyr)
library(lubridate)
library(car)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(xgboost)
library(readr)
library(MLmetrics)

training <- read_csv("R/wd-PAF/training_prep_2.csv")
testing <- read_csv("R/wd-PAF/testing_2.csv")

mod.mat <- model.matrix( ~ .,
                         data = training %>% select (-result))
response <- training$result
d.mat <- xgb.DMatrix(data = mod.mat, label = response)

mod.mat.testing <- model.matrix( ~ .,
                         data = testing)
d.mat.testing <- xgb.DMatrix(data = mod.mat.testing)

dim(d.mat)
dim(d.mat.testing)

watchlist <- list(train=d.mat, test=d.mat.testing)
param <- list(max_depth = 2, eta = 1, nthread = 2, objective = 'binary:logistic')
nrounds <- 100

bstDMatrix <- xgboost(data = d.mat, max.depth = 2, 
                      eta = 1, nthread = 2, nrounds = 2,
                      objective = "binary:logistic")

xgb.importance(model = bstDMatrix)

cat('running cross validation\n')
xgb.cv(param, d.mat, nrounds, nfold = 20, metrics = {'error'})
xgb.cv(param, d.mat, nrounds, nfold = 20, metrics = 'error', showsd = FALSE)

pred <- predict(bstDMatrix, mod.mat)
pred_1 <- ifelse(pred > 0.5, 1, 0)

ypred1 <- predict(bstDMatrix, d.mat.testing, ntreelimit = 1)
ypred1_re <- ifelse(ypred1 > 0.5, 1, 0)

ypred2 <- predict(bstDMatrix, d.mat.testing)
ypred1_re <- ifelse(ypred2 > 0.5, 1, 0)

cat('error of ypred1=', mean(as.numeric(pred > 0.5)), '\n')
cat('error of ypred1=', mean(as.numeric(ypred1 > 0.5)), '\n')
cat('error of ypred1=', mean(as.numeric(ypred2 > 0.5)), '\n')

F1_Score(y_true = response, pred_1, positive = "1")


