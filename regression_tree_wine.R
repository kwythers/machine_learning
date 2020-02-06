# laod libraries
library(tidyverse)
library(rpart)
library(rpart.plot)

# Estimating the quality of wines with regression trees and model trees
wine <- read.csv("~/R_code/machine_learning/9781789618006_Code/Data/wine_white.csv", sep=";")

# quick look at the data structure and histogram
str(wine)
hist(wine$quality)
summary(wine)

# divide data into training and testing sets
# assume randomsess and split numerically (70:30)
wine_train <- wine[1:3750, ]
wine_test <- wine[3751:4898, ]

# if you do not believe that the data are random , use the following code to create an ID and split (70:30)
df <- wine %>% mutate(id = row_number() ) # check IDs
head(df$id) # create training set
train <- df %>% sample_frac(.70) # create test set
test  <- anti_join(df, train, by = 'id')

##### train the model on the data
# build the model
m.rpart <- rpart(quality ~ ., data = wine_train)

# bsic info about the tree
m.rpart
##### for each node in the tree, the number of examples reaching the decision point is listed - in this case
##### all 3750 examples bein at the root node, of which 2372 have 'alcohol < 10.85 - because alcohol was used
##### in the first, it is the single most important predictor of wine quality - nodes denoted by '*' are teminal
##### or leaf nodes, which means they result in a prediction (yval) - for example node 5 has a yval of 5.971091 - 
##### when the tree is used for predictions, any winde samples with alcohol < 10.85 and volitile.acidity < 0.2425 
##### would be predicted to have a quality value of 5.881912

# more details
summary(m.rpart)

##### visualizing descision trees
# plot the descision tree
rpart.plot(m.rpart, digits = 3)

# a few useful options
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

##### evaluate model performance
# make predictions on test data, use predict()
p.rpart <- predict(m.rpart, wine_test)

# check out the predictions and the test data
summary(p.rpart)
summary(wine_test$quality)
##### a quick look at the summary stats show a potential problem - the predictions fall on a much narrower range than
##### the true values - this suggests that the model is correctly identifying the tail cases (the best and worst wines)

# check the correlation for a quick evaluation of performance
cor(p.rpart, wine_test$quality)
##### 0.49 is acceptable, but it only measures how strongly the predictions are related to the true value, not a 
##### measure of how far off the predictions were from the true values

##### measure performance with absolute mean error
# create a smaple MAE function
MAE <- function(actual, predicted) { 
  mean(abs(actual - predicted))
}

# run MAE
MAE(p.rpart, wine_test$quality)
##### implies, on average, the difference between out models predictions and the true quality score was about 0.57
##### on a scale from 0 - 10, this seems pretty good

# mean quality rating of the training data
mean(wine_train$quality)
MAE(5.87, wine_test$quality)
##### if we predicted 5.89 for every wine sample, we would have a mean aboslute error of 0.581, our regression tree 

##### improve model performance with model tree #####
# load libraries
library(Cubist)

# make model with cubist() - specify columns used in the model (in this case )
m.cubist <- cubist(x = wine_train[-12], y = wine_train$quality)

# basic info
m.cubist
summary(m.cubist)

##### check model performance
# get predictions from the test set
p.cubist <- predict(m.cubist, wine_test)

# look at the predictions
summary(p.cubist)
summary(wine_test$quality)
##### we have gotten closer to the quality range of the test test

# check correlations
cor(p.cubist, wine_test$quality)
cor(p.rpart, wine_test$quality)
##### cubist gives us a much better correlation

##### compare MAEs
MAE(wine_test$quality, p.cubist)
MAE(wine_test$quality, p.rpart)
##### slightly reduced the mean absolute error, we eused a much simplier model to xceeded the levels of 
##### a neural network model published by Cortez, but not to the 0.45 level from a support vector machine model

