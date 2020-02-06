
# do you have a package installed?
any(grepl('<name of your package>', installed.packages()))

library(tidyverse)
library(ggvis)
library(class)
library(gmodels)
library(caret)

#########################################
##### machine learning with 'knn()' #####
#########################################

# read in 'iris' data
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), 
                header = FALSE) 

# look at the data
head(iris)

# add column names
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

# iris scatter plot
iris %>% 
  ggvis(~Sepal.Length, ~Sepal.Width, fill = ~Species) %>% 
  layer_points()

iris %>% 
  ggvis(~Petal.Length, ~Petal.Width, fill = ~Species) %>%
  layer_points()

# correlation 'Petal.Length' and 'Petal.Width'
cor(iris$Petal.Length, iris$Petal.Width)

# return values of 'iris' levels 
x = levels(iris$Species)

# print Setosa correlation matrix
print(x[1])
cor(iris[iris$Species == x[1], 1:4])

# print Versicolor correlation matrix
print(x[2])
cor(iris[iris$Species == x[2], 1:4])

# print Virginica correlation matrix
print(x[3])
cor(iris[iris$Species == x[3], 1:4])

# return all 'iris' data
iris

# Return first 5 lines of 'iris'
head(iris)

# Return structure of 'iris'
str(iris)

# division of 'Species'
table(iris$Species) 

# percentual division of 'Species'
round(prop.table(table(iris$Species)) * 100, digits = 1)

# summary overview of 'iris'
summary(iris) 

# refined summary overview
summary(iris[c("Petal.Width", "Sepal.Width")])

##### normalization #####
# build a function to normalize the data with 'normalize()' 
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# normalize the 'iris' data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

# summarize 'iris_norm'
summary(iris_norm)
summary(iris)

##### training and test set approach #####
# assign probability wieghts of 0.67 and 0.33 to 'ind'
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# create a training set
iris.training <- iris[ind == 1, 1:4]

# look at the training set
head(iris.training)

# create a test set
iris.test <- iris[ind == 2, 1:4]

# look at the test set
head(iris.test)


# store the class labels in factor vectors and divide them over the training and test sets
# compose 'iris' training labels
iris.trainLabels <- iris[ind == 1,5]

# look at the result
print(iris.trainLabels)

# compose 'iris' test labels
iris.testLabels <- iris[ind==2, 5]

# look at the result
print(iris.testLabels)

##### build the model with knn() #####
# build the model
iris_pred <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)

# inspect 'iris_pred'
iris_pred

##### evaluate your model #####
# put 'iris.testLabels' in a data frame
irisTestLabels <- data.frame(iris.testLabels)

# merge 'iris_pred` and 'iris.testLabels' m
merge <- data.frame(iris_pred, iris.testLabels)

# Specify column names for `merge`
names(merge) <- c("Predicted Species", "Observed Species")

# Inspect `merge` 
merge

###### further model analysis #####
CrossTable(x = iris.testLabels, y = iris_pred, prop.chisq = FALSE)

########################################################
##### another example, same date but use 'caret()' #####
########################################################
# Create index to split based on labels  
index <- createDataPartition(iris$Species, p = 0.75, list = FALSE)

# Subset training set with index
iris.training <- iris[index, ]

# Subset test set with index
iris.test <- iris[-index, ]

# overview of algos supported by caret
names(getModelInfo())

# train your model
model_knn <- train(iris.training[, 1:4], iris.training[, 5], method = 'knn')

# predict the labels of the test set
predictions <- predict(object = model_knn, iris.test[, 1:4])

# ealuate the predictions
table(predictions)

# confusion matrix 
confusionMatrix(predictions, iris.test[ ,5])
