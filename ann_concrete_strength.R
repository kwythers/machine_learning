# load lobraries
library(tidyverse)
library(neuralnet)


# get data
concrete <- read.csv('~/R_code/machine_learning/9781789618006_Code/Data/concrete.csv')

# explore and prep the data
str(concrete)

# look at the distributeion to determine best way to normalize
ggplot(gather(concrete), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')
##### if the disributions are normal, then it makes sense to use the scale() function, hoever, since
##### these distributions are non-normal (or uniform), then use a normalize function

# build the function
normalize <- function(x) {
  return(( x - min(x)) / (max(x) - min(x)))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))

# compare the two data sets
summary(concrete)
summary(concrete_norm)

# partition into training and testing data (75:25), assume randomness
# if you do not believe that the data are random , use the following code to create an ID and split (70:30)
# df <- wine %>% mutate(id = row_number() ) # check IDs
# head(df$id) # create training set
# train <- df %>% sample_frac(.70) # create test set
# test  <- anti_join(df, train, by = 'id')
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

# train simple multilayer feedforward network - default settings - single hidden layer
concrete_model <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, 
                            data = concrete_train)

# plot the model
plot(concrete_model)

##### evaluating model performance
# generate predictions on the test data
model_results <- compute(concrete_model, concrete_test[1:8])
# save the predicted values from net.result
predicted_strength <- model_results$net.result

# correlation between predicted strength and true values
cor(predicted_strength, concrete_test$strength)
#### fairly good correlation given that we are using a single node

##### improving model performance 
# add more hidden layers
concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, 
                            data = concrete_train, hidden = 5)

plot(concrete_model2)
###### note big drop in SSE (from 5.08 to 1.63) and the big rise in training steps

# check correlation for new predictions
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

##### further improvements
# add more hidden layers, change activiation functions
# rectifier  - softplus #
# define the softplus function
softplus <- function(x) { log(1 + exp(x)) }

# use this activation function in neuralnet with the act.fct paramter and a second hidden layer of 5 nodes with 
# the hidden paramter and an int vector of c(5, 5) - results in a two layer network 

set.seed(12345)
concrete_model3 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age,
                             data = concrete_train, 
                             hidden = c(5, 5), 
                             act.fct = softplus)

# plot the model
plot(concrete_model3)

model_results3 <- compute(concrete_model3, concrete_test[1:8])
