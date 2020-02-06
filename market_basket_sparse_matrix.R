##### Finding patterns in dbs #####

# set path
setwd('~/R_code/machine_learning/')

# load libraries
library(arules)

# read in the data
groceries <- read.transactions('~/R_code/machine_learning/9781789618006_Code/Data/groceries.csv')

summary(groceries_t)

# view sparse matrix contents
inspect(groceries[1:5])

# view the support level for first three items 
itemFrequency(groceries[ ,1:3])

# visualize
itemFrequencyPlot(groceries, support = 0.1) # broken
itemFrequencyPlot(groceries, topN = 10)

# plot the sparse matrix
image(groceries[1:5])

# random sample of 100 purchases
image(sample(groceries, 100))

##### train the model on the data
# support, purchased 2x a day is 60 times per month, or 0.006 ~= 60 / 9835 
# confidence is 0.25 (correct 25% of the time)
# minlen = 2 (eliminates rules containing fewer than 2 itmes)
 grocery_rules <- apriori(groceries, parameter = list(support = 0.006, confidence = 0.25, minlen = 2))

 grocery_rules
 
 ##### evaluating model performance
 summary(grocery_rules)
 
 # look at spefic rules
 inspect(grocery_rules[1:5])
 # sort by confidence for all 36 rules
 inspect(sort(grocery_rules, by = 'confidence')[1:36])

 