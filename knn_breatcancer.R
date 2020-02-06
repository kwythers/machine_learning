library(tidyverse)
library(class)
library(gmodels)

wbcd <- read.csv('/Users/kirkw/R_code/machine_learning/9781789618006_Code/Data/wisc_bc_data.csv', stringsAsFactors = FALSE)

# drop first column
wbcd <- wbcd[-1]

# benign or malignant
table(wbcd$diagnosis)

# classify as a factor and give better names
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c('B', 'M'), 
                         labels = c('Benign', 'Malignant'))

# % of each factor
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)
# what kind of data
str(wbcd)
# closer look at three columns
summary(wbcd[c('radius_mean', 'area_mean', 'smoothness_mean')])

# create a 'normalize' function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# test the function
normalize(c(0,1,2,3,4,5,6,7,8,9,10))
normalize(c(10,20,30,40,50,60,70,80,90,100))

# use lapply to apply function to all numeric columns while keeping first two columns
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize)) 

# create random training and testing datasets
# wbcd_train <- wbcd %>% sample_frac(0.70) 
# wbcd_test  <- anti_join(wbcd, wbcd_train, by = 'id') 

# since these data were previously randomized, simply split
wbcd_n_train <- wbcd_n[1:469, ]
wbcd_n_test <- wbcd_n[470:569, ]

# train and test labels
wbcd_n_train_labels <- wbcd[1:469, 1]
wbcd_n_test_labels <- wbcd[470:569, 1]

# knn to test the data
wbcd_n_test_pred <- knn(train = wbcd_n_train, test = wbcd_n_test, 
                      cl = wbcd_n_train_labels, k = 21)

# model performance eval
CrossTable(x = wbcd_n_test_labels, y = wbcd_n_test_pred, prop.chisq = FALSE)

# respeat above apporach with z-score to see if any different from normalized data
# use scale() to apply z-score to oringinal data
wbcd_z <- as.data.frame(scale(wbcd[-1]))

# since these data were previously randomized, simply split
wbcd_z_train <- wbcd_z[1:469, ]
wbcd_z_test <- wbcd_z[470:569, ]

# train and test labels
wbcd_z_train_labels <- wbcd[1:469, 1]
wbcd_z_test_labels <- wbcd[470:569, 1]

# knn to test the data
wbcd_z_test_pred <- knn(train = wbcd_z_train, test = wbcd_z_test, 
                      cl = wbcd_z_train_labels, k = 21)

# model performance eval
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)

summary(wbcd_z$area_mean)


