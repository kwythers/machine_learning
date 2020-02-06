##### clustering with k-means ##### 
# aka - unsupervised classification, because it classifies unlabled examples

# load libraries
library(stats)

# read in the data
teens <- read.csv('~/R_code/machine_learning/9781789618006_Code/Data/snsdata.csv')

# quick look
str(teens)

# example details
table(teens$gender)
# any NAs - non reported gender
table(teens$gender, useNA = 'ifany')
summary(teens)
##### obvious problems range of age!

# recode the age column to be NA if age is > 20 or < 13
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, 
                    teens$age, NA)
summary(teens$age)

# create a catagorical vaiable for "unknown gender" if not female or unknow, then must be male
teens$female <- ifelse(teens$gender == 'F' & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1 , 0)

# check results
table(teens$gender, useNA = 'ifany')
table(teens$female, useNA = 'ifany')
table(teens$no_gender, useNA = 'ifany')

##### data prep - imputing missing values
# we can get a mean for age, but we have to remove the NAs
mean(teens$age, na.rm = TRUE)
# aggrigate 
aggregate(data = teens, age ~ gradyear, mean, na.rm = TRUE)
# use ave() to coerce this into a vector so that it is easy to get back into the original df
ave_age <- ave(teens$age, teens$gradyear, FUN = 
                 function(x) mean(x, na.rm = TRUE))
# use ifelse to impute ave_age back onto the df
teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
# check that NAs are gone with summary()
summary(teens$age)

##### train the model on the data
# start by considering a smaller set (cols 5:540) of interests
interests <- teens[, 5:40]

# normalize with z-score and coerce back into df
interests_z <- as.data.frame(lapply(interests, scale))
# confirm with summary()
summary(interests$basketball)
summary(interests_z$basketball) # mean of zero and a max of 29... someone really mentions basketball a lot!!!

# pick a k (number of clusters) for teenagers... sometimes sterotypes are a useful place to being
# breakfast club... athlete, smart, criminal, princess, goofball (5) 
# use set.seed if you want to repeat your results
RNGversion('3.5.2')
set.seed(2345)

# run the kmeans function
teen_clusters <- kmeans(interests_z, 5)

##### evaluate model performance
# get size of each cluser
teen_clusters$size # big range from small to large clusters, but hard to interprit without more info

# look at cluster centroids
teen_clusters$centers # cluster 5 shows below average scores for every single interest

##### improving model performance
# apply the clusters back onto the full data set
teens$cluster <- teen_clusters$cluster

# how cluster assignment relates to individual characteristics - spersonal info for fisrt 5 teenagers
teens[1:5, c('cluster', 'gender', 'age', 'friends')]
# use aggrigate() to look at demographic characteristics
aggregate(data = teens, age ~ cluster, mean) # unsupriingly, age does not vary much by cluster
aggregate(data = teens, female ~ cluster, mean) # however, there are substantial differences with gender
aggregate(data = teens, friends ~ cluster, mean) # interesting number of friends seems to be related to steotype
