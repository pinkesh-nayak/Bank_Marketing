## Set Working Directory
#setwd('D:/Quarter3/STAT-642/Project/final')

## Load Libraries
library(e1071)
library(caret)
library(corrplot)
library(mice)
library(DMwR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(nnet)
library(e1071)
library(factoextra)
library(cluster)
library(fpc)
library(gridExtra)
library(class)

## Load Data
bank <- read.csv(file = 'bank-full.csv', sep=";")

## View the structure and summary information for the bank data. 
str(bank)
summary(bank)

# variable poutcome contains more than 75 % of the records which 
# are unknown. Value 'unknown' are the null records in the data.
table(bank$poutcome)
bank <- bank[ , !(names(bank) %in% "poutcome")]

#previous
table(bank$previous, bank$y)

# 'previous' variable has value ranging from 0-275 which contains value '0' for 
# more than 75% of the records i.e. no contacts performed before this campaign for 
#most of the records Data is skewed. This colummn contains outliers and is not significant.
bank <- bank[ , !(names(bank) %in% "previous")]

# Histogram
hist(bank$pdays, 
     main="pdays", 
     xlab="", 
     col="steelblue")

# Density plot
plot(density(bank$pdays), main="pdays")

# pdays = -1, means client was not previously contacted
# we can replace -1 with nulls
# From above graph more than 75% of the data have pdays = -1 or not contacted
# We need to drop this coulmn since it will not add significant value.

bank <- bank[ , !(names(bank) %in% "pdays")]

#Correltation
nums <- unlist(lapply(bank, is.numeric))
cor(bank[, nums])

# to identify any redundant variables
corrplot(cor(bank[ , nums]), method="circle")

# Scatterplot matrix
#pairs(bank[, nums])

# We can perform a Chi-Square Test for Independence between the two categorical
# variables
# H0: The two variables are not dependent
# H1: The two variables are dependent
table(bank$job, bank$y)
chisq.test(table(bank$job, bank$y))

table(bank$marital, bank$y)
chisq.test(table(bank$marital, bank$y))

table(bank$education, bank$y)
chisq.test(table(bank$education, bank$y))

table(bank$contact, bank$y)
chisq.test(table(bank$contact, bank$y))

# Identifying Outliers
# Boxplot
# Using a boxplot, we can visually identify outliers as those points
# that extend beyond the whiskers and use $out 
boxplot(bank$campaign, main="campaign")
#outlier <- boxplot.stats(bank$campaign)$out
boxplot(bank$duration, main="duration")
boxplot(bank$balance, main="balance")
boxplot(bank$age, main="age")

# absolute value of the Z-Score is greater than 3
nrow(bank[abs(scale(bank$campaign))>3,])
nrow(bank[abs(scale(bank$duration))>3,])
nrow(bank[abs(scale(bank$balance))>3,])
nrow(bank[abs(scale(bank$age))>3,])

# For variable 'campaign', large number (>12) of contacts performed seems
# unusual values in this dataset, and they can distort statistical analyses
# Since there are very less number of outliers, Dropping it.
#bank <- bank[!bank$campaign %in% outlier,]
bank <- bank[abs(scale(bank$campaign))<=3,]

# Variables 'duration', 'balance', 'age' are actual values which 
# can not be neglected. Capping the data.
# replace the outliers which are less than (Q1 - 1.5*IQR) with the 5th percentile and 
# replace the outliers which are greater than (Q3 + 1.5*IQR) with the 95th percentile of the data.
# Ref: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
# Ref: https://stackoverflow.com/questions/13339685/how-to-replace-outliers-with-the-5th-and-95th-percentile-values-in-r
hist(bank$duration, 
     main="duration", 
     xlab="", 
     col="steelblue")

hist(bank$balance, 
     main="balance", 
     xlab="", 
     col="steelblue")

hist(bank$age, 
     main="age", 
     xlab="", 
     col="steelblue")

fun <- function(x){
        quantiles <- quantile( x, c(0.05, 0.25, 0.75, 0.95 ) )
        IQR = quantiles[3] - quantiles[2]
        x[ x < quantiles[1] ] <- (quantiles[2] - 1.5*IQR)
        x[ x > quantiles[4] ] <- (quantiles[3] + 1.5*IQR)
        x
}

bank$duration <- fun(bank$duration)
bank$balance <- fun(bank$balance)
bank$age <- fun(bank$age)

# summary
summary(bank)

# After looking at the summary, variables 'job', 'education', 'contact'
# contains value 'unknown' which are the  missing records in the data.
# Replacing such records as NULL
bank[bank == "unknown"] <- NA
bank$job <- factor(bank$job)
bank$education <- factor(bank$education)
bank$contact <- factor(bank$contact)

summary(bank)

# Determine how many rows are missing values:
nrow(bank[!complete.cases(bank),])

# To identify duplicate observations
bank[duplicated(bank),] 
# No duplicate observations found

# Imputing missing values using mice package
# Since all the 3 variables are categorical, using method = "polyreg".
# Using Polytomous logistic regression to predict the level of missing data.
# Number of multiple imputations: m = 5
# Number of iterations: maxit = 5
# Ref: https://datascienceplus.com/handling-missing-data-with-mice-package-a-simple-approach/
init = mice(bank, maxit=0) 
meth = init$method
predM = init$predictorMatrix

meth[c("job")]="polyreg" 
meth[c("education")]="polyreg"
meth[c("contact")]="polyreg"

set.seed(123)
bank_imputed = mice(bank, method=meth, predictorMatrix=predM, m=5)

bank_imputed_df <- complete(bank_imputed)
summary(bank_imputed_df)

nrow(bank[!complete.cases(bank_imputed_df),])

# Label encoding on month variable.
bank2 <- bank_imputed_df

bank2$month <- match(tolower(bank2$month),tolower(month.abb))

summary(bank2)

# Creating dummy variables, conveting categorial variables.
dum <- dummyVars(~job+marital+education+default+housing+loan+contact, data=bank2,
                 sep="_", fullRank = TRUE)
df <- predict(dum, bank2)
bank3 <- data.frame(bank2[,!names(bank2) %in% c("job", "marital", "education", "default", "housing", "loan", "contact")], df)
summary(bank3)

# Data for kmeans
bank_kmeans <- bank3
bank_kmeans_y <- bank_kmeans[,7]
bank_kmeans=bank_kmeans[,-7]

# Using Scaled Data for kmeans
# Scaling only first 6 variables since they are continuous
# or ordinal variable while others are binary variables.
bank_kmeans_scaled <- sapply(bank_kmeans[,1:6], FUN = scale)
bank_kmeans <- bank_kmeans[,7:24]
bank_kmeans <- cbind(bank_kmeans_scaled, bank_kmeans)

#write.csv(bank_kmeans, file="bank_kmeans.csv")
#write.csv(bank3, file="bank3.csv")

# k-Means Clustering
# Plotting our possible k values (up to 9) on the x axis and tss on 
# the y-axis 
#fviz_nbclust(bank_kmeans, FUNcluster=kmeans, method="wss", k.max=9)

# Choose k  (plot total sum of squares)
tss<-rep(1,9)
kmeans1<-rep(1,9)
for (k in 1:9) {
        set.seed(123)
        kmeans1[k]=list(kmeans(bank_kmeans, centers=k, trace=FALSE, nstart=30))
        tss[k]=kmeans1[[k]]$tot.withinss
}
plot(1:9,tss)

# From above plot, its not proper elbow, so
# lets view for cluster 2, 3, 4, 5
grid.arrange(fviz_cluster(kmeans1[[2]], bank_kmeans), 
             fviz_cluster(kmeans1[[3]], bank_kmeans), 
             fviz_cluster(kmeans1[[4]], bank_kmeans),
             fviz_cluster(kmeans1[[5]], bank_kmeans), 
             nrow = 2)

dev.off()

# Since our data contains more binary variables,
# instead of using euclidean distance, we are using
# manhattan distance. Since our data is large
# we are using clara function which considers a small 
# sample of the data with fixed size (sampsize) and applies 
# the PAM algorithm to generate an optimal set of medoids for the sample.
cl2 <- clara(bank_kmeans, 2, metric = "manhattan",  
      samples = 10000, pamLike = TRUE)
	  
cl2$clusinfo

fviz_cluster(cl2, bank_kmeans)

cl3 <- clara(bank_kmeans, 3, metric = "manhattan",  
      samples = 10000, pamLike = TRUE)
	  
cl3$clusinfo

fviz_cluster(cl3, bank_kmeans)

cl4 <- clara(bank_kmeans, 4, metric = "manhattan",  
      samples = 10000, pamLike = TRUE)
	  
cl4$clusinfo

fviz_cluster(cl4, bank_kmeans)

cl5 <- clara(bank_kmeans, 5, metric = "manhattan",  
      samples = 10000, pamLike = TRUE)
	  
cl5$clusinfo

fviz_cluster(cl5, bank_kmeans)


# For cluster3, we can see data seperation.

# The mean of the dissimilarities of the observations
# to their closest medoid. This is used as a measure of 
# the goodness of the clustering.
cl3$clusinfo

## External Validation
table(y=bank_kmeans_y, Cluster=cl3$clustering)

# Using Min-Max Normalization for classification
prepObj <- preProcess(x=bank3, method="range")
bank3 <- predict(prepObj, bank3)

## Training & Testing
# Splitting the data into training and
# testing sets using a 80/20 split rule
set.seed(123)
samp <- createDataPartition(bank3$y, p=.80, list=FALSE)
train = bank3[samp, ] 
test = bank3[-samp, ]

# We can view the distribution of the dependent variable before resampling
table(train$y)

## SMOTE Synthetic Minority Oversampling Technique
set.seed(123)
train_sm <- SMOTE(y~., data=train)

# We can look at the class distribution after
table(train_sm$y)

#write.csv(train_sm, file="train_sm.csv")
#write.csv(test, file="test.csv")

# Feature selection
## Decision Trees
can.rpart <- rpart(formula = y ~ ., 
                   data = train_sm, 
                   method = "class")
# We can see the basic output of our tree
can.rpart

rpart.plot(can.rpart)

# We can view the variable importance, which is an element of
# our decision tree object, can.rpart
can.rpart$variable.importance

## Random Forest
set.seed(123)
can.rf <- randomForest(y~.,
                       data=train_sm,
                       importance=TRUE, 
                       ntree=500)

# We can view the output from our model
can.rf
# We can view the importance information for each class
can.rf$importance
# We can view the variable importance plot
varImpPlot(can.rf)

## Variable Importance using the varImp() function from the 
# caret package

# Decision Tree Model
varImp(can.rpart)

# Random Forest Model
varImp(can.rf)

## Recursive Feature Elimination (caret)
# We can also use an automatic feature selection method from the
# caret package to help us select variables.

# We first set up our control object. By setting functions=rfFuncs
# we are using random forest.
set.seed(123)
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 10,
                      repeats = 3,
                      verbose = FALSE)

can_rfe <- rfe(x = train_sm[,-7], 
               y = train_sm$y,
               rfeControl = control)

can_rfe

# The optVariables component of the rfe object contains the optimal
# variables chosen. 
can_rfe$optVariables
keeps <- c(as.list(can_rfe$optVariables), "y")

# If we want to select the variables based on this, we can create 
# new training and testing sets that contain only the selected 
# features
train_fs <- train_sm[, colnames(train_sm) %in% keeps]
test_fs <- test[,colnames(test) %in% keeps]

# ANN
## Hyperparameter Tuning 
# Using nnet package. We can adjust the size and decay.

# Size: number of nodes in the hidden layer. 
# There can only be one hidden layer using nnet
# Decay: weight decay. regularization parameter to avoid overfitting, 
# which adds a penalty for complexity.
# Use a random search with a tuneLength value of 10. Set the maximum
# number of iterations to 500.
## Training the model using Repeated 10-fold Cross Validation and
# grid search
ctrl_rocgrid <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 3,
                             search = "random",
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             savePredictions = "final")

set.seed(123)
annMod <- train(y~ ., data = train_fs, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl_rocgrid,
                tuneLength = 10,
                metric = "ROC",
                verbose=FALSE)

annMod
plot(annMod)
confusionMatrix(annMod)
annMod$bestTune
annMod$results
## Optimal size = 18 and optimal decay = 0.06966921
## ROC = 0.84, Sensitivity = 0.745, Specificity = 0.75

# We can apply our best fitting model to our training data
# to obtain predictions
ann_inpreds <- predict(annMod, newdata=train_fs)
ann_train_perf <- confusionMatrix(ann_inpreds, train_fs$y, mode="prec_recall", positive="yes")

# Finally, we can apply our best fitting model to our
# testing data to obtain our outsample predictions
ann_outpreds <- predict(annMod, newdata=test_fs)
ann_test_perf <- confusionMatrix(ann_outpreds, test_fs$y, mode="prec_recall", positive="yes")

## Performance information
cbind(ann_train=ann_train_perf$overall, ann_test=ann_test_perf$overall)
cbind(ann_train=ann_train_perf$byClass, ann_test=ann_test_perf$byClass)

## Performance information
## Training set: Accuracy = 84%, Kappa = 0.69, F1 = 0.82
## Testing set: Accuracy = 83%, kappa = 0.44, F1 = 0.53
## Accuracy for the model is good on both train and test. 
## F1 measure (goodness of fit) values on train is good but 
## on test it is fairly consistent with the train data.
## Precision is high on train but on test its low.
## Recall is high on both test and train.

## Decision Tree
## Decision Trees can handle missing values, irrelevant and redundant variables
## and are not distance based, so no need to perform rescaling of the variables. 
## For this reason, we can use the dataset as-is in our modeling,
## without any preprocessing and transformations.

## Hyperparameter Tuning/ Model Pruning for DT Models

# We want to tune the cost complexity parameter, or cp
# We choose the cp that is associated with the smallest
# cross-validated error (highest accuracy)

# We will use the caret package to perform 1) a grid search
# and 2) a random search for the optimal cp value.

cs.rpart <- rpart(formula = y ~ ., 
                  data = train_fs, 
                  method = "class")
# We can see the basic output of our tree
cs.rpart

tree.is.preds = predict(object=cs.rpart,
                        newdata=train_fs, type = "class")

tree_train_perf <- confusionMatrix(data=tree.is.preds, 
                                 reference=train_fs$y, 
                                 mode="prec_recall", positive="yes")

grids <- expand.grid(cp=seq(from=0,to=.4,by=.02))

tree.test.preds = predict(object=cs.rpart,
                        newdata=test_fs, type = "class")

tree_test_perf <- confusionMatrix(data=tree.test.preds, 
                                reference=test_fs$y,
                                mode="prec_recall", positive="yes")

## Comparing Performance							 
cbind(train=tree_train_perf$overall, test=tree_test_perf$overall)
cbind(train=tree_train_perf$byClass, test=tree_test_perf$byClass)

# We will use repeated 10-Fold cross validation and specify
# search="grid"
ctrl_grid <- trainControl(method="repeatedcv",
                          number = 10,
                          repeats = 3,
                          search="grid")
						  
# Creating the DTFit object, which is our 
# cross-validated model, in which we are tuning the
# cp hyperparameter
set.seed(123)
DTFit <- train(form=y ~ ., 
               data = train_fs, 
               method = "rpart",
               trControl = ctrl_grid, 
               tuneGrid=grids)

# Basic summary for the DTFit object
DTFit

## Optimal complexity parameter (cp) = 0
## its accuracy = 86% and kappa value = 0.72

# Plotting the cp value and Accuracy
plot(DTFit)

# We can obtain variable importance information from our model
varImp(DTFit)

# We can get the averaged confusion matrix across
# our resampled cross-validation models
confusionMatrix(DTFit)

# We will use our best model, found using grid search as our 
# final model. finalModel is a component of our DTFit object.
# We can get the a summary of our best fit model
DTFit$finalModel

# We can plot the tree for our best fitting model.
rpart.plot(DTFit$finalModel)

# Applying our best fitting model to our training data
# to obtain predictions
inpreds <- predict(object=DTFit, newdata=train_fs)

dt_train_perf <- confusionMatrix(data=inpreds, 
                              reference=train_fs$y, 
                              mode="prec_recall", positive="yes")
							  
# Finally, we can apply our best fitting model to our
# testing data to obtain our outsample predictions					  
outpreds <- predict(object=DTFit, newdata=test_fs)

dt_test_perf <- confusionMatrix(data=outpreds, 
                             reference=test_fs$y,
                             mode="prec_recall", positive="yes")

## Comparing Performance							 
cbind(train=dt_train_perf$overall, test=dt_test_perf$overall)
cbind(train=dt_train_perf$byClass, test=dt_test_perf$byClass)

## Performance information
## Training set: Accuracy = 92%, Kappa = 0.53, F1 = 0.90
## Testing set: Accuracy = 84%, kappa = 0.41, F1 = 0.50
## The model performs well on the data on which it was trained but not 
## on test since F1 measure (goodness of fit) values on train is good but 
## on test data its not.
## Recall and precision data are high for train set but not for test.
## Performance information on test data is fairly consistent with the train data.
detach("package:class", unload=TRUE)

#KNN
## Using a 'best guess' value of k
# Let's find a good starting point for k
k.choice <- ceiling(sqrt(nrow(train_fs)))
k.choice
# k value is 172 but we should opt for either 171 or 173, so that we avoid ties
k.choice <- 173

knn.pred <- knn(train=train_fs[,-7],
                test=test_fs[,-7],
                cl=train_fs$y,
                k=k.choice)

# With kNN, no model is built, so we cannot assess performance
# on the training set. For this reason, we move on to looking at
# performance on the testing set.
# Our default confusion matrix and metrics
confusionMatrix(data=knn.pred, reference=test_fs$y, positive="yes",)
# Our confusion matrix with Precision, Recall and F1 Measure
confusionMatrix(data=knn.pred, reference=test_fs$y,positive="yes",
                mode="prec_recall")

## Hyperparameter Tuning for kNN Models
# We want to tune the number of nearest neighbors, or k value
# We choose the k that is associated with the smallest
# cross-validated error (highest accuracy)

# We will use the caret package to perform 1) a grid search using
# accuracy as the performance metric and 2) a grid search using
# the AUC as the performance metric.


## 1) Grid Search for Optimal k: Accuracy
# First, we set up our grid to be only odd numbers from 3 to 33
knn_grids <- expand.grid(k=seq(from=149,to=199,by=2))
knn_grids

# We will use repeated 10-fold cross validation, using grid
# search method
ctrl_grid <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3,
                          search = "grid")

# We set our initial seed
set.seed(123)

# Next, we create the knnFit object, which is our
# cross-validated model, in which we are tuning the
# k hyperparameter using a grid search, with Accuracy as
# our performance measure (default)
knnFit <- train(form = y ~ .,
                data = train_fs,
                method = "knn",
                trControl = ctrl_grid,
                tuneGrid = knn_grids)

# We can view the basic summary for the knnFit object
knnFit

# We can plot the k value and Accuracy
plot(knnFit)

# Finally, we can get the averaged confusion matrix across
# our resampled cross-validation models
confusionMatrix(knnFit)


## 1) Grid Search for Optimal k: AUC
# Especially if we have problems with imbalance,we will
# prefer to use other metrics than Accuracy to assess our
# model.
#In our dataset we balanced the dataset by using SMOTE technique. But still we wanted 
#to compare the results but the accuracy obtained was the same while using both the metrics.


# If we are more concerned about the trade-off between the
# true positive rate and false positive rate, we may prefer
# to use the AUC to choose our k value.
ctrl_rocgrid <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 3,
                             search = "grid",
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             savePredictions = "final")

# We set our seed
set.seed(123)

# Next, we create the knnFit2 object, which is our
# cross-validated model, in which we are tuning the
# k hyperparameter using AUC as our performance measure
knnFit2 <- train(form = y ~ .,
                 data = train_fs,
                 method = "knn",
                 trControl = ctrl_rocgrid,
                 tuneGrid = knn_grids,
                 metric = "ROC"
)

# We can view the basic summary for the knnFit2 object
knnFit2
# We can plot the k value and Area Under the Curve
plot(knnFit2)
# Finally, we can get the averaged confusion matrix across
# our resampled cross-validation models
confusionMatrix(knnFit2)


## Outsample Prediction using Tuned Model
# First, we use the predict() function to predict the value of
# the y variable using the knnFit2 model and the testing
# data set
outpreds <- predict(knnFit2, newdata=test_fs)

# Next, we use the confusionMatrix() function and as input
# use our prediction (outpreds) and the actual y to
# obtain performance information on our testing dataset for our
# best kNN model based on AUC

# Default
confusionMatrix(outpreds, test_fs$y,positive="yes")

# Precision/Recall
knn_perf <- confusionMatrix(outpreds, test_fs$y, positive="yes",mode="prec_recall")

## Performance information
## Accuracy = 77.6%, Kappa = 0.3, F1 = 0.41
## We are getting a good model accuracy.
## F1 measure (goodness of fit) value is low.
## Low precision. Recall = 0.67

## Comparing all the models
cbind(knn=knn_perf$overall, dt_train=dt_train_perf$overall, dt_test=dt_test_perf$overall, ann_train=ann_train_perf$overall, ann_test=ann_test_perf$overall)
cbind(knn=knn_perf$byClass, dt_train=dt_train_perf$byClass, dt_test=dt_test_perf$byClass, ann_train=ann_train_perf$byClass, ann_test=ann_test_perf$byClass)

## Of all the models created, the ANN model is preferred because
## its accuracy is high for both train and test. Its F1 measure 
## (goodness of fit) is high for the training set but for the test 
## set its fairly consistent. Recall value is high for both train 
## and test. Precision is high for the training set and low for the 
## test. Here our main focus is on recall than precision because we 
## need to maximize minority class 'yes', client subscribing for the 
## term deposit. We can afford to wrongly predict a few 'no' class as 
## 'yes' because we can manage to make a few more calls but we can not 
## lose potential customers/clients who are more likely to subscribe 
## to a term deposit.

save.image("Bank_Marketing.RData")
