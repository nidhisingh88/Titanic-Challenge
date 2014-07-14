#Feature Engineering
library(rpart)
library(dplyr)
library(plyr)
library(aod)
library(randomForest)
library(party)
library(e1071)
library(Amelia)


imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[ 
      which( filter.var == v)])
  }
  return (impute.var)
}

train_csv<-read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/train.csv')
test_csv<-read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/test.csv')
test_csv$Survived <- NA
combi <- rbind(train_csv, test_csv)
combi$Name <- as.character(combi$Name)
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
combi$Title[combi$Title %in% c('Mme', 'Mlle', 'Ms')] <- 'Miss'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Col','Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combi$Title <- factor(combi$Title)
combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm=TRUE)

combi$FamilySize <- combi$SibSp + combi$Parch + 1
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

options(digits=2)
require(Hmisc)
# bystats(trai$Age, train$Title, 
#         fun=function(x)c(Mean=mean(x),Median=median(x)))
titles.na.train <- c("Dr", "Master", "Mrs", "Miss", "Mr")
combi$Embarked[which(is.na(combi$Embarked))] <- 'S'

## Try Amelia package for imputing missing values in Age
df.combi <- data.frame(combi)
noms<-c('Embarked','Title')
ords<-c('Age','FamilySize')
idvars<-c('Survived','Name','PassengerId','Ticket','Cabin','Parch','SibSp','FamilyID','Surname','Pclass','Fare','Sex')
a.out<-amelia(df.combi,noms=noms,idvars=idvars)
new.combi<-a.out$imputations[[1]]


combi$Age <- imputeMedian(combi$Age, combi$Title, 
                          titles.na.train)

## Trevor's Blog method
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])


train <- combi[1:891,]
test <- combi[892:1309,]


## Logistic Regression (0.78469 value on Kaggle and 505 rank)
logit.fit<- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, 
                       data = train, family="binomial")

Prediction <- predict(logit.fit,test,type="response")
Prediction <- ifelse(Prediction>0.5,1,0)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstlogit.csv", row.names = FALSE)

## Decision trees
set.seed(415)
dt.fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, method="class")
dt.predict <- predict(dt.fit, test, type = "class")                       
dt.submit <- data.frame(PassengerId = test$PassengerId, Survived = dt.predict)
write.csv(dt.submit, file = "firstdt.csv", row.names = FALSE)

## Random Forest
set.seed(415)
rf.fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, 
                       data=train, importance=TRUE, ntree=2000)
rf.predict <- predict(rf.fit, test, type = "class")                       
rf.submit <- data.frame(PassengerId = test$PassengerId, Survived = rf.predict)
write.csv(rf.submit, file = "firstrf.csv", row.names = FALSE)

## Conditional inference trees
set.seed(416)
ci.fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

ci.predict <- predict(ci.fit, test, OOB=TRUE, type = "response")                       
ci.submit <- data.frame(PassengerId = test$PassengerId, Survived = ci.predict)
write.csv(ci.submit, file = "firstci.csv", row.names = FALSE)


## SVM
svm.fit <- svm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                 data = train,type='C')
svm.predict <- predict(svm.fit, test)
svm.submit <- data.frame(PassengerId = test$PassengerId, Survived = svm.predict)
write.csv(svm.submit, file = "firstsvm.csv", row.names = FALSE)