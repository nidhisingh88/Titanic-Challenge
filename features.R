library(rpart)
library(rattle)
library(rpart)
library(RColorBrewer)
library(caret)

train<-read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/train.csv')
test<-read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/test.csv')


#Combine datasets
test$Survived <- NA
combi <- rbind(train, test)

# Fill missing age values
# combi$Age[is.na(combi$Age)] <- mean(combi$Age)
combi$Age[is.na(combi$Age)] <- mean(combi$Age[!is.na(combi$Age)])

#Classify by title
combi$Name <- as.character(combi$Name)
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
# Combine small title groups
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
combi$Title <- factor(combi$Title)
combi$FamilySize <- combi$SibSp + combi$Parch + 1

combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'

#FamilyID
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

#Reduce factor levels for FamilyID
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

# Split back into test and train sets
train <- combi[1:891,]
test <- combi[892:1309,]

training.sample <- createDataPartition(train$Survived, p = 0.8, list = FALSE)
train.batch <- train[training.sample, ]
test.batch <- train[-training.sample, ]


