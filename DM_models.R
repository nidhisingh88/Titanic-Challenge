library(knitr)
library(randomForest)
library(ROCR)
library(caret)
library(kernlab)
library(stargazer)
library(vcd)
library(ggplot2)
library(rpart)
library(e1071)
set.seed(45)

# logistic regression with base features
# logit.m1 <- glm(glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, 
#                     data = train, family="binomial"))
logit.m1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + 
                  Parch + Fare + Embarked, 
                data = train.batch, 
                family="binomial")
logit.m2 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + 
                  Embarked + Title +
                  FamilyID2,
                data = train.batch, 
                family="binomial")
#logit prediction
logit.predict <- predict(logit.m2,test.batch,type="response")
Prediction <- ifelse(logit.predict>0.5,1,0)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstlogit.csv", row.names = FALSE)

# Naive Bayes
nb.classifier <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + 
                              Age + SibSp + Parch + Embarked + 
                              Title + FamilyID2, 
                            data = train.batch)
#NB prediction
nb.predict <- predict(nb.classifier, test.batch, type='raw')

submit <- data.frame(PassengerId = test$PassengerId, Survived = nb.predict)
write.csv(submit, file = "firstNB.csv", row.names = FALSE)

#DT
dt.1 <- rpart(Survived ~ Pclass + Sex + Age + Fare + 
                Embarked + Title + FamilyID2,
              data = train.batch, method = "class", 
              control = rpart.control(minsplit = 2, 
                                      cp = 0.001))
dt.2 <- rpart(Survived ~ Pclass + Fare + Embarked + 
                Title + FamilyID2, data=train.batch,
              method="class")

dt.3 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + 
                Embarked + Title +
                FamilyID2, data=train.batch,
              method="class",control=rpart.control(minsplit=16,cp=0.001))

dt.3.prob <- predict(dt.3, type="prob", test.batch)

# SVM
svm.model.1 <- ksvm(Survived ~ Sex + Pclass + Age + Fare + SibSp + Parch + Embarked, data = train.batch, kernel="rbfdot",type="C-svc")

svm.model.2 <- ksvm(Survived ~ Sex + Pclass + Embarked, 
                    data = train.batch, kernel="rbfdot",
                    type="C-svc")

svm.model.3 <- ksvm(Survived ~ Sex + Pclass + Embarked + 
                      Title + FamilyID2, 
                    data = train.batch, kernel="rbfdot",
                    type="C-svc",
                    prob.model=TRUE)

svm.model.4 <- ksvm(Survived ~ Pclass + Sex + Age + SibSp + Parch + 
                      Embarked + Title +
                      FamilyID2,
                    data = train.batch, kernel="rbfdot",
                    type="C-svc",
                    prob.model=TRUE)

svm.prob <- predict(svm.model.4, type="prob", test.batch)

# For kaggle
svm.prob <- predict(svm.model.4, type="prob", test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = svm.prediction)
write.csv(submit, file = "svm.csv", row.names = FALSE)

#Random Forest
bestmtry <- tuneRF(data.frame(train.batch$Pclass,train.batch$Sex,
                              train.batch$Age,train.batch$SibSp,
                              train.batch$Parch,train.batch$Embarked,
                              train.batch$Title,train.batch$FamilyID2),
                   train.batch$Survived, ntreeTry=100, 
                   stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, 
                   dobest=FALSE)

rf.m1 <-randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                       Parch + Fare+ FamilySize +
                       Embarked + Title + FamilyID2,
                     data=train.batch, mtry=3, ntree=1000, 
                     keep.forest=TRUE, importance=TRUE)
rf.predict.1 <- predict(rf.m1, test.batch, type = "response")
confusionMatrix(rf.predict.1, test.batch$Survived)

rf.m2 <-randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                       Parch + Fare+ FamilySize +
                       Embarked + Title + FamilyID2,
                     data=train.batch, mtry=3, ntree=2000, 
                     keep.forest=TRUE, importance=TRUE)
rf.predict.2 <- predict(rf.m2, test, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = rf.predict.2)
write.csv(submit, file = "rf.csv", row.names = FALSE)
confusionMatrix(rf.predict.2, test.batch$Survived)

rf.m2.prob <- predict(rf.m2,test.batch,type='prob')

rf.m3 <-randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                       Parch + Fare+ FamilySize +
                       Embarked + Title + FamilyID2,
                     data=train.batch, mtry=3, ntree=5000, 
                     keep.forest=TRUE, importance=TRUE)
rf.predict.3 <- predict(rf.m3, test.batch, type = "response")
confusionMatrix(rf.predict.3, test.batch$Survived)

##ROCR

logit.rocr.pred <- prediction(logit.predict,test.batch$Survived)
nb.rocr.pred <- prediction(nb.predict[,2],test.batch$Survived)
dt.rocr.pred <- prediction(dt.3.prob[,2],test.batch$Survived)
rf.rocr.pred <- prediction(rf.m2.prob[,2],test.batch$Survived)
svm.rocr.pred <- prediction(svm.prob[,2],test.batch$Survived)
logit.perf <- performance(logit.rocr.pred, "tpr","fpr")
nb.perf <- performance(nb.rocr.pred, "tpr","fpr")
dt.perf <- performance(dt.rocr.pred, "tpr","fpr")
svm.perf <- performance(svm.rocr.pred, "tpr","fpr")
rf.perf <- performance(rf.rocr.pred, "tpr","fpr")
plot(dt.perf, col=2)
plot(svm.perf,col=3,add=TRUE)
plot(rf.perf,col=4,add=TRUE)
plot(logit.perf,col=5,add=TRUE)
plot(nb.perf,col=6,add=TRUE)
#legend(0.5, 0.5, c('dt', 'svm', 'rforest','logit','naive bayes'), 2:6)

c.legend<-c('dtree, auc=','svm, auc=','rforest, auc=','logit, auc=','nBayes, auc=')
c.legend[1]<-paste(c.legend[1],round((performance(dt.rocr.pred,'auc')@y.values)[[1]],3))
c.legend[2]<-paste(c.legend[2],round((performance(svm.rocr.pred,'auc')@y.values)[[1]],3))
c.legend[3]<-paste(c.legend[3],round((performance(rf.rocr.pred,'auc')@y.values)[[1]],3))
c.legend[4]<-paste(c.legend[4],round((performance(logit.rocr.pred,'auc')@y.values)[[1]],3))
c.legend[5]<-paste(c.legend[5],round((performance(nb.rocr.pred,'auc')@y.values)[[1]],3))
legend(0.3,0.5, c.legend,lty=c(1,1,1,1,1),lwd=c(2,2,2,2,2),col=2:6)

# Specificity and Sensitivity
logit.perf.ss <- performance(logit.rocr.pred, measure="sens", x.measure="spec")
nb.perf.ss <- performance(nb.rocr.pred, measure="sens", x.measure="spec")
dt.perf.ss <- performance(dt.rocr.pred, measure="sens", x.measure="spec")
svm.perf.ss <- performance(svm.rocr.pred, measure="sens", x.measure="spec")
rf.perf.ss <- performance(rf.rocr.pred, measure="sens", x.measure="spec")

plot(dt.perf.ss, col=2)
plot(svm.perf.ss,col=3,add=TRUE)
plot(rf.perf.ss,col=4,add=TRUE)
plot(logit.perf.ss,col=5,add=TRUE)
plot(nb.perf.ss,col=6,add=TRUE)

legend(0.1, 0.5, c('dt', 'svm', 'rforest','logit','naive bayes'), 2:6)
