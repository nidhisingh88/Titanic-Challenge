## Logistic Regression

library(rpart)
test_csv<-read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/test.csv')
test_csv$Survived <- rep(0, 418)
log.fit<-glm(Survived ~ Sex + Pclass + Age + Fare + Embarked, 
             data = train_csv, family=binomial("logit"))
Prediction <- predict(log.fit, test_csv)
submit <- data.frame(PassengerId = test_csv$PassengerId, Survived = test_csv$Survived)
write.csv(submit, file = "perish.csv", row.names = FALSE)
