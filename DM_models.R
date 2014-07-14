# logistic regression with base features
# logit.m1 <- glm(glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, 
#                     data = train, family="binomial"))
logit.m1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                    data = train.batch, family="binomial")
logit.m2 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, 
                    data = train.batch, family="binomial")
logit.m3 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data = train.batch, family="binomial")
logit.m4 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, 
                data = train.batch, family="binomial")

# Naive Bayes
nb.m1 <- 