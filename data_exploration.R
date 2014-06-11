setwd('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem')
library(vcd)
library(ggplot2)
train_csv <- read.csv('/home/nidhi/Courses//TUDelft-Data Analytics/Kaggle:Titanic problem/train.csv')
str(train_csv)
prop.table(table(train_csv$Survived))
summary(train_csv$Age)

prop.table(table(train_csv$Pclass))

mosaicplot(train_csv$Pclass ~ train_csv$Sex, 
           main='Passenger Class by Gender',
           shade=FALSE, 
           color=TRUE, xlab="Pclass", ylab="Sex of passenger")

bpa <- ggplot(train_csv, aes(factor(Pclass), Age))
bpa + geom_boxplot()

bpf <- ggplot(train_csv, aes(factor(Pclass), Fare))
bpf + geom_boxplot()+ylim(c(0,300))

qplot(train_csv$SibSp,main='Travelling with Siblings/Spouse',xlab='No. of Siblings/Spouse')

qplot(train_csv$Parch,main='Travelling with Parents/Children',xlab='No. of Parents/Children')

qplot(train_csv$Embarked,main='Travelers Place of Embarkment',xlab='Place of Embarkment')

bsa <- ggplot(train_csv, aes(factor(Survived), Age))
bsa + geom_boxplot() + xlab('Passenger Survived')+ geom_jitter()

bsf <- ggplot(train_csv, aes(factor(Survived), Fare))
bsf + geom_boxplot()+ylim(c(0,300))

structable(data=train_csv,Survived~Sex)
structable(data=train_csv,Survived+Sex~Pclass)
structable(data=train_csv,Survived~Pclass)

doubledecker(Survived ~ Sex + Pclass, data=train_csv)
doubledecker(Survived ~ Sex, data=train_csv)
doubledecker(Survived ~ Pclass, data=train_csv)
