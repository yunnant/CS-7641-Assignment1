#Clean space
rm(list=ls())

#Loading the required libraries
library('caret')
library('pROC')
library('ROCR')
library('clusterSim')

#Open parallel calculation
library(doParallel)
cl <- makePSOCKcluster(5)
#registerDoParallel(cl)

#Seeting the random seed
set.seed(7)

#Loading the NHANES dataset
data <- read.csv('/Users/yunnantao/Documents/Programming/Machine Learning/data_NHANES.csv', header = TRUE)
#data <- read.csv('/Users/yunnantao/Documents/Programming/Machine Learning/data_PIMA.csv', header = TRUE)
data$status<-ifelse(data$status=='N',0,1)
data$DMDEDUC2<-as.factor(data$DMDEDUC2)
data$INDHHIN2<-as.factor(data$INDHHIN2)
data$RIDRETH1<-as.factor(data$RIDRETH1)
data$PAQ<-as.factor(data$PAQ)

#Check the structure and missing value of dataset
str(data)
sum(is.na(data))

#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = data,fullRank = T)
data_transformed <- data.frame(predict(dmy, newdata = data))

#Converting the dependent variable back to categorical
data_transformed <- data.Normalization (data_transformed,type="n4",normalization="column")
data_transformed$status<-ifelse(data_transformed$status=='0','N','Y')
data_transformed$status<-as.factor(data_transformed$status)

#Check the structure and missing value of dataset
str(data_transformed)
sum(is.na(data_transformed))

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(data_transformed$status, p=0.75, list=FALSE)
trainSet <- data_transformed[ index,]
testSet <- data_transformed[-index,]

#Defining the training control
fitControl <- trainControl(method = "cv", number = 5, savePredictions = 'final', classProbs = T)

#Defining the predictors and outcome
predictors <- names(trainSet)[!names(trainSet) %in% 'status']
outcomeName <- 'status'

#Training the Neural Networks model
nnGrid <-  expand.grid(size = c(7, 9), 
                        decay = c(0.3,0.5)) 
knnGrid <-  expand.grid(k=c(1,3,5,7,9,13,15,17,19,21))                     

#model_nn<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',trControl=fitControl,tuneLength=3,tuneGrid=nnGrid)

#Training the knn model
#model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3,tuneGrid=knnGrid)

#Training the svm model
model_svm<-train(trainSet[,predictors],trainSet[,outcomeName],method='svmPoly',trControl=fitControl,tuneLength=3)

#Training the decision tree model
#model_dt<-train(trainSet[,predictors],trainSet[,outcomeName],method='rpartScore',trControl=fitControl,tuneLength=3)

#Training the gradient boosting model
#model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=3)

#Predicting the out of fold prediction probabilities for training data
trainSet$pred_svm<-model_svm$pred$Y[order(model_svm$pred$rowIndex)]
if(false){
trainSet$pred_nn<-model_nn$pred$Y[order(model_nn$pred$rowIndex)]
trainSet$pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
trainSet$pred_svm<-model_svm$pred$Y[order(model_svm$pred$rowIndex)]
trainSet$pred_dt<-model_dt$pred$Y[order(model_dt$pred$rowIndex)]
trainSet$pred_gbm<-model_gbm$pred$Y[order(model_gbm$pred$rowIndex)]
}
#Predicting probabilities for the test data
testSet$pred_svm<-predict(model_svm,testSet[,predictors],type='prob')$Y
if(false){
testSet$pred_knn<-predict(model_knn,testSet[,predictors],type='prob')$Y
testSet$pred_nn<-predict(model_nn,testSet[,predictors],type='prob')$Y
testSet$pred_svm<-predict(model_svm,testSet[,predictors],type='prob')$Y
testSet$pred_dt<-predict(model_dt,testSet[,predictors],type='prob')$Y
testSet$pred_gbm<-predict(model_gbm,testSet[,predictors],type='prob')$Y
}
knn.ROC <- roc(predictor=testSet$pred_knn, response=testSet$status,levels=rev(levels(testSet$status)))
plot(knn.ROC,main="KNN - ROC",print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)
svm.ROC <- roc(predictor=testSet$pred_svm, response=testSet$status,levels=rev(levels(testSet$status)))
plot(svm.ROC,main="SVM - ROC",print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)
dt.ROC <- roc(predictor=testSet$pred_dt, response=testSet$status,levels=rev(levels(testSet$status)))
plot(dt.ROC,main="DT- ROC",print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)
gbm.ROC <- roc(predictor=testSet$pred_gbm, response=testSet$status,levels=rev(levels(testSet$status)))
plot(gbm.ROC,main="GBM - ROC",print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)
nn.ROC <- roc(predictor=testSet$pred_nn, response=testSet$status,levels=rev(levels(testSet$status)))
plot(nn.ROC,main="NN - ROC",print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)

pred <- prediction(testSet$pred_nn, testSet$status)
RP.knn <- performance(pred, "prec", "rec");
plot (RP.knn);

testSet$pred_svm_YN<-as.factor(ifelse(testSet$pred_svm> 0.5,"Y","N"))
if(false){
testSet$pred_knn_YN<-as.factor(ifelse(testSet$pred_knn> 0.5,"Y","N"))
testSet$pred_svm_YN<-as.factor(ifelse(testSet$pred_svm> 0.5,"Y","N"))
testSet$pred_dt_YN<-as.factor(ifelse(testSet$pred_dt> 0.5,"Y","N"))
testSet$pred_gbm_YN<-as.factor(ifelse(testSet$pred_gbm> 0.5,"Y","N"))
testSet$pred_nn_YN<-as.factor(ifelse(testSet$pred_nn> 0.5,"Y","N"))
}
precision <- posPredValue(testSet$pred_knn_YN, testSet$status, positive="N")
recall <- sensitivity(testSet$pred_knn_YN, testSet$status, positive="N")
F1 <- (2 * precision * recall) / (precision + recall)

confusionMatrix(testSet$status,testSet$pred_knn_YN)
confusionMatrix(testSet$status,testSet$pred_svm_YN)
confusionMatrix(testSet$status,testSet$pred_dt_YN)
confusionMatrix(testSet$status,testSet$pred_gbm_YN)
confusionMatrix(testSet$status,testSet$pred_nn_YN)


Accuracy_data <-
  learning_curve_dat(dat = trainSet,
                     outcome = "status",
                     test_prop = 1/5,
                     ## `train` arguments:
                     method = "svmPoly",
                     metric = "Accuracy",
                     trControl = fitControl)
ggplot(Accuracy_data, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw()

trellis.par.set(caretTheme())
plot(model_knn) 
ggplot(model_dt)
#Close parallel calculation
#stopCluster(cl)
