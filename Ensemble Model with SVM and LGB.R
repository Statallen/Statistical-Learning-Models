library(tidyverse)
library(knitr)
library(MASS)
library(kernlab)
library(e1071)
library(caret)
library(bmrm)
library(umap)
library(gbm)
library(OpenImageR)
library(glmnet)
library(scales)
library(lightgbm)

# Data
train<- read_csv("C:\\Users\\absur\\Desktop\\Pis\\[2] UIUC\\[1] Courses\\[3] Spring 2022\\Stat 542\\Project\\fashion-mnist_train.csv")
test<- read_csv("C:\\Users\\absur\\Desktop\\Pis\\[2] UIUC\\[1] Courses\\[3] Spring 2022\\Stat 542\\Project\\fashion-mnist_test.csv")

# Frequency Table & summary statistics
kable(t(table(train[,1])),caption = "Training Data")
kable(t(table(test[,1])),caption = "Testing Data")

# Dimension Reduction
## Training Data
### Dimension Reduced Data - PCA
pcafit<- prcomp(train[,-1],center = T,scale. = F,rank. = 20)

train_reduced<- cbind(train[,1],pcafit$x) #20 components
train_reduced[,1]<- as.factor(train_reduced[,1])

train_reduced_1<- cbind(train[,1],pcafit$x[,1:2]) # 2 components for comparison
train_reduced_1[,1]<- as.factor(train_reduced_1[,1])

#### Standard Deviation for components & Cumulative plot
train_pc<- prcomp(train[,-1], scale = FALSE, center = TRUE) 
plot(train_pc, type = "l", pch = 19) # elbow rule

pca_plot_1<- data.frame(stdv=summary(train_pc)[[6]][1,1:30],components=1:30)
ggplot(data = pca_plot_1)+
  geom_line(aes(x=as.factor(components),y=stdv,group=1))+
  geom_point(aes(x=as.factor(components),y=stdv))+
  labs(x="Principal Components",y="Standard Deviation",title = "Principle Component Analysis")

pca_plot_1<- data.frame(cum=summary(train_pc)[[6]][3,1:30],components=1:30)
ggplot(data = pca_plot_1)+
  geom_line(aes(x=as.factor(components),y=cum,group=1))+
  geom_point(aes(x=as.factor(components),y=cum))+
  labs(x="Principal Components",y="CUmulative Proportion",title = "Principle Component Analysis")

#### Separation Visualization in 2D
ggplot(data=train_reduced)+
  geom_point(aes(x=train_reduced$PC1,y=train_reduced$PC2,color=train_reduced$label))+
  labs(x="",y="",title = "Principle Component Analysis")

ggplot(data=train_reduced)+
  geom_point(aes(x=train_reduced$PC3,y=train_reduced$PC1,color=train_reduced$label))+
  labs(x="",y="",title = "Principle Component Analysis")

## Testing Data - projection to training components
### PCA - 20 components
pcafit_test<- predict(pcafit,test) # projection on the principle vectors

test_reduced<- cbind(test[,1],pcafit_test) # 20 components
test_reduced[,1]<- as.factor(test_reduced[,1])

# splitting data set
## PCA 20 components
set.seed(542)
index<-sample(1:60000) # random splitting into 15 subset
subset<- list()
for (i in 1:15){
  subset[[i]]<- train_reduced[index[((i-1)*4000+1):(i*4000)],]
}

###############################################################################
###############################################################################

# training 15 smaller SVM binary models
sigest(label~.,data=train_reduced) #estimated sigma to choose from

fit_svm_pca<- list()

set.seed(542)
for (i in 1:length(subset)){
  ## this will take 3 hrs
  fit_svm_pca[[i]]<- train(label~., data = subset[[i]], method = "svmRadial",
                           preProcess = c("center", "scale"),
                           tuneGrid = expand.grid(C = seq(1,12,length.out=6), sigma = seq(0.01,0.06,length.out=6)),
                           trControl = trainControl(method = "cv", number = 5))
}

# tuning variable & error
tuning_svm_pca<- matrix(0,ncol=15,nrow = 3)
for (i in 1:15){
  tuning_svm_pca[,i]<- c(as.numeric(fit_svm_pca[[i]]$bestTune), percent(max(fit_svm_pca[[i]]$results$Accuracy),0.01))
}

tuning_svm_pca<- as.data.frame(tuning_svm_pca)
rownames(tuning_svm_pca)<- c("Sigma","C","Error")
colnames(tuning_svm_pca)<- 1:15

kable(tuning_svm_pca, caption = "SVM Tuning Variables")

# training visualization
training_svm_results<- list()
temp_dataframe<- data.frame(grid=1:36,accuracy=0,model=0)

for (i in 1:15){
  temp_dataframe$accuracy<- fit_svm_pca[[i]]$results$Accuracy
  temp_dataframe$model<-rep(i,36)
  training_svm_results[[i]]<- temp_dataframe
}

training_svm_results_dataframe<- matrix(0,ncol = 3,nrow = 1)
colnames(training_svm_results_dataframe)<- c("grid","accuracy","model")
for (i in 1:15) {
  training_svm_results_dataframe<- rbind(training_svm_results_dataframe,training_svm_results[[i]])
}
training_svm_results_dataframe<-training_svm_results_dataframe[-1,]
training_svm_results_dataframe<- as.data.frame(training_svm_results_dataframe)

ggplot(training_svm_results_dataframe,aes(x=factor(grid),y=1-accuracy,group=factor(model),color=factor(model)))+
  geom_line()+
  geom_point()+
  labs(x="Tuning Combination",y="Error",title = "SVM Mis-classification Rate for Each Classifier",color="Classifier")

# Predicted Labels from 15 SVMs
Predicted_label_svm_pca<- matrix(0,ncol = 15,nrow = 10000)
for (i in 1:15){
  Predicted_label_svm_pca[,i]<-predict(fit_svm_pca[[i]],newdata=test_reduced)
}


# Voted Labels - Majority Vote
Modes <- function(x) {
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  ux[tab == max(tab)]
}

voted_label_svm_pca<- apply(Predicted_label_svm_pca, 1, Modes)

for (i in 1:10000){
  if (length(voted_label_svm_pca[[i]])>1){
    #randomly break the ties
    set.seed(542)
    voted_label_svm_pca[[i]]<- sample(voted_label_svm_pca[[i]],1)
  }
}

voted_label_svm_pca<- unlist(voted_label_svm_pca)

# Accuracy
## training
accuracy_training_svm_pca<- matrix(0,ncol = 15,nrow = 1)
for (i in 1:15){
  accuracy_training_svm_pca[1,i]<- fit_svm_pca[[i]]$results$Accuracy %>% max() %>% round(digits = 2)
}

## testing
accuracy_testing_svm_pca<- matrix(0,ncol = 15,nrow = 1)
for (i in 1:15){
  accuracy_testing_svm_pca[1,i]<- percent(table(Predicted_label_svm_pca[,i],test_reduced[,1]) %>% diag() %>% sum()/10000, 0.01)
}
colnames(accuracy_testing_svm_pca)<- 1:15

svm_pca_results<- rbind(tuning_svm_pca,accuracy_testing_svm_pca)
rownames(svm_pca_results)<- c("Sigma","C","Training","Testing")
kable(svm_pca_results,caption = "SVM Results")

## voted 
voted_label_svm_pca_relabelled<- factor(voted_label_svm_pca-1,levels = levels(test_reduced[,1]))

table(voted_label_svm_pca_relabelled,test_reduced[,1])
# table(voted_label_svm_pca,test_reduced[,1])
voted_label_svm_pca_accuracy<- percent(round((table(voted_label_svm_pca_relabelled,test_reduced[,1]) %>% diag() %>% sum())/10000,4),0.01)

svm_pca_results<- cbind(svm_pca_results,c(NA,NA,NA,voted_label_svm_pca_accuracy))
colnames(svm_pca_results)[16]<- "Combine"
kable(svm_pca_results[1:8],caption = "SVM Results-1")
kable(svm_pca_results[9:16],caption = "SVM Results-2")

### Confusion Matrix 
cm_svm_pca <- confusionMatrix(factor(voted_label_svm_pca_relabelled), factor(test_reduced[,1]), dnn = c("Prediction", "Truth"))
result_svm_pca <- as.data.frame(cm_svm_pca$table)
result_svm_pca$Prediction <- factor(result_svm_pca$Prediction, levels=rev(levels(result_svm_pca$Prediction)))

labels_name<- c("T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")
ggplot(result_svm_pca, aes(Prediction,Truth, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="darkorange") +
  labs(x = "Prediction",y = "Truth")+
  scale_y_discrete(limits = levels(result_svm_pca$Prediction),labels=rev(labels_name))+
  scale_x_discrete(limits = rev(levels(result_svm_pca$Prediction)),labels=labels_name)+
  labs(title = "SVM with PCA - 86.24% Accuracy")

###############################################################################
###############################################################################

# KNN
## PCA
fit_knn_pca<- train(label ~ ., method = "knn", 
                    preProcess = c("center", "scale"),
                    data = train_reduced,
                    tuneGrid = data.frame(k = seq(1, 25, 1)),
                    trControl = trainControl(method = "cv", number = 5))

accuracy_training_knn_pca<- fit_knn_pca$results$Accuracy %>% max() %>% round(digits = 4)
predicted_lable_knn_pca<- predict(fit_knn_pca,newdata=test_reduced[,-1])
accuracy_testing_knn_pca<- percent(table(predicted_lable_knn_pca,test_reduced[,1]) %>% diag() %>% sum()/10000,0.01)

### Results
knn_pca_results<- data.frame("Training Accuracy"=accuracy_training_knn_pca,K=7,"Testing Accuracy"=accuracy_testing_knn_pca)
kable(knn_pca_results,caption = "KNN Classification Results")

### Training Error
knn_pca_missclassification<- data.frame(K=1:25, Mis_class=1-fit_knn_pca$results$Accuracy)
ggplot(knn_pca_missclassification)+
  geom_line(aes(x=factor(K),y=Mis_class,group=1),color="darkorange")+
  geom_point(aes(x=factor(K),y=Mis_class,group=1),color="darkorange")+
  labs(x="K-Neighbor",y="Misclassification Rate",title = "KNN Training Error")

### Confusion Matrix 
cm_knn_pca <- confusionMatrix(factor(predicted_lable_knn_pca), factor(test_reduced[,1]), dnn = c("Prediction", "Truth"))
result_knn_pca <- as.data.frame(cm_knn_pca$table)
result_knn_pca$Prediction <- factor(result_knn_pca$Prediction, levels=rev(levels(result_knn_pca$Prediction)))

labels_name<- c("T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")
ggplot(result_knn_pca, aes(Prediction,Truth, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="darkorange") +
  labs(x = "Prediction",y = "Truth")+
  scale_y_discrete(limits = levels(result_knn_pca$Prediction),labels=rev(labels_name))+
  scale_x_discrete(limits = rev(levels(result_knn_pca$Prediction)),labels=labels_name)+
  labs(title = "KNN with PCA - 85.45% Accuracy")
################################################################################
################################################################################

# Ensemble
## Specific Group
## 0, 2, 4, 6
train_reduced_specific<- cbind(train[,1],pcafit$x)
train_reduced_specific<- filter(train_reduced,label==0|label==2|label==4|label==6)
train_reduced_specific[,1]<- factor(train_reduced_specific[,1],levels = c("0","2","4","6"))

test_reduced_specific<- cbind(test[,1],pcafit_test)
test_reduced_specific<-filter(test_reduced_specific,label==0|label==2|label==4|label==6)
test_reduced_specific[,1]<- factor(test_reduced_specific[,1],levels = c("0","2","4","6"))

### Light GB 0,2,4,6
train_reduced_lgb<- train_reduced_specific
levels(train_reduced_lgb[,1])<- c(0,1,2,3)
train_reduced_lgb[,1]<- as.numeric(levels(train_reduced_lgb[,1]))[train_reduced_lgb[,1]] 
train_reduced_matrix<- as.matrix(train_reduced_lgb)

test_reduced_lgb<- test_reduced_specific
levels(test_reduced_lgb[,1])<- c(0,1,2,3)
test_reduced_lgb[,1]<- as.numeric(levels(test_reduced_lgb[,1]))[test_reduced_lgb[,1]] 
test_reduced_matrix<- as.matrix(test_reduced_lgb)

lgb_data_train<- lgb.Dataset(train_reduced_matrix[,-1],label = train_reduced_matrix[,1])
lgb_data_test<- lgb.Dataset(lgb_data_train,data=test_reduced_matrix[,-1],label = test_reduced_matrix[,1])

### Tuning
max_depth<- 5:12
num_leaf<- seq(from=20, to=160, by=20)
min_data_in_leaf<- seq(from=160, to=20, by=-20)

lgb_traing_results<- data.frame(index=1:512,depth=0,num_leaf=0,min_data=0,error=0)
for (i in 1:length(max_depth)) {
  for (j in 1:length(num_leaf)){
    for (k in 1:length(min_data_in_leaf)){
      current_position<- (i-1)*64 + (j-1)*8 + k
      lgb_traing_results[current_position,2:4]=c(i,j,k)
      params <- list(objective = "multiclass", num_class=4L ,alpha = 0.1, 
                     boosting = "gbdt",shrinkage_rate = 0.1,
                     max_depth=max_depth[i], 
                     num_leaf=num_leaf[j],
                     min_data_in_leaf=min_data_in_leaf[k],
                     num_iterations=375)
      model_CV_training<- lgb.cv(params = params, data=lgb_data_train,nrounds = 30L,
                                 nfold = 5L,eval="multi_error",early_stopping_round=25)
      lgb_traing_results[current_position,5]<-model_CV_training$best_score
    }
  }
  
}

max_depth_1<- 13:16
num_leaf_1<- seq(from=180, to=240, by=20)

lgb_traing_results_1<- data.frame(index=1:16,depth=0,num_leaf=0,min_data=0,error=0)
for (i in 1:length(max_depth_1)) {
  for (j in 1:length(num_leaf_1)){
    current_position<- (i-1)*4 + j
    lgb_traing_results_1[current_position,2:3]=c(i,j)
    params <- list(objective = "multiclass", num_class=4L ,alpha = 0.1, 
                   boosting = "gbdt",shrinkage_rate = 0.1,
                   max_depth=max_depth_1[i], 
                   num_leaf=num_leaf_1[j],
                   num_iterations=375)
    model_CV_training<- lgb.cv(params = params, data=lgb_data_train,nrounds = 30L,
                               nfold = 5L,eval="multi_error",early_stopping_round=25)
    lgb_traing_results_1[current_position,5]<-model_CV_training$best_score
  }
  
}

lgb_tuning_results<- rbind(lgb_traing_results,lgb_traing_results_1)
lgb_tuning_results$index<- 1:528

best_tuning_results<-lgb_tuning_results[526,-1]
rownames(best_tuning_results)<- ""
colnames(best_tuning_results)<-c("Depth","Leaf","Min_data","Accuracy")
best_tuning_results[1,]<- c(16,200,20,percent(1-best_tuning_results[1,4],0.01))
kable(best_tuning_results,caption = "Gradient Boosting Tuning Result")

## Visualization of Tuning
ggplot(lgb_tuning_results)+
  geom_line(aes(x=index,y=error),color="darkorange")+
  geom_point(aes(x=index,y=error),color="darkorange")+
  labs(x="Tuning Combination",y="Mis-classification Rate",title = "Gradient Boosting Training Error")


## Testing using actual test data set
params <- list(objective = "multiclass", num_class=4L ,alpha = 0.1, 
               boosting = "gbdt",shrinkage_rate = 0.1,min_data_in_leaf=50, 
               num_iterations=500, max_depth=10, num_leaf=100)

params_optimized<- list(objective = "multiclass", num_class=4L ,alpha = 0.1, 
                        boosting = "gbdt",shrinkage_rate = 0.1,
                        max_depth=max_depth[8], 
                        num_leaf=num_leaf[8],
                        min_data_in_leaf=min_data_in_leaf[8],
                        num_iterations=375)

params_optimized_1<-list(objective = "multiclass", num_class=4L ,alpha = 0.1, 
                         boosting = "gbdt",shrinkage_rate = 0.1,
                         max_depth=16, 
                         num_leaf=200,
                         num_iterations=375)

# model <- lgb.train(params = params, data=lgb_data_train)
# model_optimized <- lgb.train(params = params_optimized, data=lgb_data_train)
# model_CV<- lgb.cv(params = params, data=lgb_data_train,nrounds = 30L,nfold = 5L,eval="multi_error",early_stopping_round=50)

model_optimized_1<- lgb.train(params = params_optimized_1, data=lgb_data_train)

predicted_probability<- predict(model_optimized_1,test_reduced_matrix[,-1]) 
predicted_probability_matrix<-matrix(predicted_probability,ncol = 4,byrow = T)
predicted_label_lgb<- apply(predicted_probability_matrix, MARGIN = 1, FUN=which.max)
predicted_label_lgb<- as.factor(predicted_label_lgb)
levels(predicted_label_lgb)<- c(0,2,4,6)

table(predicted_label_lgb,test_reduced_specific[,1]) %>% diag() %>% sum()/ 4000


### Testing for two-stage data
#### Stage 1
stage_1_predicted<-cbind(voted_label_svm_pca_relabelled,test_reduced[,-1])
colnames(stage_1_predicted)[1]<- "predicted_label"
stage_1_predicted_specific_data<- filter(stage_1_predicted,predicted_label==0|predicted_label==2|predicted_label==4|predicted_label==6)
stage_1_predicted_specific_data[,1]<- factor(stage_1_predicted_specific_data[,1],levels = c("0","2","4","6"))

#### Stage 2
#### Light GBM
stage_2_predicted_prob_lgb_specific<- predict(model_optimized_1,as.matrix(stage_1_predicted_specific_data[,-1]))
stage_2_predicted_prob_matrix<-matrix(stage_2_predicted_prob_lgb_specific,ncol = 4,byrow = T)
stage_2_predicted_label_lgb_specific<- apply(stage_2_predicted_prob_matrix, MARGIN = 1, FUN=which.max)
stage_2_predicted_label_lgb_specific<- as.factor(stage_2_predicted_label_lgb_specific)
levels(stage_2_predicted_label_lgb_specific)<- c(0,2,4,6)

index<- rep(0,10000)
numeric_label<- as.numeric(levels(stage_1_predicted[,1]))[stage_1_predicted[,1]] 
for (i in 1:length(numeric_label)){
  if (numeric_label[i]== 0|numeric_label[i]==2|numeric_label[i]==4|numeric_label[i]==6){
    index[i]<-TRUE
  }else{
    index[i]<-FALSE
  }
}

stage_2_predicted_lgb<- stage_1_predicted
stage_2_predicted_lgb[which(index==1),1]<- stage_2_predicted_label_lgb_specific

table(stage_2_predicted_lgb[,1],test_reduced[,1]) %>% diag() %>% sum()

### Confusion Matrix

cm_lgb_pca <- confusionMatrix(factor(stage_2_predicted_lgb[,1]), factor(test_reduced[,1]), dnn = c("Prediction", "Truth"))
result_lgb_pca <- as.data.frame(cm_lgb_pca$table)
result_lgb_pca$Prediction <- factor(result_lgb_pca$Prediction, levels=rev(levels(result_lgb_pca$Prediction)))

labels_name<- c("T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")
ggplot(result_lgb_pca, aes(Prediction,Truth, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="darkorange") +
  labs(x = "Prediction",y = "Truth")+
  scale_y_discrete(limits = levels(result_knn_pca$Prediction),labels=rev(labels_name))+
  scale_x_discrete(limits = rev(levels(result_knn_pca$Prediction)),labels=labels_name)+
  labs(title = "Ensemble Model- SVM & Gradient Boosting - 86.71% Accuracy")

