### Background

This is project is a model prediction assignment. The goal is to use
data from six participants from accelerometers on the belt, forearm,
arm, and dumbell to predict the manner of the exercise: if the barbell
lifts were performed correctly and incorrectly in the following manner:

-   Exactly according to the specification (Class A)
-   Throwing the elbows to the front (Class B)
-   Lifting the dumbbell only halfway (Class C)
-   Lowering the dumbbell only halfway (Class D)
-   Throwing the hips to the front (Class E)

More information is available in the section <b>Weight Lifting Exercise
Dataset</b> from the website here:
<code><http://groupware.les.inf.puc-rio.br/har></code>.

The class assignment is located :
<code><https://www.coursera.org/learn/practical-machine-learning/peer/R43St/prediction-assignment-writeup></code>

### Overview

Create a report describing how the model was built, how cross validation
was used, what the expected out of sample error is, and why the choices
were made. Also use the prediction model to predict 20 different test
cases.

### Executive Summary

I downloaded the training and test data sets, performed some exploratory
analysis to get a feel for the data, and cleaned up the data to have a
managable set of predictors. Using the <code>caret</code> and
<code>randomForest</code> packages, I trained a <b>Random Forest</b>
model to predict the variable <b>classe</b>. The final accuracy of the
model was 98.99%, and the model predicted the 20 test variables
accurately.

### The Data

Load the libraries; download the training and test datasets

    library(caret)
    library(ggplot2)
    library(randomForest)

    if (!file.exists("data")) {
        dir.create("data")
    }

    trainfile <- "pml-training.csv"
    testfile <- "pml-testing.csv"

    if (!file.exists(trainfile)) {
         fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
         download.file(fileUrl, destfile = "data/pml-training.csv", method = "libcurl")
    }

    if (!file.exists(testfile)) {
         fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
         download.file(fileUrl, destfile = "data/pml-testing.csv", method = "libcurl")
    }

### Load the Data / Tidy the Data

    traindata <- read.csv("data/pml-training.csv")
    testdata <- read.csv("data/pml-testing.csv")

Count which columns have missing values

    na_count <-sapply(traindata, function(y) sum(is.na(y)))
    data.frame(na_count)

    ##                          na_count
    ## X                               0
    ## user_name                       0
    ## raw_timestamp_part_1            0
    ## raw_timestamp_part_2            0
    ## cvtd_timestamp                  0
    ## new_window                      0
    ## num_window                      0
    ## roll_belt                       0
    ## pitch_belt                      0
    ## yaw_belt                        0
    ## total_accel_belt                0
    ## kurtosis_roll_belt              0
    ## kurtosis_picth_belt             0
    ## kurtosis_yaw_belt               0
    ## skewness_roll_belt              0
    ## skewness_roll_belt.1            0
    ## skewness_yaw_belt               0
    ## max_roll_belt               19216
    ## max_picth_belt              19216
    ## max_yaw_belt                    0
    ## min_roll_belt               19216
    ## min_pitch_belt              19216
    ## min_yaw_belt                    0
    ## amplitude_roll_belt         19216
    ## amplitude_pitch_belt        19216
    ## amplitude_yaw_belt              0
    ## var_total_accel_belt        19216
    ## avg_roll_belt               19216
    ## stddev_roll_belt            19216
    ## var_roll_belt               19216
    ## avg_pitch_belt              19216
    ## stddev_pitch_belt           19216
    ## var_pitch_belt              19216
    ## avg_yaw_belt                19216
    ## stddev_yaw_belt             19216
    ## var_yaw_belt                19216
    ## gyros_belt_x                    0
    ## gyros_belt_y                    0
    ## gyros_belt_z                    0
    ## accel_belt_x                    0
    ## accel_belt_y                    0
    ## accel_belt_z                    0
    ## magnet_belt_x                   0
    ## magnet_belt_y                   0
    ## magnet_belt_z                   0
    ## roll_arm                        0
    ## pitch_arm                       0
    ## yaw_arm                         0
    ## total_accel_arm                 0
    ## var_accel_arm               19216
    ## avg_roll_arm                19216
    ## stddev_roll_arm             19216
    ## var_roll_arm                19216
    ## avg_pitch_arm               19216
    ## stddev_pitch_arm            19216
    ## var_pitch_arm               19216
    ## avg_yaw_arm                 19216
    ## stddev_yaw_arm              19216
    ## var_yaw_arm                 19216
    ## gyros_arm_x                     0
    ## gyros_arm_y                     0
    ## gyros_arm_z                     0
    ## accel_arm_x                     0
    ## accel_arm_y                     0
    ## accel_arm_z                     0
    ## magnet_arm_x                    0
    ## magnet_arm_y                    0
    ## magnet_arm_z                    0
    ## kurtosis_roll_arm               0
    ## kurtosis_picth_arm              0
    ## kurtosis_yaw_arm                0
    ## skewness_roll_arm               0
    ## skewness_pitch_arm              0
    ## skewness_yaw_arm                0
    ## max_roll_arm                19216
    ## max_picth_arm               19216
    ## max_yaw_arm                 19216
    ## min_roll_arm                19216
    ## min_pitch_arm               19216
    ## min_yaw_arm                 19216
    ## amplitude_roll_arm          19216
    ## amplitude_pitch_arm         19216
    ## amplitude_yaw_arm           19216
    ## roll_dumbbell                   0
    ## pitch_dumbbell                  0
    ## yaw_dumbbell                    0
    ## kurtosis_roll_dumbbell          0
    ## kurtosis_picth_dumbbell         0
    ## kurtosis_yaw_dumbbell           0
    ## skewness_roll_dumbbell          0
    ## skewness_pitch_dumbbell         0
    ## skewness_yaw_dumbbell           0
    ## max_roll_dumbbell           19216
    ## max_picth_dumbbell          19216
    ## max_yaw_dumbbell                0
    ## min_roll_dumbbell           19216
    ## min_pitch_dumbbell          19216
    ## min_yaw_dumbbell                0
    ## amplitude_roll_dumbbell     19216
    ## amplitude_pitch_dumbbell    19216
    ## amplitude_yaw_dumbbell          0
    ## total_accel_dumbbell            0
    ## var_accel_dumbbell          19216
    ## avg_roll_dumbbell           19216
    ## stddev_roll_dumbbell        19216
    ## var_roll_dumbbell           19216
    ## avg_pitch_dumbbell          19216
    ## stddev_pitch_dumbbell       19216
    ## var_pitch_dumbbell          19216
    ## avg_yaw_dumbbell            19216
    ## stddev_yaw_dumbbell         19216
    ## var_yaw_dumbbell            19216
    ## gyros_dumbbell_x                0
    ## gyros_dumbbell_y                0
    ## gyros_dumbbell_z                0
    ## accel_dumbbell_x                0
    ## accel_dumbbell_y                0
    ## accel_dumbbell_z                0
    ## magnet_dumbbell_x               0
    ## magnet_dumbbell_y               0
    ## magnet_dumbbell_z               0
    ## roll_forearm                    0
    ## pitch_forearm                   0
    ## yaw_forearm                     0
    ## kurtosis_roll_forearm           0
    ## kurtosis_picth_forearm          0
    ## kurtosis_yaw_forearm            0
    ## skewness_roll_forearm           0
    ## skewness_pitch_forearm          0
    ## skewness_yaw_forearm            0
    ## max_roll_forearm            19216
    ## max_picth_forearm           19216
    ## max_yaw_forearm                 0
    ## min_roll_forearm            19216
    ## min_pitch_forearm           19216
    ## min_yaw_forearm                 0
    ## amplitude_roll_forearm      19216
    ## amplitude_pitch_forearm     19216
    ## amplitude_yaw_forearm           0
    ## total_accel_forearm             0
    ## var_accel_forearm           19216
    ## avg_roll_forearm            19216
    ## stddev_roll_forearm         19216
    ## var_roll_forearm            19216
    ## avg_pitch_forearm           19216
    ## stddev_pitch_forearm        19216
    ## var_pitch_forearm           19216
    ## avg_yaw_forearm             19216
    ## stddev_yaw_forearm          19216
    ## var_yaw_forearm             19216
    ## gyros_forearm_x                 0
    ## gyros_forearm_y                 0
    ## gyros_forearm_z                 0
    ## accel_forearm_x                 0
    ## accel_forearm_y                 0
    ## accel_forearm_z                 0
    ## magnet_forearm_x                0
    ## magnet_forearm_y                0
    ## magnet_forearm_z                0
    ## classe                          0

The columns with min, max, var, avg, stddev, amp, kurt, and skew all
have missing values. They also appear to be calculations of raw data.

I remove the columns/data from the test and training data sets with id
variables, timestamp info, and have missing values. This leaves 52
features to run the model on the predictor variable <code>classe</code>.

    traindata.tidy <- traindata[, -grep("X|timestamp|window|user_name|min|max|var|avg|stddev|amp|kurt|skew", colnames(traindata))]

    testdata.tidy <- testdata[, -grep("X|timestamp|window|user_name|min|max|var|avg|stddev|amp|kurt|skew", colnames(traindata))]

### Slice the Training Data

I divided the data into a training and test set. I used a 60/40 split
because a training set split of 60% is as high as I could go to increase
the accuracy without killing my old laptop's memory.

    library(caret)
    set.seed(1565)
    inTrain <- createDataPartition(traindata.tidy$classe, p=0.6, list=FALSE)
    training <- traindata.tidy[inTrain,]
    testing <- traindata.tidy[-inTrain,]

    dim(training); dim(testing)

    ## [1] 11776    53

    ## [1] 7846   53

### Model the Data

Using the caret and randomforest packages, I trained the model using a
random forest with 5 fold cross-validation. I played with some different
levels of CV. Five-fold gave the better accuracy than three-fold. But
the higher I went, the more memory was being used and the model ended up
puking on my slow laptop. The Random Forest model was used because it is
highly accurate, but slow and memory hogging.

    model.RF <- train(classe~.,data=training,method="rf",trControl=trainControl(method="cv",number=5),prox=TRUE)

#### Print the model data

    model.RF

    ## Random Forest 
    ## 
    ## 11776 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 9420, 9421, 9420, 9421, 9422 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD   
    ##    2    0.9889610  0.9860346  0.0025100042  0.003174630
    ##   27    0.9889607  0.9860339  0.0015881545  0.002011035
    ##   52    0.9841202  0.9799101  0.0009306364  0.001177863
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 2.

Model iteration <code>mtry =2</code> is the optimal model.

#### Print the best model

    model.RF$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.85%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3347    0    0    0    1 0.0002986858
    ## B   18 2253    8    0    0 0.0114085125
    ## C    0   15 2035    4    0 0.0092502434
    ## D    0    0   43 1886    1 0.0227979275
    ## E    0    0    4    6 2155 0.0046189376

The final model had 500 trees.

#### Train the model

    predict.Rf <- predict(model.RF, testing)
    confusionMatrix(testing$classe, predict.Rf)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232    0    0    0    0
    ##          B   15 1498    5    0    0
    ##          C    0   14 1353    1    0
    ##          D    0    0   35 1251    0
    ##          E    0    0    2    7 1433
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9899         
    ##                  95% CI : (0.9875, 0.992)
    ##     No Information Rate : 0.2864         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9873         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9933   0.9907   0.9699   0.9936   1.0000
    ## Specificity            1.0000   0.9968   0.9977   0.9947   0.9986
    ## Pos Pred Value         1.0000   0.9868   0.9890   0.9728   0.9938
    ## Neg Pred Value         0.9973   0.9978   0.9935   0.9988   1.0000
    ## Prevalence             0.2864   0.1927   0.1778   0.1605   0.1826
    ## Detection Rate         0.2845   0.1909   0.1724   0.1594   0.1826
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9967   0.9938   0.9838   0.9942   0.9993

The model has Accuracy of 98.99% with a 95% CI of (0.9875, 0.992). The
out-of-sample error rate is 1.01%.

### Use the Test Data Set

I predict the model on the downloaded test data set of 20 observations
and print the results.

    quiz <- predict(model.RF, testdata.tidy[, -length(names(testdata.tidy))])
