# Credit_Risk_Analysis

# Overview 
In this module we have applied machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we have employed different techniques to train and evaluate models with unbalanced classes. We have used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we have oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, we used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next,we have compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

## Resources
* Data: LoanStats_2019Q1.csv
* Software: jupyter notebook, python 3.7, Pandas, Collections, Numpy, sklearn, imblearn

## Analysis
Using imblearn and sklearn module we built a series of supervised machine learning models to assess and predict loan risk. Then we evaluated them to find the best fitting model.

## Results
Note: In the confusion matrix tables:

"0" implies high-risk loan
"1" implies low-risk loan

## Model 1: Naive Random Oversampling model
Model Parameters:
    * Model: imblearn RandomOverSampler
    * Random State: 1
Model Results:
    * Balanced Acurracy Score: 64%

   * Image 1: Naive Random Oversampling Confusion Matrix
 <img width="345" alt="image 1" src="https://user-images.githubusercontent.com/79999761/130372594-7308fd60-a01f-455e-ba36-d646cba6c810.png">

   * Image 1.2: Naive Random Oversampling Classification Report
 <img width="967" alt="naive oversampling classification" src="https://user-images.githubusercontent.com/79999761/130372600-16b5e8cb-c0a9-44f8-843b-d1a451f2db8c.png">
   
## Model 2: SMOTE (Synthetic Minority Oversampling Technique) Oversampling model
Model Parameters:
   * Model: imblearn SMOTE
   * Random State: 1
Model Results:
   * Balanced Acurracy Score: 66%
   
   Image 2.1: SMOTE Sampling Confusion Matrix
   
  <img width="342" alt="SMOTE oversampling" src="https://user-images.githubusercontent.com/79999761/130372729-8cbacb18-1b7a-49b5-83ea-706ed83b9872.png">
   
   Image 2.2: SMOTE Sampling Classification Report
   <img width="955" alt="SMOTE Class" src="https://user-images.githubusercontent.com/79999761/130372746-d9c35d3e-944d-4ca2-a874-588cd6ff7dca.png">
    

## Model 3: Cluster Centroids Undersampling model

Model Parameters:
    * Model: imblearn ClusterCentroids
    * Random State: 1
Model Results:
    * Balanced Acurracy Score: 54%
    
   Image 3.1: Cluster Centroids Sampling Confusion Matrix
   <img width="353" alt="Cluster Centroids Undersampling confusion" src="https://user-images.githubusercontent.com/79999761/130372756-3829a893-b8eb-4bad-a520-6ffff6dbd618.png">
   
   Image 3.2: Cluster Centroids Sampling Classification Report
   <img width="992" alt="Cluster Centroids Sampling Class" src="https://user-images.githubusercontent.com/79999761/130372764-8efd719d-c3d2-44c2-93ca-2322acf75c59.png">
   

## Model 4: SMOTEENN (SMOTE and Edited Nearest Neighbors) Combination Sampling model
Model Parameters:
   * Model: imblearn SMOTEENN
   * Random State: 1
Model Results:
   * Balanced Acurracy Score: 67%
   
   Image 4.1: SMOTEENN Sampling Confusion Matrix
   <img width="355" alt="SMOTEENN Confusion Matrix" src="https://user-images.githubusercontent.com/79999761/130372778-efe5b9fb-e6f6-4098-99e6-be045ae2b51e.png">
   
   Image 4.2: SMOTEENN Sampling Classification Report
   <img width="938" alt="SMOTEEN Classification" src="https://user-images.githubusercontent.com/79999761/130372782-a0b3f573-1623-440e-abee-053049384f2b.png">
   

## Model 5: Balanced Random Forest Classifier model
Model Parameters:
   * Model: imblearn BalanceRandomForestClassifier
   * Random State: 1
Model Results:
   * Balanced Acurracy Score: 79%
   
   Image 5.1: Balanced Random Forest Classifier Confusion Matrix
   <img width="342" alt="Balanced random Forest Confusion Matrix" src="https://user-images.githubusercontent.com/79999761/130372796-6daff3a2-58f3-4909-9608-65466aefdc5f.png">
   
   Image 5.2: Balanced Random Forest Classifier Classification Report
   <img width="931" alt="Balanced Random Forest Classifications" src="https://user-images.githubusercontent.com/79999761/130372802-07d774ba-aad2-4e3c-a800-0c50c53ab411.png">
   

## Model 6: Easy Ensemble AdaBoost Classifier model
Model Parameters:
   * Model: imblearn EasyEnsembleClassifier
   * Random State: 1
Model Results:
   * Balanced Acurracy Score: 93%
   
   Image 6.1: Easy Ensemble AdaBoost Classifier Confusion Matrix
   <img width="351" alt="Easy Ensemble AdaBoost Classifier Confusion Matrix" src="https://user-images.githubusercontent.com/79999761/130372826-cf4f89dd-9bac-46df-80e8-87eec13ec8ed.png">
   
   Image 6.2: Easy Ensemble AdaBoost Classifier Classification Report
   <img width="953" alt="Easy Ensemble AdaBoost Classifications" src="https://user-images.githubusercontent.com/79999761/130372830-26a6beea-4e85-43d5-a6d4-0257313b0ee4.png">
   

## Conclusion

Evaluating the above supervised machine learning models to evaluate credit card risk, we can conclude that the most accurate model was the Easy Ensemble AdaBoost Classifier model using a linear regression fit with 100 estimators. The Easy Ensemble AdaBoost Classifier model had the highest balanced accuracy score with 93%. This is compared to the next highest score for the Balanced Random Forest Classifier model at 79%. Additionally, the EEC model also performed well in prediction, recall, and f1 score for high-risk loans. Specifically for loan-risk evaluation, false positives, or wrongly predicting low risk ("1") when the actual loan is high-risk ("0"), is the prediction needed to be most accurate and the best performing model for this was also the EEC model.

## Limitations of current models

To tune the models to produce more accurate predictions, we can add scaling to some of the data to eliminate more of the modeling variability using the StandardScaler() module.

Using the Balanced Random Forest Classifier feature importances ranking list we can start to eliminate columns that may be less impactful on modeling predictions and therefore reduce the variability of the modeling.

Additionally, can add more estimators or change the type of model fit to something other than linear regression to see if it improves the model's accuracy.
