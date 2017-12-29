# cross-validation-and-pca

### The dataset has 5,000 instances and 1,598 instances in the training set and test set, and each instance has 166 features
    knn_data.mat
    K = { k|2l+1, l = 0, 1, · · · , 8}
   ### Cross Validation
   Implemented 5-fold cross validation for kNN and plotted the average accuracy on the validation set vs. each possible k ∈ K. Chose the best parameter based on these accuracies and use it to predict on the test data. 
   
   Reported the parameter k=5 from above experiment and the corresponding accuracy on the test data.
   
   ### PCA
   Implemented PCA to reduce the dimensionality of the data from 166 to 50.Then on this new reduced dataset, implemented 5-fold cross validation for kNN and plotted the average accuracy on the validation set vs. each possible k ∈ K.
   
   
