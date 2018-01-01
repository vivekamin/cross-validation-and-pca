import scipy as sc
import pandas as pd
import scipy.io as sio 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

readFile = sio.loadmat('knn_data.mat')

knnData = np.array(readFile['train_data'])
knnLabel = np.array(readFile['train_label'])

knnTestData = np.array(readFile['test_data'])
knnTestLabel = np.array(readFile['test_label'])

def PCA(knnData1):
    X_std = StandardScaler().fit_transform(knnData1)



    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    cov_mat.shape
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    cor_mat1 = np.corrcoef(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    cor_mat2 = np.corrcoef(knnData1.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat2)
    u,s,v = np.linalg.svd(X_std.T)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = np.hstack((eig_pairs[0][1].reshape(166,1),eig_pairs[1][1].reshape(166,1)));
    for i in range(2,50,1):
        matrix_w = np.hstack((matrix_w,eig_pairs[i][1].reshape(166,1)));
    return X_std.dot(matrix_w)


transformed = PCA(knnData)

transformed, knnLabel = shuffle(transformed, knnLabel, random_state=0)

i = 0
trainingData = np.zeros((4000,transformed.shape[1]))
accuracy_for_k = []
k_s = []
for l in range(0,9,1):
    k = ((2*l) + 1)
    total_accuracy = 0
    for x in range(0,5,1):
        knn = KNeighborsClassifier(n_neighbors=k)
        z = x * int(knnLabel.shape[0]/5)
        trainingData = transformed[z+int(knnLabel.shape[0]/5):]
        trainingLabel = knnLabel[z+int(knnLabel.shape[0]/5):]
        temp = transformed[0:int(z)]
        temp1 = knnLabel[0:int(z)]
        if(x > 0):
            trainingData = np.concatenate((trainingData,temp),axis = 0)
            trainingLabel = np.concatenate((trainingLabel,temp1),axis = 0)

        testData = transformed[z : z+int(knnLabel.shape[0]/5)]
        testLabel = knnLabel[z : z+int(knnLabel.shape[0]/5)]
        knn.fit(trainingData, trainingLabel)
        pred = knn.predict(testData)
        total_accuracy += accuracy_score(testLabel, pred)
    print('Average Accuracy for ',k, total_accuracy/5)
    k_s.append(k)
    accuracy_for_k.append(total_accuracy/5)


fig, ax = plt.subplots()
A = k_s
B = [ '%.5f' % elem for elem in accuracy_for_k ]

index = accuracy_for_k.index(max(accuracy_for_k))
k1 = A[index],
y1 = max(accuracy_for_k),
plt.plot(A,B)
for xy in zip(k1, y1):
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',color='green')
plt.title("KNN Cross Validation with PCA")
plt.xlabel("K(Number of neighbours)")
plt.ylabel("Accuracy")
plt.show()

k_clusters = A[index]
print(k_clusters)
transformed_test = PCA(knnTestData)
knn = KNeighborsClassifier(n_neighbors=k_clusters)
knn.fit(transformed, knnLabel)
pred = knn.predict(transformed_test)
print (accuracy_score(knnTestLabel, pred))
