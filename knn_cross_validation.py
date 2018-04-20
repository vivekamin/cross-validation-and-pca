import scipy.io as sio 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


knn_data = sio.loadmat('knn_data.mat')
trainData = np.array(knn_data['train_data'])
trainLabel = np.array(knn_data['train_label'])
testData = np.array(knn_data['test_data'])
testLabel = np.array(knn_data['test_label'])

trainLabel = trainLabel.reshape((trainLabel.shape[0],))
testLabel = testLabel.reshape((testLabel.shape[0],))
trainLabel[0:int(trainLabel.shape[0]/5)].shape


trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
i = 0
trainingData = np.zeros((4000,trainData.shape[1]))
accuracy_for_k = []
k_s = []
for l in range(0,9,1):
    k = ((2*l) + 1)
    total_accuracy = 0
    for x in range(0,5,1):
        knn = KNeighborsClassifier(n_neighbors=k)
        z = x * int(trainLabel.shape[0]/5)
        trainingData = trainData[z+int(trainLabel.shape[0]/5):]
        trainingLabel = trainLabel[z+int(trainLabel.shape[0]/5):]
        temp = trainData[0:int(z)]
        temp1 = trainLabel[0:int(z)]
        if(x > 0):
            trainingData = np.concatenate((trainingData,temp),axis = 0)
            trainingLabel = np.concatenate((trainingLabel,temp1),axis = 0)

        testData = trainData[z : z+int(trainLabel.shape[0]/5)]
        testLabel = trainLabel[z : z+int(trainLabel.shape[0]/5)]
        knn.fit(trainingData, trainingLabel)
        pred = knn.predict(testData)
        total_accuracy += accuracy_score(testLabel, pred)
    print('Average Train Accuracy for k=',k,'is', total_accuracy/5)
    k_s.append(k)
    accuracy_for_k.append(total_accuracy/5)
    
    

fig, ax = plt.subplots()
A = k_s
B = [ '%.5f' % elem for elem in accuracy_for_k ]

index = accuracy_for_k.index(max(accuracy_for_k))
k1 = A[index],
y1 = max(accuracy_for_k),
plt.plot(A,B)
for xy in zip(k1, y1):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',color='green') # <--
#plt.plot(k_s,accuracy_for_k)
plt.title("KNN Cross Validation")
plt.xlabel("K(Number of neighbours)")
plt.ylabel("Accuracy")
plt.show()

kclusters = A[index]
#print(y1)
# trainData = np.array(knn_data['train_data'])
# trainLabel = np.array(knn_data['train_label'])
testData = np.array(knn_data['test_data'])
testLabel = np.array(knn_data['test_label'])
trainLabel = trainLabel.reshape(5000,)
testLabel = testLabel.reshape(1598,)

#Accuracy on Test Data
knn = KNeighborsClassifier(n_neighbors=kclusters)
knn.fit(trainData, trainLabel)
pred = knn.predict(testData)
print ("Test Accuracy for  k=", kclusters,'is', accuracy_score(testLabel, pred))

