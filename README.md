# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: Collect and preprocess the email dataset, converting text data into numerical features, often using techniques like TF-IDF.
2.Data Splitting: Split the dataset into training and testing sets to train and evaluate the model.
3.Model Training: Use a Support Vector Machine (SVM) classifier on the training set to learn to distinguish between spam and non-spam emails.
4.Model Evaluation: Test the trained model on the testing set and assess its accuracy in predicting spam versus non-spam emails.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sukhmeet Kaur G
RegisterNumber: 2305001032
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

#Extract features
X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![Screenshot 2024-11-06 192431](https://github.com/user-attachments/assets/8f53ebdd-4cc3-469e-8a43-6ade53ab04c5)
![Screenshot 2024-11-06 192446](https://github.com/user-attachments/assets/0efca472-599f-4252-98fd-08fded25a38e)
![Screenshot 2024-11-06 192457](https://github.com/user-attachments/assets/d1d843f6-61f7-4be2-9a83-6bd07e83bb63)
![Screenshot 2024-11-06 192519](https://github.com/user-attachments/assets/5a9bc410-dbd8-4bc4-ad07-ea775b008b96)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
