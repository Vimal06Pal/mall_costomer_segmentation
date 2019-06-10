#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,3:5].values

#using dendogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendrogram =sch.dendrogram(sch.linkage(x,method='ward'))
plt.xlabel('customer')
plt.ylabel('euclien distance')
plt.title('dendrogram')
plt.show()

#fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#visualising the result
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='blue',label='careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='green',label='standard 1')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='red',label='target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='yellow',label='sensible')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='brown',label='careless')
plt.xlabel('age')
plt.ylabel('salary')
plt.title('hierarchy cluster')
plt.legend()
plt.show()
