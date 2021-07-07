import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

#ohter library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#importing CSV file and getting info
df =pd.read_csv('heart.csv')
df.info()

#Plotting Correlation Matrix
f = plt.figure(figsize=(12,8))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()
#Plotting Histogram
df.hist()
plt.show()
#Spliting into training and testing dataset
y = df['target']
X = df.drop(['target'],axis =1)
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#Scaling dataset
standardScaler =StandardScaler()
columns_to_scale =['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

 #using Decision Tree Classifier
dt_scores =[]
for i in range(1,len(X.columns)+1):
    dt_classifier = DecisionTreeClassifier(max_features=i,random_state=0)
    dt_classifier.fit(X_train,y_train)
    dt_scores.append(dt_classifier.score(X_test,y_test))

#selected maximum number of features from 1to 13 for split
plt.plot([i for i in range(1,len(X.columns)+1)],dt_scores,color = "green")
for i in range(1,len(X.columns)+1) :
    plt.text(i,dt_scores[i-1],(i,dt_scores[i-1]))
    plt.xticks([i for i in range(1,len(X.columns)+1)])
    plt.xlabel('Max features')
    plt.ylabel('SCORES')
    plt.title('Decision Tree Calssifier scores for different number of maximum features')
plt.show()