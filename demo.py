import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
dataset =pd.read_csv('heart.csv')
dataset.info()
dataset.describe()
plt.show()

#Plotting Correlation Matrix
rcParams['figure.figsize']=12,8
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]),dataset.columns)
plt.xticks(np.arange(dataset.shape[1]),dataset.columns)
plt.colorbar()

#Plotting Histogram
dataset.hist()
plt.show()

rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(),dataset['target'].value_counts(),color =['red','green'])
plt.xticks([0,1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

#Scaling dataset
dataset = pd.get_dummies(dataset,columns= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler =StandardScaler()
columns_to_scale =['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#Spliting into training and testing dataset
y = dataset['target']
X = dataset.drop(['target'],axis =1)
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

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
print("The score for Decision Tree Classifier is ",format(dt_scores[17]*100),"%")
plt.show()

#K Nearest Neighbor
knn_score=[]
for k in range(1,21):
    knn_classifier =KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    knn_score.append(knn_classifier.score(X_test,y_test))
plt.plot([k for k in range(1,21)],knn_score,color='red')
for i in range(1,21):
    plt.text(i,knn_score[1-i],(i,knn_score[1-i]))
    plt.xticks([i for i in range(1,21)])
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Scores')
    plt.title('K neighbors Classifier Score for different K values')
print("The score KNN is ",format(knn_score[7]*100),"%")
plt.show()

#Support Vector Machine
svc_score=[]
kernels =['linear','poly','rbf','sigmoid']
for i in range(len(kernels)) :
    svc_classifier = SVC(kernel=kernels[i])
    svc_classifier.fit(X_train,y_train)
    svc_score.append(svc_classifier.score(X_test,y_test))
colors= rainbow(np.linspace(0,1,len(kernels)))
plt.bar(kernels,svc_score,color=colors)
for i in range(len(kernels)) :
    plt.text(i,svc_score[i],svc_score[i])
    plt.xlabel('kernels')
    plt.ylabel('Score')
    plt.title('Suport Vector Classifier for different Kernels')
print("The score for SVM is ",format(svc_score[0]*100),"%")
plt.show()

#Random Forest Classifier
rf_scores=[]
estimators = [10,200,500,1000]
for i in estimators :
    rf_classifier = RandomForestClassifier(n_estimators=i,random_state=0)
    rf_classifier.fit(X_train,y_train)
    rf_scores.append(rf_classifier.score(X_test,y_test))
colors= rainbow(np.linspace(0,1, len(estimators)))
plt.bar([i for i in range(len(estimators))],rf_scores,color=colors,width=0.8)
for i in range(len(estimators)) :
    plt.text(i,rf_scores[i],rf_scores[i])
    plt.xticks([i for i in range(len(estimators))],[str(estimators) for estimator in estimators])
    plt.xlabel('Number of Estimators')
    plt.ylabel('Scores')
    plt.title('Random Forest Classifier scores for different number of estimators')
print("The score for Random Forest Classifier is ",format(rf_scores[2]*100),"%")
plt.show()