from cProfile import label
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import graphviz

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

#Importing data frame
df = pd.read_csv('connect-4.data', header=None)
df.columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'result']
df.to_csv('connect-4.csv', index=False)
 
#Converting data frame into categorical
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.describe() 
   
#Value = 42
#Spliting data into train and test
x = df.values[:,0:42]
y = df.values[:,42]

#Train/Test = 22/20
feature_train, feature_test, label_train, label_test = train_test_split(x, y, train_size=22, random_state=4)

#Decision Tree
clf = DecisionTreeClassifier(criterion='gini',random_state = None)
clf = clf.fit(feature_train, label_train)

#Using sklearn to interpret classification report and the confusionmatrix

#Prediction
y_pred = clf.predict(feature_test)
print("Predicted values:")
print(y_pred)

#Accuracy
print("Confusion Matrix:\n ",confusion_matrix(label_test, y_pred))    
print ("Accuracy : ",accuracy_score(label_test,y_pred)*100,'%')     
print("Report :\n ",classification_report(label_test, y_pred))


#Visualizing the tree
dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
fig = plt.figure(figsize=(15,10))
plot_tree(clf,rounded=True,fontsize =14, filled=True)
plt.title("Decision Tree")

#Saving
fig.savefig('decision_tree.png')

#Printing
plt.show()

#The depth and accuracy of a decision tree

#Change to 80/20 training set/test set
feature_train, feature_test, label_train, label_test = train_test_split(x, y, train_size=0.8,shuffle = False)

index = None #Change max depth to filled to table
clf = DecisionTreeClassifier(criterion='gini',random_state = None,max_depth=index) 
clf = clf.fit(feature_train, label_train)

#Accuracy
y_pred = clf.predict(feature_test)
print ("Accuracy of depth ",index,':',accuracy_score(label_test,y_pred)*100,'%')  