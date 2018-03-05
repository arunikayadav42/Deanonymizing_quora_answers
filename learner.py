# Natural Language Processing

#Importing Libraries

import csv,os,sys
import nltk
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

#importing the dataset

path1 = 'C50train/'
authors = os.listdir(path1)[:50]

y_train = []
X_train = np.array([])
for auth in authors:
    files = os.listdir(path1+auth+'/')
    
    for file in files:
        y_train = y_train+[auth]
        f=open(path1+auth+'/'+file,'r')
        data = f.read().replace('\n', ' ')
        #cleaned data
        #removing punctuations
        cleaned = re.sub('[^a-zA-z]',' ',data)
        #converting to lower case
        cleaned = cleaned.lower()
        #removing stop words using nltk library
        cleaned = cleaned.split()
        # Stemming
        ps = PorterStemmer()
        cleaned = [ps.stem(word) for word in cleaned if not word in set(stopwords.words('english'))]
        cleaned = ' '.join(cleaned)
        X_train = np.append(X_train,cleaned)
        #X_train = ' '.join(X_train)
        f.close()
path2 = 'C50test/'
authors = os.listdir(path2)[:50]
for auth in authors:  

	files = os.listdir(path2+auth+'/')
	for file in files:
         y_train = y_train+[auth]
         f = open(path2+auth+'/'+file, 'r')
         data = f.read().replace('\n', ' ')
         cleaned = re.sub('[^a-zA-z]',' ',data)
         #converting to lower case
         cleaned = cleaned.lower()
         #removing stop words using nltk library
         cleaned = cleaned.split()
         # Stemming
         ps = PorterStemmer()
         cleaned = [ps.stem(word) for word in cleaned if not word in set(stopwords.words('english'))]
         cleaned = ' '.join(cleaned)
         X_train = np.append(X_train,cleaned)
         f.close()
 # Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()        

X = cv.fit_transform(X_train).toarray()
#np.concatenate(y_train)
#list(sum(y_train, )
y = y_train

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes classifier to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

#Fitting SVM Classifier to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = np.trace(cm)
print ((2500-accuracy)/25)

#visualizing training set results
from matplotlib.colors import ListedColorMaps 
