"""
Implementation of bagging ensemble classifiers in Scikit-learn and Python
Author: Aleksandar Haber

Note that this code file imports the function "visualizeClassificationAreas"
from the file: functions.py

"""
# support vector machine classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC

# decision tree classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

# bagging classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
from sklearn.ensemble import BaggingClassifier

# data set database for generating different data sets for testing the algorithms
# https://scikit-learn.org/stable/datasets.html
from sklearn import datasets
# accuracy_score metric to test the performance of the classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import accuracy_score
# train_test split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
# standard scaler used to scale and standardize the data set
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler 

# function for visualizing the classification areas
from functions import visualizeClassificationAreas

# load the data set
# Moons data set
# https://aleksandarhaber.com/scatter-plots-for-classification-problems-in-python-and-scikit-learn/
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

# Moons data set
Xtotal, Ytotal = datasets.make_moons(n_samples=400, noise = 0.15)

# split the data set into training and test data sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtotal, Ytotal, test_size=0.2)

# create a standard scaler
scaler1=StandardScaler()
# scale the training and test input data
# fit_transform performs both fit and transform at the same time
XtrainScaled=scaler1.fit_transform(Xtrain)
# here we only need to transform
XtestScaled=scaler1.transform(Xtest)

# Define the bagging classifier by using the decision tree base classifier
# define the base classifier
# max_depth=None means that nodes are expanded until all leaves are pure
treeCLF = DecisionTreeClassifier(criterion='entropy',max_depth=None,random_state=42)
BaggingCLF_tree=BaggingClassifier(treeCLF,n_estimators=50,max_samples=0.5,bootstrap=True,n_jobs=-1)
# max samples=0.5 means that the number of random samples drawn from the original data set is equal to 
# 0.5*X.shape[0], where X is the input data set of samples (same applies to the output data set)
# n_jobs=-1 - means using all the processors to do computations
# note that since the decision tree base classifier implements predict_proba() method
# then the predicted class is selected as the class with the highest mean probability

# Define the bagging classifier by using the support vector machine base classifier
# define the base classifier
SVMCLF=SVC()
# define the bagging classifier
BaggingCLF_SVM=BaggingClassifier(SVMCLF,n_estimators=50,max_samples=0.5,bootstrap=True,n_jobs=-1)


# create a list of classifier tuples
# (classifier name, classifier object)
classifierList=[('Bagging_Tree',BaggingCLF_tree),('Bagging_SVM',BaggingCLF_SVM)]

# this dictionary is used to store the classification scores
classifierScore={}

# here we iterate through the classifiers and compute the accuracy score
# and store the accuracy store in the list
for nameCLF,CLF in classifierList:
    CLF.fit(XtrainScaled,Ytrain)
    CLF_prediction=CLF.predict(XtestScaled)
    classifierScore[nameCLF]=accuracy_score(Ytest,CLF_prediction)

# visualize the classification regions
visualizeClassificationAreas(BaggingCLF_tree,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_Bagging_Tree.png', plotDensity=0.01)
visualizeClassificationAreas(BaggingCLF_SVM,XtrainScaled, Ytrain,XtestScaled, Ytest, filename='classification_results_Bagging_SVM.png', plotDensity=0.01)

