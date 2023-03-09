# -*- coding: utf-8 -*-
"""
This file contains the function for visualizing the classification regions

"""

def visualizeClassificationAreas(usedClassifier,XtrainSet, ytrainSet,XtestSet, ytestSet, filename='classification_results.png', plotDensity=0.01 ):
    '''
    This function visualizes the classification regions on the basis of the provided data
 
    usedClassifier      - used classifier
 
    XtrainSet,ytrainSet - train sets of features and target classes 
                        - the region colors correspond to to the class numbers in ytrainSet
 
    XtestSet,ytestSet   - test sets for visualizng the performance on test data that is not used for training
                        - this is optional argument provided if the test data is available
    
    filename            - file name with the ".png" extension to save the classification graph
 
    plotDensity         - density of the area plot, smaller values are making denser and more precise plots
 
    IMPORTANT COMMENT:  -If the number of classes is larger than the number of elements in the list 
                    caller "colorList", you need to increase the number of colors in 
                    "colorList"
                    - The same comment applies to "markerClass" list defined below
 
    '''
    import numpy as np
    import matplotlib.pyplot as plt    
    # this function is used to create a "cmap" color object that is used to distinguish different classes
    from matplotlib.colors import ListedColormap 
     
    # this list is used to distinguish different classification regions
    # every color corresponds to a certain class number
    colorList=["blue", "green", "orange", "magenta", "purple", "red"]
    # this list of markers is used to distingush between different classe
    # every symbol corresponds to a certain class number
    markerClass=['x','o','v','#','*','>']
     
    # get the number of different classes 
    classesNumbers=np.unique(ytrainSet)
    numberOfDifferentClasses=classesNumbers.size
                  
    # create a cmap object for plotting the decision areas
    cmapObject=ListedColormap(colorList, N=numberOfDifferentClasses)
                  
    # get the limit values for the total plot
    x1featureMin=min(XtrainSet[:,0].min(),XtestSet[:,0].min())-0.5
    x1featureMax=max(XtrainSet[:,0].max(),XtestSet[:,0].max())+0.5
    x2featureMin=min(XtrainSet[:,1].min(),XtestSet[:,1].min())-0.5
    x2featureMax=max(XtrainSet[:,1].max(),XtestSet[:,1].max())+0.5
         
    # create the meshgrid data for the classifier
    x1meshGrid,x2meshGrid = np.meshgrid(np.arange(x1featureMin,x1featureMax,plotDensity),np.arange(x2featureMin,x2featureMax,plotDensity))
         
    # basically, we will determine the regions by creating artificial train data 
    # and we will call the classifier to determine the classes
    XregionClassifier = np.array([x1meshGrid.ravel(),x2meshGrid.ravel()]).T
    # call the classifier predict to get the classes
    predictedClassesRegion=usedClassifier.predict(XregionClassifier)
    # the previous code lines return the vector and we need the matrix to be able to plot
    predictedClassesRegion=predictedClassesRegion.reshape(x1meshGrid.shape)
    # here we plot the decision areas
    # there are two subplots - the left plot will plot the decision areas and training samples
    # the right plot will plot the decision areas and test samples
    fig, ax = plt.subplots(1,2,figsize=(15,8))
    ax[0].contourf(x1meshGrid,x2meshGrid,predictedClassesRegion,alpha=0.3,cmap=cmapObject)
    ax[1].contourf(x1meshGrid,x2meshGrid,predictedClassesRegion,alpha=0.3,cmap=cmapObject)
     
    # scatter plot of features belonging to the data set 
    for index1 in np.arange(numberOfDifferentClasses):
        ax[0].scatter(XtrainSet[ytrainSet==classesNumbers[index1],0],XtrainSet[ytrainSet==classesNumbers[index1],1], 
                   alpha=1, c=colorList[index1], marker=markerClass[index1], 
                   label="Class {}".format(classesNumbers[index1]), edgecolor='black', s=80)
         
        ax[1].scatter(XtestSet[ytestSet==classesNumbers[index1],0],XtestSet[ytestSet==classesNumbers[index1],1], 
                   alpha=1, c=colorList[index1], marker=markerClass[index1],
                   label="Class {}".format(classesNumbers[index1]), edgecolor='black', s=80)
         
    ax[0].set_xlabel("Feature X1",fontsize=14)
    ax[0].set_ylabel("Feature X2",fontsize=14)
    ax[0].text(0.05, 0.05, "Decision areas and TRAINING samples", transform=ax[0].transAxes, fontsize=14, verticalalignment='bottom')
    ax[0].legend(fontsize=14)
     
    ax[1].set_xlabel("Feature X1",fontsize=14)
    ax[1].set_ylabel("Feature X2",fontsize=14)
    ax[1].text(0.05, 0.05, "Decision areas and TEST samples", transform=ax[1].transAxes, fontsize=14, verticalalignment='bottom')
    #ax[1].legend()
    plt.savefig(filename,dpi=500)