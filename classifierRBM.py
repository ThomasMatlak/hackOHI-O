"""
This is an image classifier for DNA Origami Hinges
"""
from __future__ import division
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics, neural_network, neighbors, tree, linear_model
import os
import pickle
import easygui as eg
import numpy as np
from imageio import imread, imwrite
from random import shuffle
from sklearn.externals import joblib	
from sklearn.pipeline import Pipeline



def getImages(hingeDirectory, nonHingeDirectory):
	#Open the directory, pulls each image and returns an array with an np array as the 1st element and either a 1 or 0 as the 2nd
	#1 = Hinge, 0 = Non-Hinge
	n=0
	FImages = np.zeros((len(os.listdir(hingeDirectory))+len(os.listdir(nonHingeDirectory))), dtype=np.int).tolist()
	for file in os.listdir(hingeDirectory):
		rawNPArray = imread(hingeDirectory + '\\' + file) 
		# print rawNPArray
		# print file
		FImages[n] = [flatten(rawNPArray), 1]
		n+=1
	for file in os.listdir(nonHingeDirectory):
		rawNPArray = imread(nonHingeDirectory + '\\' + file) 
		FImages[n] = [flatten(rawNPArray), 0]
		n+=1
	print len(FImages[0][0])
	return FImages

def flatten(image):
	# rowLen = len(image[0])
	# print image
	index = 0
	try:
		image = zip(*image[0]) #This strips the superflous rgb values
	except TypeError:
		print 'huh'
	flatImage = image[0]
	# print image
	flatImage = np.hstack(image)
	print flatImage
	return flatImage 



if __name__ == '__main__':
	#isn't python great? This is the same thing as the main method in java.
	#start by getting the images

	#Testing stuff
	print "Started"
	hingeDir = eg.diropenbox(msg="Open labeled Hinge directory", title="Hinge dir")
	# print os.listdir(hingeDir)
	nonhingeDir = eg.diropenbox(msg="Open labeled Non-Hinge directory", title="Non-Hinge dir")
	images = getImages(hingeDir, nonhingeDir) #Getting training data
	print "Got the images!"

	# print zip(*images[1])
	shuffle(images)
	# print zip(*images[1])
	trainingImages = images[len(images)//2:] #Splits images into testing and training sets
	testingImages = images[:len(images)//2]
	# classifier = svm.SVC(gamma=userGamma, tol=tolerence)
	print "Initializing NN!"
	# classifier = neural_network.MLPClassifier(hidden_layer_sizes=(400, 300, 200, 100), solver="lbfgs", max_iter=10000, alpha=.0001, activation="tanh", verbose=False)
	# classifier = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_features="auto")
	logistic = linear_model.LogisticRegression()
	rbm = neural_network.BernoulliRBM()
	print("Starting to fit, hold on tight!")
	# print zip(*trainingImages)[1]	
	classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

	rbm.learning_rate = 0.03
	rbm.n_iter = 15
	# More components tend to give better prediction performance, but larger
	# fitting time
	rbm.n_components = 100
	logistic.C = 6000.0

	# Training RBM-Logistic Pipeline
	classifier.fit(zip(*trainingImages)[0], zip(*trainingImages)[1])

	# Training Logistic regression
	logistic_classifier = linear_model.LogisticRegression(C=100.0)
	logistic_classifier.fit(zip(*trainingImages)[0], zip(*trainingImages)[1])

	# classifier.fit(zip(*trainingImages)[0], zip(*trainingImages)[1])
	# print("Fitted!")

	# expected = zip(*testingImages)[1]
	# predicted = classifier.score_samples(zip(*testingImages)[0])
	# print predicted

	# print("Classification report for classifier %s:\n%s\n"
 #      % (classifier, metrics.classification_report(expected, predicted)))
	# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        zip(*testingImages)[1],
        classifier.predict(zip(*testingImages)[0]))))

	print("Logistic regression using raw pixel features:\n%s\n" % (
	    metrics.classification_report(
        zip(*testingImages)[1]	,
        logistic_classifier.predict(zip(*testingImages)[0]))))

	joblib.dump(classifier, 'RBMClassifier.pkl')