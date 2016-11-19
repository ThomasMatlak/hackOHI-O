"""
This is an image classifier for DNA Origami Hinges
"""
from __future__ import division
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
import os
import pickle
import easygui as eg
import numpy as np
from imageio import imread
from random import shuffle



def getImages(hingeDirectory, nonHingeDirectory):
	#Open the directory, pulls each image and returns an array with an np array as the 1st element and either a 1 or 0 as the 2nd
	#1 = Hinge, 0 = Non-Hinge
	n=0
	FImages = np.zeros((len(os.listdir(hingeDirectory))+len(os.listdir(nonHingeDirectory))), dtype=np.int).tolist()
	for file in os.listdir(hingeDirectory):
		rawNPArray = imread(hingeDirectory + '\\' + file) 
		FImages[n] = [flatten(rawNPArray), 1]
		n+=1
	for file in os.listdir(nonHingeDirectory):
		rawNPArray = imread(nonHingeDirectory + '\\' + file) 
		FImages[n] = [flatten(rawNPArray), 0]
		n+=1
	return FImages

def flatten(image):
	# rowLen = len(image[0])
	index = 0
	image = zip(*image[0]) #This strips the superflous rgb values
	flatImage = np.zeros(len(image) * len(image[0]), dtype=np.int)
	for row in range(0, len(image)):
		for pixel in range(0, row):
			flatImage[index] = image[row][pixel]
			index += 1
		# print flatImage
	return flatImage 



if __name__ == '__main__':
	#isn't python great? This is the same thing as the main method in java.
	#start by getting the images

	#Testing stuff
	hingeDir = eg.diropenbox(msg="Open labeled Hinge directory", title="Hinge dir")
	# print os.listdir(hingeDir)
	nonhingeDir = eg.diropenbox(msg="Open labeled Non-Hinge directory", title="Non-Hinge dir")
	images = getImages(hingeDir, nonhingeDir) #Getting training data

	shuffle(images)
	trainingImages = images[len(images)//2:] #Splits images into testing and training sets
	testingImages = images[:len(images)//2]

	userGamma, tolerence = eg.multenterbox("Enter custom values", "Customize SVC", ["Gamma", "Tolerence"])
	if userGamma is None:
		userGamma = "auto"
	if tolerence is None:
		tolerence = 0.001

	classifier = svm.SVC(gamma=userGamma, tol=tolerence)
	# print zip(*trainingImages)[1]
	classifier.fit(zip(*trainingImages)[0], zip(*trainingImages)[1])

	expected = zip(*testingImages)[1]
	predicted = classifier.predict(zip(*testingImages)[0])

	print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
