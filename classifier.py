"""
This is an image classifier for DNA Origami Hinges


"""
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import os
from __future__ import division
import pickle
import easygui as eg
import numpy as np
from imageio import imread
from random import shuffle

if __name__ == '__main__':
	#isn't python great? This is the same thing as the main method in java.
	#start by getting the images

	#Testing stuff
	if (eg.ynbox('Testing?', 'test', ('Use actual data', 'Use MNIST Data'))):
		hingeDir = eg.fileopenbox(msg="Open labeled Hinge directory", title="Hinge dir")
		nonhingeDir = eg.fileopenbox(msg="Open labeled Non-Hinge directory", title="Non-Hinge dir")
		images = getImages(hingeDir, nonhingeDir) #Getting training data
	else:
		images = getMNIST()
	shuffle(images)
	trainingImages = images[len(images)/2:] #Splits images into testing and training sets
	testingImages = images[:len(images)/2]

	classifier = svm.SVC(gamma=0.001)





def getImages(hingeDirectory, nonHingeDirectory):
	#Open the directory, pulls each image and returns an array with an np array as the 1st element and either a 1 or 0 as the 2nd
	#1 = Hinge, 0 = Non-Hinge
	images[]
	n=0
	for file in os.listdir(hingeDirectory):
		images[n] = [imread(file), 1]
		n+=1
	for file in os.listdir(nonHingeDirectory):
		images[n] = [imread(file), 0]
		n+=1
	return images

def getMNIST():
	digits = datasets.load_digits()
	images_and_labels = list(zip(digits.images, digits.target))
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))

