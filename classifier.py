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
	hingeDir = eg.fileopenbox(msg="Open labeled Hinge directory", title="Hinge dir")
	nonhingeDir = eg.fileopenbox(msg="Open labeled Non-Hinge directory", title="Non-Hinge dir")


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

