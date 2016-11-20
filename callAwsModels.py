"""
Call the AWS Machine Learning API to make predictions

Requires the AWS Python SDK
	pip install boto3

Accept the following arguments:
	Requires either
		--batch or -b + csv file to send for requests
		--realtime or -r + image file to send as request
	Optional:
		--model or -m + number of the model to use (2 or 3)
"""

import sys
import json
import requests
import Image
import boto3
import StringIO

# Options
ml_model = 3
ml_model_id = "ml-eMOBy0niEOW"
predict_endpoint = "https://realtime.machinelearning.us-east-1.amazonaws.com"
batch = False
batch_source_file = ""
realtime_source_image = ""

# Get command line args
args = sys.argv

if len(args) > 1:
	for i in range(1,len(args)):
		if args[i] == "--batch" or args[i] == "-b":
			batch = True
			batch_source_file = args[i + 1]
			i += 1
			continue
		elif args[i] == "--realtime" or args[i] == "-r":
			batch = False
			realtime_source_image = args[i + 1]
			i += 1
			continue
		elif args[i] == "--model" or args[i] == "-m": # Current valid values are 2 and 3
			ml_model = int(args[i + 1])
			i += 1
			continue
		elif args[i] == "--help" or args[i] == "-h" or args[i] == "-?":
			print "Usage:"
			print "  python callAwsModels.py [options]"
			print "  One of the following options must be used:"
			print "    -r, --realtime <path/to/image>       Choose the image to use to generate a single prediction."
			print "    -b --batch <path/to/dir|path/to/csv> Choose the path to a directory of images to create a batch request."
			print "  The following options are optional:"
			print "    -m, --model <2/3>                    Choose the ML model to use."
			print "    -h, -?, --help                       Show this screen."
			exit()
		else:
			continue
			print args[i] + " is an invalid option."

if ml_model == 2:
	ml_model_id = "ml-mTkWlU4nziF"
	predict_endpoint = "https://realtime.machinelearning.us-east-1.amazonaws.com"
elif ml_model == 3:
	ml_model_id = "ml-eMOBy0niEOW"
	predict_endpoint = "https://realtime.machinelearning.us-east-1.amazonaws.com"

client = boto3.client('machinelearning')

if not batch:
	record = {}

	g = Image.open(realtime_source_image).load()

	count = 0
	pixel_num = 0
	for i in range(161):
		for j in range(161):
			if count % 27 == 0:
				record[str(pixel_num)] = str(g[j,i][0])
				pixel_num += 1
			count += 1
	
	response = client.predict(
		MLModelId = ml_model_id,
		Record = record,
		PredictEndpoint = predict_endpoint
	)

	prediction = response['Prediction']['predictedLabel']

	if prediction == "1":
		print "With " + str(response['Prediction']['predictedScores']['1']) + "% certainty, " + realtime_source_image + " is a match."
	elif prediction == "0":
		print "With " + str(response['Prediction']['predictedScores']['1']) + "% certainty, " + realtime_source_image + " is not a match."
else:
	print "WIP"