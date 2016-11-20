"""
Call the AWS Machine Learning API to make predictions

Requires the AWS Python SDK
	pip install boto3

Accept the following arguments:
	Requires either
		--batch or -b + csv file to send for requests
		--realtime or -r + image file to send as request
	Optional:
		--model or -m + number of the model to use (only current option is 3)
"""

import sys
import Image
import boto3

def callAwsModels(ml_model, ml_model_id, predict_endpoint, batch, batch_source_file, realtime_source_image):
	# if ml_model == 2:
	# 	ml_model_id = "ml-mTkWlU4nziF"
	# 	predict_endpoint = "https://realtime.machinelearning.us-east-1.amazonaws.com"
	if ml_model == 3:
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
		percentCertainty = response['Prediction']['predictedScores']

		if prediction == "1":
			print realtime_source_image + " is a match. Certainty: " + str(100 * percentCertainty['1']) + " %"
		elif prediction == "0":
			print realtime_source_image + " is not a match. Certainty: " + str(100 * percentCertainty['0']) + " %"
	else:
		print "Future Work"

if __name__ == '__main__':
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
				print ""
				print "Usage:"
				print "  python callAwsModels.py [options]"
				print "  One of the following options must be used:"
				print "    -r, --realtime <path/to/image>       Choose the image to use to generate a single prediction."
				print "    -b --batch <path/to/dir|path/to/csv> Choose the path to a directory of images to create a batch request. (not yet implemented)"
				print "  The following options are optional:"
				print "    -m, --model <3>                      Choose the ML model to use. The only model available currently is model 3."
				print "    -h, -?, --help                       Show this screen."
				exit()
			else:
				continue
				print args[i] + " is an invalid option."
	callAwsModels(ml_model, ml_model_id, predict_endpoint, batch, batch_source_file, realtime_source_image)