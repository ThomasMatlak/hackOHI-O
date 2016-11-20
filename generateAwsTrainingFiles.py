# Create the .csv file to teach Amazon Machine Learning
import csv
import Image
import os

with open("hinges_training_data.csv", "wb") as csvfile:
	csvwriter = csv.writer(csvfile)
	# Create Schema
	schema = open("hinges_training_data.csv.schema", "wb")
	schema.write("{" + "\n")
	schema.write("\t" + '"version": "1.0",' + "\n")
	schema.write("\t" + '"targetAttributeName": "isHinge",' + "\n")
	schema.write("\t" + '"dataFormat": "CSV",' + "\n")
	schema.write("\t" + '"dataFileContainsHeader": true,' + "\n")
	schema.write("\t" + '"attributes": [' + "\n")

	count = 0
	for i in range(31):
		for j in range(31):
			schema.write("\t\t{\n\t\t\t" + '"attributeName":' + '"' + str(count) + '",' + "\n")
			schema.write("\t\t\t" + '"attributeType":' + '"NUMERIC"},' + "\n")
			count += 1

	schema.write("\t\t{\n\t\t\t" + '"attributeName":' + '"isHinge",' + "\n")
	schema.write("\t\t\t" + '"attributeType":' + '"BINARY"}' + "\n")

	schema.write("\t]\n}")

	row = []
	count = 0
	for i in range(31):
		for j in range(31):
		   row.append(count)
		   count += 1

	row.append("isHinge")
	csvwriter.writerow(row)

	# Generate training CSV
	good = os.listdir("good hinges")
	bad = os.listdir("not hinges")

	good_c = 0
	bad_c = 0

	while good_c < len(good) and bad_c < len(bad):
		if good_c < len(good):
			flattened_image = []

			g = Image.open("good hinges/" + good[good_c]).load()

			count = 0
			for i in range(161):
				for j in range(161):
					if count % 27 == 0:
						# flattened_image.append(str(g[i,j][0]))
						flattened_image.append(str(g[j,i][0]))
					count += 1

			csvwriter.writerow(flattened_image + [1])

		if bad_c < len(bad):
			flattened_image = []

			g = Image.open("not hinges/" + bad[bad_c]).load()

			count = 0
			for i in range(161):
				for j in range(161):
					if count % 27 == 0:
						# flattened_image.append(str(g[i,j][0]))
						flattened_image.append(str(g[j,i][0]))
					count += 1

			csvwriter.writerow(flattened_image + [0])

		good_c += 1
		bad_c += 1