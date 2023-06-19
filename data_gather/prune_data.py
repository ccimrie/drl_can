import os
import csv

## Check if completed
csv_list = sorted(list(filter(lambda x: '.csv' in x, os.listdir('dataset/boxes/'))))

for last_csv in csv_list:	
	print("STARTING process")
	print(last_csv)
	file=open('dataset/boxes/'+last_csv)
	csvreader=csv.reader(file)
	items=sum(1 for row in csvreader)
	print(items)
	if items<2:
		# print(csv)
		while (os.path.exists('dataset/boxes/'+last_csv)):
			os.remove('dataset/boxes/'+last_csv)
		while (os.path.exists('dataset/images/image'+last_csv[5:-3]+'png')):
			os.remove('dataset/images/image'+last_csv[5:-3]+'png')