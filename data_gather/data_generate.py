import csv
import os
import numpy as np

images=os.listdir('dataset/images/')
# boxes=os.listdir('dataset/boxes/')
# print(len(images), len(boxes))
	
data_train=open('data_train.txt','w')
data_test=open('data_test.txt','w')

end=int(0.9*len(images))

for i in np.arange(end):
	# print(images[i])
	val=images[i][6:-4]

	file=open('dataset/boxes/boxes_'+val+'.csv')
	csvreader=csv.reader(file)
	header=[]
	header=next(csvreader)
	rows=[]
	for row in csvreader:
		rows.append(row)
	# print(rows)
	data_train.write('./images/'+images[i]+'\n')
	label_file=open('labels/'+str(images[i])[:-4]+'.txt','w')
	for row in rows:
		CLASS_ID=row[0]
		x=str(float(row[1])/640.0)
		y=str(float(row[2])/480.0)
		width=str(float(row[3])/640.0)
		height=str(float(row[4])/480.0)

		# if float(x)==1 or float(y)==1 or float(width)==1 or float(height)==1:
		# 	print('uh oh more')
		if float(x)==0:
			x=str(float(x)+1e-10)
			width=str(float(width)+1e-10)
		if float(y)==0:
			y=str(float(y)+1e-10)
			height=str(float(height)+1e-10)

		label_file.write(CLASS_ID+' '+x+' '+y+' '+width+' '+height+'\n')

for i in np.arange(end,len(images)):
	# print(images[i])
	val=images[i][6:-4]

	file=open('dataset/boxes/boxes_'+val+'.csv')
	csvreader=csv.reader(file)
	header=[]
	header=next(csvreader)
	rows=[]
	for row in csvreader:
		rows.append(row)
	# print(rows)
	data_test.write('./images/'+images[i]+'\n')
	label_file=open('labels/'+str(images[i])[:-4]+'.txt','w')
	for row in rows:
		CLASS_ID=row[0]
		x=str(float(row[1])/640.0)
		y=str(float(row[2])/480.0)
		width=str(float(row[3])/640.0)
		height=str(float(row[4])/480.0)

		if float(x)==0:
			x=str(float(x)+1e-10)
			width=str(float(width)+1e-10)
		if float(y)==0:
			y=str(float(y)+1e-10)
			height=str(float(height)+1e-10)

		label_file.write(CLASS_ID+' '+x+' '+y+' '+width+' '+height+'\n')


