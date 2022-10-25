import subprocess
import os
import numpy as np
import time
## Generate sdf files
# arena.sdf=arena.txt+x*coke.txt+y*light.txt+close.txt

N=1500
for count in np.arange(N):
	print("Count: "+str(count)+"/"+str(N))
	f_full=open('arena.sdf','w')
	f_base=open('arena.txt','r').read()
	f_coke=open('coke.txt','r').read()
	f_light=open('light.txt','r').read()
	f_camera=open('camera_bb.txt','r').read()

	f_full.write(f_base+'\n')

	## Add coke coke cans to file
	for i in np.arange(np.random.randint(11)):
		coke_inital='<model name=\'coke_'+str(i)+'\'>\n'
		pose_str=''
		## randomise x,y
		x=(1-2*np.random.rand())*4.65
		y=(1-2*np.random.rand())*4.65
		bound=(x>2.85 and y>2.85)
		# print(bound)
		while bound:
			x=(1-2*np.random.rand())*4.65
			y=(1-2*np.random.rand())*4.65
			bound=(x>2.85 and y>2.85)

		# x=0.1
		# y=0.1

		pose_str=pose_str+str(x)+' '+str(y)+' '
		## z, ax, ay is 0
		pose_str=pose_str+'0 0 0 '
		## randomise rotation
		pose_str=pose_str+str(np.random.rand()*np.pi*2)+' '
		pose='<pose> '+pose_str+'</pose>\n'
		coke_txt=coke_inital+pose+f_coke
		# print(pose_str)
		f_full.write(coke_txt+'\n')

	# ## Add light fixtures to file
	for i in np.arange(np.random.randint(11)):
		light_inital='<model name=\'light_'+str(i)+'\'>\n'
		pose_str=''
		## randomise x,y
		x=(1-2*np.random.rand())*4.65
		y=(1-2*np.random.rand())*4.65
		bound=(x>2.85 and y>2.85)
		# print(bound)
		while bound:
			x=(1-2*np.random.rand())*4.65
			y=(1-2*np.random.rand())*4.65
			bound=(x>2.85 and y>2.85)

		pose_str=pose_str+str(x)+' '+str(y)+' '
		## z,ax,ay,az (doesn't matter as symmetrical design)
		for j in np.arange(4):
			pose_str=pose_str+str(0)+' '
		pose='<pose> '+pose_str+'</pose>\n'
		light_txt=light_inital+pose+f_light
		f_full.write(light_txt+'\n')

	## Add bounding-box camera
	camera_initial='<model name=\'camera\'>\n'
	pose_str=''
	## randomise x,y
	x=(1-2*np.random.rand())*4.65
	y=(1-2*np.random.rand())*4.65
	bound=(x>2.85 and y>2.85)
	while bound:
		x=(1-2*np.random.rand())*4.65
		y=(1-2*np.random.rand())*4.65
		bound=(x>2.85 and y>2.85)
	pose_str=pose_str+str(x)+' '+str(y)+' '
	## z, ax, ay is 0
	pose_str=pose_str+'0 0 0 '
	## randomise rotation
	pose_str=pose_str+str(np.random.rand()*np.pi*2)+' '
	pose='<pose> '+pose_str+'</pose>\n'
	camera_txt=camera_initial+pose+f_camera
	f_full.write(camera_txt+'\n')

	## End file
	f_full.write('\n  </world>\n</sdf>')
	f_full.close()

	## Launch IGN for a few seconds
	p=subprocess.Popen("ls dataset/images/ | wc -l", stdout=subprocess.PIPE, shell=True)
	s=p.communicate()[0]
	number=s.decode("utf-8")
	number=int(number.split("\n")[0])

	completed=0
	os.system('ign gazebo -r arena.sdf &')
	while not completed:
		# os.system('ls | wc -l')
		p=subprocess.Popen("ls dataset/images/ | wc -l", stdout=subprocess.PIPE, shell=True)
		s=p.communicate()[0]
		a=s.decode("utf-8")
		a=int(a.split("\n")[0])
		if a>number:
			completed=1


	## Kill IGN
	rby_done=0
	while not rby_done:
		p=subprocess.Popen("sudo ps -a", stdout=subprocess.PIPE, shell=True)
		s=p.communicate()[0]
		a=s.decode("utf-8")
		a=a.split("\n")

		pid=[]

		for i in a[0:-1]:
			t=i.split()
			if t[-1]=="ruby":
				pid.append(t[0])
		# print(pid)
		if len(pid)==0:
			rby_done=1
		else:
			for i in pid:
				subprocess.run(["sudo", "kill","-9",i])