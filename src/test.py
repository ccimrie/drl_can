#!/usr/bin/env python3
import rospy

class test(object):
    def __init__(self):
        print("Did it!")

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    print("IN METHOD")
    print("STARTING NODE")
    rospy.init_node("testNode", anonymous=True)
    print("Making robot")
    robot=test()
    print("Successful!")
    robot.start()