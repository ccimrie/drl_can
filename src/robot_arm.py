import numpy as np
import rospy
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2

class RobotArm(object):
    def __init__(self):
        self.br = CvBridge()
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.arm_activate=0
        self.arm=0
        self.acquired=0

        self.timer=0
        self.timeout=500

        self.cmd_vel=rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.arm_update_pub=rospy.Publisher('/arm_motion', Int8, queue_size=1)
        self.arm_update_sub=rospy.Subscriber('/arm_motion', Int8, self.armUpdate, queue_size=1)
        self.openGripper()

    def armUpdate(self, data):
        arm_temp=data.data
        if arm_temp==1:
            twist_vel=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.cmd_vel.publish(twist_vel)
            # self.moveArmHome()
            ## Initiate grab
            self.moveArmAngles([1.1,-0.1,-1.0])
            self.openGripper()
            self.registerArmCam()

    def unregisterArmCam(self):
        self.cam_arm.unregister()
        print("DONE")
        self.arm_update_pub.publish(0)

    def registerArmCam(self):
        self.timer=0
        self.cam_arm=rospy.Subscriber('/camera_arm/rgb/image_raw', Image, self.visualServoying, queue_size=1)

    def armUpdate(self, data):
        print("Arm change")
        arm_temp=data.data
        if arm_temp==1:
            twist_vel=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.cmd_vel.publish(twist_vel)
            # self.moveArmHome()
            ## Initiate grab
            self.moveArmAngles([1.1,-0.1,-1.0])
            self.openGripper()
            self.arm=arm_temp
            rospy.sleep(1.0)
            self.registerArmCam()
        self.arm=arm_temp

    def visualServoying(self, data):
        if self.timer>self.timeout:
            self.moveArmHome()
            self.unregisterArmCam()
            return
        else: 
            self.timer=self.timer+1
            if self.timer%100==0:
                print("Arm timer: ", self.timer)

        twist=Twist()

        image=self.br.imgmsg_to_cv2(data)[0:450,:]#,"bgr8")#/255.0
        centre_x=np.shape(image)[1]/2
        centre_y=int(0.99*np.shape(image)[0])

        image2=cv2.Canny(image,0,50)

        kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(image2, kernel, iterations=20)
        image2 = cv2.erode(d_im, kernel, iterations=5)         

        # # get the (largest) contour
        contours = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        # # draw white filled contour on black background
        result = np.zeros_like(image2)
        cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

        M = cv2.moments(result)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # cY=int(0.95*np.shape(image)[0])
        # put text and highlight the center
        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        bound=10

        ## check horizontal alignment first
        if cX<centre_x-bound:
            twist.angular.z=0.02
        elif cX>centre_x+bound:
            twist.angular.z=-0.02
        else:
            twist.angular.z=0.0
        ## check vertical alignment if centred
        if cY>centre_y+bound:
            ## pull back
            twist.linear.x=-0.005
        elif cY<centre_y-bound:
            twist.linear.x=0.005
        else:
            twist.linear.x=0
        self.cmd_vel.publish(twist)
        if twist.linear.x==0 and twist.angular.z==0:
            self.gripperClose()
            self.moveArmHome()
            self.arm=0
            self.unregisterArmCam()

    ## In radians
    def moveArmAngles(self, angles):
        arm_joint_goal = [0.0, angles[0], angles[1], angles[2]]
        # Move the arm
        self.move_group_arm.go(arm_joint_goal, wait=True)
        # The above should finish once the arm has fully moved.
        # However, to prevent any residual movement,we call the following as well.
        self.move_group_arm.stop() 

    def moveArmPose(self, pose_goal):
        # pose=self.move_group_arm.get_current_pose().pose
        # # pose_goal = geometry_msgs.msg.Pose()
        # pose.position.x=pose_goal[0]
        # pose.position.y=pose_goal[1]
        # pose.position.z=pose_goal[2]
        print("Moving arm")
        pose=pose_goal

        self.move_group_arm.set_pose_target(pose)
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success=self.move_group_arm.go(wait=True)

        # Calling `stop()` ensures that there is no residual movement
        self.move_group_arm.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group_arm.clear_pose_targets()

    def moveArmHome(self):
        joint_angles=self.move_group_arm.get_current_joint_values()
        joint_angles[0]=0.0
        joint_angles[1]=0.0
        joint_angles[2]=0.0
        joint_angles[3]=0.0
        self.move_group_arm.go(joint_angles,wait=True)
        self.move_group_arm.stop()

    def openGripper(self):
        joint_gripper=self.move_group_gripper.get_current_joint_values()
        joint_gripper[0]=0.018
        joint_gripper[1]=0.018
        self.move_group_gripper.go(joint_gripper, wait=True)
        self.move_group_gripper.stop()

    def closeGripper(self):
        joint_gripper=self.move_group_gripper.get_current_joint_values()
        joint_gripper[0]=-0.01
        joint_gripper[1]=-0.01
        self.move_group_gripper.go(joint_gripper, wait=True)
        self.move_group_gripper.stop()