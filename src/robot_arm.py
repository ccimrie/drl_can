import numpy as np
import rospy
import moveit_commander
import moveit_msgs.msg
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
        self.acquired=0

        self.cmd_vel=rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def unregisterArmCam(self):
        self.cam_arm.unregister()

    def registerArmCam(self):
        self.cam_arm=rospy.Subscriber('/camera_arm/rgb/image_raw', Image, self.armImageSub, queue_size=1)

    def armImageSub(self, data):
        img=np.array(self.br.imgmsg_to_cv2(data))

        img_interest=np.logical_and(img[:,:,1]>100,img[:,:,0]<120)*np.ones([np.shape(img)[0], np.shape(img)[1]])
        cv2.imshow('test', img_interest)
        cv2.waitKey(1)

        ## 420k for distance thresh
        size=np.sum(img_interest)
        vel=Twist()
        if size<85000 and self.arm_activate==0:
            print("Going forward")
            vel.linear.x=0.01
        else:
            self.arm_activate=1
        M = cv2.moments(img_interest)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        thresh=10
        if cX<340-thresh:
            print("Rotating left")
            vel.angular.z=0.05
        elif cX>340+thresh:
            print("Rotating right")
            vel.angular.z=-0.05
        self.cmd_vel.publish(vel)
        if self.arm_activate==1 and vel.angular.z==0:
            pose=self.move_group_arm.get_current_pose().pose
            pose.position.x=0.28
            pose.position.y=0
            pose.position.z=0.19
            pose.orientation.x=0
            pose.orientation.y=0
            pose.orientation.z=0
            pose.orientation.w=1

            self.moveArmPose(pose)

            if pose.position.x>0.27:
                self.acquired=1
                self.cam_arm.unregister()


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
        joint_angles=self.move_group.get_current_joint_values()
        joint_angles[0]=0.0
        joint_angles[1]=0.0
        joint_angles[2]=0.0
        joint_angles[3]=0.0
        self.move_group.go(joint_angles,wait=True)
        self.move_group.stop()

    def openGripper(self):
        gripper_joint_goal = [0.018,0.018]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()

    def closeGripper(self):
        gripper_joint_goal = [-0.01,-0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()