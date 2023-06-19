import numpy as np
import rospy
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
import geometry_msgs.msg

class TurtlebotMovement(object):

    def __init__(self):
        self.vel_pub=rospy.Publisher('cmd/vel', Twist, queue_size=1)
        ## Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("Wait for the action server to come up")

        # print("Position set")
        ## Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))
        print("Got movebase server")

    def checkIfRunning(self):
        state=self.move_base.get_state()
        return (state==GoalStatus.ACTIVE or state==GoalStatus.PENDING)

    def moveToGoal(self, pose):
        goal_x=pose[0]
        goal_y=pose[1]
        goal_euler=pose[2]

        goal_orient_quat=np.sin(goal_euler/2.0)
        goal_w_quat=np.cos(goal_euler/2.0)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(goal_x, goal_y, 0.00),
                                             geometry_msgs.msg.Quaternion(0.0, 0, goal_orient_quat, goal_w_quat))

        # Start moving
        self.move_base.send_goal(goal)

    def returnSearch(self):
        # Send a goal

        x=3.5
        y=3.5
        euler=-3*np.pi/4
        quat_z=np.sin(euler/2)
        quat_w=np.cos(euler/2)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(x, y, 0.000),
                                         geometry_msgs.msg.Quaternion(0.0, 0, quat_z, quat_w))

        # Start moving
        self.move_base.send_goal(goal)
        # Allow TurtleBot up to 60 seconds to complete task
        # success = self.move_base.wait_for_result(rospy.Duration(60)) 
        state = self.move_base.get_state()
        print(state)
        while state==GoalStatus.ACTIVE or state==GoalStatus.PENDING:
            state=self.move_base.get_state()
            print(state)
        print("Searching")
        
        return  

    def returnHome(self):
        # Send a goal
        home_x=4.25
        home_y=3.5
        home_euler=np.pi/2
        home_quat_z=np.sin(home_euler/2.0)
        home_quat_w=np.cos(home_euler/2.0)


        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(home_x, home_y, 0.000),
                                         geometry_msgs.msg.Quaternion(0.0, 0, home_quat_z, home_quat_w))

        # Start moving
        self.move_base.send_goal(goal)
        # Allow TurtleBot up to 60 seconds to complete task
        # success = self.move_base.wait_for_result(rospy.Duration(60)) 
        state = self.move_base.get_state()
        print(state)
        while state==GoalStatus.ACTIVE or state==GoalStatus.PENDING:
            state=self.move_base.get_state()
            print(state)
        print("Home")
        
        self.scan=rospy.Subscriber('/scan', LaserScan, self.align)
        self.aligned=0
        while not self.aligned:
            rospy.sleep(0.2)
        vel=Twist()
        self.vel_pub.publish(vel)
        self.scan.unregister()
        return

    def align(self, data):
        vals_left=np.array(data.ranges[0:7])
        # print(np.array(vals_left))
        vals_right=np.array(data.ranges[-1:-6:-1])

        goal=0.15

        e_left=vals_left-goal
        e_right=vals_right-goal

        Kp=0.3
        Kd=0.1
        Ki=0.0

        ml=np.sum(Kp*e_left+Kd*(e_left-self.e_left_prev))
        mr=np.sum(Kp*e_right+Kd*(e_right-self.e_right_prev))

        if ml>0.10:
            ml=0.10
        elif ml<-0.10:
            ml=-0.10
        if mr>0.10:
            mr=0.10
        elif mr<-0.10:
            mr=-0.10

        self.e_left_prev=e_left
        self.e_right_prev=e_right

        # float speed_wish_right = (cmd_vel.angle*WHEEL_DIST)/2 + cmd_vel.speed;
        # float speed_wish_left = cmd_vel.speed*2-speed_wish_right;

        vel=Twist()
        vel.linear.x=(ml+mr)/2.0
        # ml_actual=ml+noise
        # vel.linear.x=(ml_actual+mr)/2.0
        vel.angular.z=(2*(mr-vel.linear.x))/0.306
        self.vel_pub.publish(vel)

        e_bound=0.1
        if (np.sum(e_left)<e_bound and np.sum(e_right)<e_bound):
            self.aligned=1
            vel.linear.x=0.0
            vel.angular.z=0.0
            self.vel_pub.publish(vel)

    def approachBox(self):
        cam_arm=self.rospy.Subscriber('/camera/rgb/image_raw', Image, armCameraApproach)
        self.approached=0
        while not self.approached:
            rospy.sleep(0.2)
        vel=Twist()
        self.vel_pub.publish(vel)
        self.scan.unregister()
        return  

    def armCameraApproach(self, data):
        img=self.br.imgmsg_to_cv2(data)
        blob=img[:,:,1]>100