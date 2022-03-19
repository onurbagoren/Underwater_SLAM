import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import sys

class CameraListener:
    def __init__(self) -> None:
        self.file_name = f'{sys.path[0]}/camera_times.txt'

    
    def callback(self, data):
        step = data.step
        img_data = data.data

        header = data.header
        time = header.stamp

        file = open(self.file_name, 'a')
        file.write(f'{time}\n')




def main():
    rospy.init_node('camera_listener', anonymous=True)
    listener = CameraListener()
    rospy.Subscriber('/camera/image_raw', Image, listener.callback)
    rospy.spin()

if __name__ == '__main__':
    main()    


