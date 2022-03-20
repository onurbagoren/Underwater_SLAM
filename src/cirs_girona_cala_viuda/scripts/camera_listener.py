import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import sys

class CameraListener:
    def __init__(self) -> None:
        self.file_name = f'{sys.path[0]}/camera_times.txt'
        self.time = 0

    
    def callback(self, data):
        step = data.step
        img_data = data.data

        header = data.header
        time = header.stamp

        # file = open(self.file_name, 'a')
        # file.write(f'{time}\n')
        print(f'Prev time: {self.time}')
        print(f'Time raw:{time}')
        print(f'Time sec:{time.secs}')
        print(f'Time nsec:{time.nsecs}')
        time_diff = (time.secs * 1e+9 + time.nsecs) - self.time
        print(f'Time diff:{time_diff}')
        self.time = time.secs * 1e+9 + time.nsecs
        print()

def main():
    rospy.init_node('camera_listener', anonymous=True)
    listener = CameraListener()
    rospy.Subscriber('/camera/image_raw', Image, listener.callback)
    rospy.spin()

if __name__ == '__main__':
    main()    


