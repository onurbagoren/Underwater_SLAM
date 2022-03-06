# UW_SLAM
## Setting up ROS
- If you have Ubuntu 20.04 or Debian Buster, you can install ROS by following the link:
http://wiki.ros.org/noetic/Installation/Debian
- If you have Ubuntu 18.04 or Debian Buster, you can install ROS by following the link: http://wiki.ros.org/melodic/Installation

## Cloning this directory
- This directory can be treated as the `catkin_ws` directory that are explained in most ROS tutorials. Simply run the following command in a directory that is not the `catkin_ws` directory:
```
git clone git@github.com:onurbagoren/UW_SLAM.git
```
- Run the following commands to compile the ROS packages:
```
cd UW_SLAM
catkin_make
source devel/setup.bash
```

## Visualizing the data
- Initally, make sure that you create a data directory in the `src/cirs_girona_cala_ciuda` file by running the following command: `mkdir -p src/cirs_girona_cala_ciuda/data`
- Then, move the data in https://drive.google.com/drive/folders/1lM9ZxQg0g3F8UxILcddMr8hU6OcUg3Hb to the `src/cirs_girona_cala_ciuda/data` directory.
- Then, run the following command to visualize the data:
```
roslaunch cirs_girona_cala_ciuda play_data.launch
```
- This should launch an RViz window that shows the robot, the robots odometry over time, and the sonar readings over time.


## Relevant Notes
- In the `src/cirs_girona_cala_ciuda/launch/play_data.launch` file, you can find notes regarding which line to comment of uncomment, depending on your ROS distribution. This is not a major issue, as the visualization will work - but will get rid of the error message that may appear when launching the file.