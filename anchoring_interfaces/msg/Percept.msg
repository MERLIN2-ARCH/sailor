
# object detection
int32 class_id
string class_name
float32 class_score
vision_msgs/BoundingBox2D bounding_box

# physical features
geometry_msgs/Vector3 position
geometry_msgs/Vector3 size

# color histogram
float32[] histogram
int32 histogram_bins_per_channel

# image
sensor_msgs/Image image
