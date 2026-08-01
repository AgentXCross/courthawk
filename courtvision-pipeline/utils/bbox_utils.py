def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    c = a * a + b * b
    return c ** 0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    bottom_y = max(y1, y2)
    return (center_x, bottom_y)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    keypoint_index = 0
    for index in keypoint_indices:
        keypoint = (keypoints[index * 2], keypoints[index * 2 + 1])
        distance = abs(point[1] - keypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            keypoint_index = index
    return keypoint_index

def get_height_of_bbox(bbox):
    _, y1, _, y2 = bbox
    return abs(y2 - y1)

def measure_xy_distance(p1, p2): 
    return (p1[0] - p2[0], p1[1] - p2[1])