import cv2
import numpy as np
import json
from moviepy.editor import VideoFileClip
from yolov5 import detect
import torch


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) 

file = 'LCW_high_hazard_1 (2).mp4'

real_world_points = np.array([
    [-1.0, 0.0],  # Bottom-left
    [1.0, 0.0],   # Bottom-right
    [1.0, 3.0],   # Top-right
    [-1.0, 3.0]   # Top-left
], dtype=np.float32)

# Image points
image_points = np.array([
    [486, 815],  # Bottom-left
    [937, 825],  # Bottom-right
    [877, 776],  # Top-right
    [553, 770]   # Top-left
], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(real_world_points, image_points)
W = 2.0  # Width of the car in meters


def update_danger_zone(speed):
    """
    Computes the real-world danger zone and maps it to the image using the homography matrix.
    """
    L = 1.5 * speed  # Danger zone length based on car speed
    # Real-world coordinates of the danger zone (bird's-eye view)
    world_points = np.array([
        [-W / 2, 0],    # Bottom-left
        [W / 2, 0],     # Bottom-right
        [W / 2, L],     # Top-right
        [-W / 2, L]     # Top-left
    ], dtype=np.float32)
    # Transform real-world points to image points
    image_zone_points = cv2.perspectiveTransform(world_points[None, :, :].copy(), H)
    return image_zone_points[0].astype(int)



def is_intersecting(danger_zone, bbox):
    """
    Checks for intersection between the danger zone polygon and a bounding box.
    """
    bbox_poly = np.array([
        [bbox[0], bbox[1]],  # Top-left
        [bbox[2], bbox[1]],  # Top-right
        [bbox[2], bbox[3]],  # Bottom-right
        [bbox[0], bbox[3]]   # Bottom-left
    ], dtype=np.int32)
    overlap = cv2.intersectConvexConvex(np.array(danger_zone, dtype=np.int32), bbox_poly)[0]
    return overlap > 0


def process_frame(frame, danger_zone):
    """
    Processes the frame to draw the danger zone and bounding boxes, and highlights warnings.
    """
    frame = frame.copy()
    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    results = model(frame_bgr)
    print(results)

    for result in results.xyxy[0]:  # Access detections from results
            x_min, y_min, x_max, y_max, confidence, cls = map(int, result[:6])
            bbox = [x_min, y_min, x_max, y_max]

            if is_intersecting(danger_zone, bbox):
                # Draw bounding box with warning
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, "WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Draw normal bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw the danger zone
    cv2.polylines(frame, [np.array(danger_zone, dtype=np.int32).copy()], isClosed=True, color=(255, 0, 0), thickness=2)
    return frame



# def load_bounding_boxes(txt_file, img_width, img_height):
#     """
#     Reads bounding boxes from a TXT file and scales them to absolute pixel values.

#     Args:
#         txt_file (str): Path to the text file containing bounding box coordinates.
#         img_width (int): Width of the image in pixels.
#         img_height (int): Height of the image in pixels.

#     Returns:
#         list: A list of bounding boxes in absolute pixel values [x_min, y_min, x_max, y_max].
#     """
#     bounding_boxes = []
#     with open(txt_file, 'r') as file:
#         for line in file:
#             x_min_rel, y_min_rel, x_max_rel, y_max_rel = map(float, line.strip().split())
#             # Scale relative coordinates to absolute pixel values
#             x_min = int(x_min_rel * img_width)
#             y_min = int(y_min_rel * img_height)
#             x_max = int(x_max_rel * img_width)
#             y_max = int(y_max_rel * img_height)
#             bounding_boxes.append([x_min, y_min, x_max, y_max])
#     return bounding_boxes

# Main function
if __name__ == "__main__":
    speed = 65

    # Update danger zone
    danger_zone = update_danger_zone(speed)

    # Load bounding boxes from JSON file
    text_file = "detections.txt"  # Replace with your JSON file path
    # bounding_boxes = load_bounding_boxes(text_file, 1492, 930)
    # print(bounding_boxes)

    # Process frame
    clip1 = VideoFileClip(file)

    white_clip = clip1.fl_image(lambda image: process_frame(image, danger_zone)) 
    white_clip.write_videofile('./output.mp4', audio=False)

    # cv2.imshow("Collision Warning", output_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
