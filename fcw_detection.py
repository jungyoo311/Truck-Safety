import json
import cv2
import numpy as np

# Load JSON data from file
with open("detections.json", "r") as f:
    data = json.load(f)

# Open video file
video_path = "camera_video.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_count / fps  # Total duration in seconds

# Output video setup
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Real-world coordinates for homography transformation (same as in the first script)
real_world_points = np.array([
    [-1.0, 0.0],  # Bottom-left
    [1.0, 0.0],   # Bottom-right
    [1.0, 140],   # Top-right
    [-1.0, 140]   # Top-left
], dtype=np.float32)

# Image points from the first frame (same as in the first script)
image_points = np.array([
    [355, 720],  # Bottom-left
    [765, 720],  # Bottom-right
    [654, 493],  # Top-right
    [532, 493]   # Top-left
], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(real_world_points, image_points)

# Function to calculate dynamic danger zone based on speed and transform to real-world coordinates
def get_danger_zone(speed_kmh):
    # Calculate the real-world danger zone length based on speed
    zone_length = 3*speed_kmh # Dynamic scaling of length
    zone_width = 2  # Fixed width for the danger zone

    # Real-world coordinates for the danger zone (in meters)
    real_world_zone = np.array([
        [-zone_width / 2, 0],       # Bottom-left
        [zone_width / 2, 0],        # Bottom-right
        [zone_width / 2, zone_length],  # Top-right
        [-zone_width / 2, zone_length]  # Top-left
    ], dtype=np.float32)

    # Transform the real-world zone to image coordinates using homography
    image_zone_points = cv2.perspectiveTransform(real_world_zone[None, :, :].copy(), H)

    # Return the points of the danger zone as integer values
    return image_zone_points[0].astype(int)

# Process frames
frame_index = 0
data_index = 0  # To track JSON entry processing
total_detections = len(data['data'])
print(total_detections)

# Dictionary to store the frame count for each detection to track how long to show the danger warning
detection_warning_frames = {}

# Number of frames to display the danger warning for each detection
warning_duration = 30  # Show warning for 30 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no more frames

    # Compute current video time (in seconds)
    current_time = (frame_index / fps)
    # print(current_time*3)
    


    # Process JSON entries that match the current frame's time
    while data_index < total_detections and ((data["data"][data_index]["time"] - data['metadata']['simulation_time_start']) <= current_time): 
        frame_data = data["data"][data_index]
        speed_kmh = frame_data.get("speed_kmh", 0)
        detections = frame_data.get("detections", [])
        danger_zone = get_danger_zone(speed_kmh)

        # Draw danger zone (red polygon)
        if danger_zone is not None:
            cv2.polylines(frame, [np.array(danger_zone, dtype=np.int32).copy()], isClosed=True, color=(0, 0, 255), thickness=2)

        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            class_name = detection.get("class", "Unknown")
            confidence = detection.get("confidence", 0)

            x_min, y_min, x_max, y_max = map(int, bbox)
            # Check if the detection is inside the danger zone
            if x_max > danger_zone[0][0] and x_min < danger_zone[2][0] and y_max > danger_zone[0][1] and y_min < danger_zone[2][1]:
                # If detection enters the danger zone, set its warning frame count
                print("danger")
                if detection.get("id") not in detection_warning_frames:
                    detection_warning_frames[detection.get("id")] = warning_duration  # Initialize with warning_duration frames
                
                # Decrement the warning frame count and display "DANGER" if still within the warning period
                if detection_warning_frames[detection.get("id")] > 0:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(frame, f"DANGER: {class_name}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    detection_warning_frames[detection.get("id")] -= 1  # Decrease warning frame count
            else:
                # If detection is not in the danger zone, reset its warning frame count
                if detection.get("id") in detection_warning_frames:
                    del detection_warning_frames[detection.get("id")]

            # If not inside the danger zone, display the regular bounding box with confidence
            if detection.get("id") not in detection_warning_frames:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        data_index += 1  # Move to next detection entry
    print(data_index)
    # Write processed frame
    out.write(frame)
    frame_index += 1
print(frame_index)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as output_video.mp4")
