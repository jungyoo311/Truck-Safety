import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp
import time, math
# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.last_time = time.time()

    # def new_function(self):  # New function example
    #     return "The meaning of life is: "
    def calculate_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        if elapsed_time > 0:
            return 1.0 / elapsed_time
        return 0.0

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Define the visual representation of the 270° 15m ROI
    range_y = int(height * 0.5)  # Simulated 15m as half the frame height
    left_x = int(width * 0.125)  # Start of 270° FOV on the left side
    right_x = int(width * 0.875)  # End of 270° FOV on the right side
    center = (width // 2, height)          # Center at the bottom middle of the frame
    axes = (int(width * 0.75), int(height * 0.5))  # Width and height to represent
    # Draw the bounding box for the 270° field of view and 15m range
    if user_data.use_frame:
        cv2.ellipse(
            frame, 
            center, 
            axes, 
            angle=0, 
            startAngle=-135, 
            endAngle=135, 
            color=(255, 255, 255),  # White color for the arc
            thickness=4             # Bold line thickness
        )
        cv2.putText(frame, "270° FOV, 15m Range", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Frame with ROI", frame)
        cv2.waitKey(1)

        print(f"Left X: {left_x}, Right X: {right_x}, Range Y: {range_y}, Width: {width}, Height: {height}")

    
    # Parse the detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            detection_count += 1
        if label == "vehicle":
            # Calculate approximate distance and angle
            center_x = (bbox.left + bbox.right) / 2
            center_y = (bbox.top + bbox.bottom) / 2
            distance = 15 * (1 - center_y / height)  # Simulated distance
            angle = math.degrees(math.atan2(center_x - width / 2, height))

            # Filter vehicles within 15m range and 270° FOV
            if distance <= 15 and -135 <= angle <= 135:
                # Draw bounding box and label for valid detections
                cv2.rectangle(frame, (int(bbox.left), int(bbox.top)), (int(bbox.right), int(bbox.bottom)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {distance:.2f}m", (int(bbox.left), int(bbox.top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection_count += 1
    if user_data.use_frame:
        fps = user_data.calculate_fps()
        fps_txt = f"FPS:{fps:.1f}"
        cv2.rectangle(frame, (5, 65), (180, 100), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, fps_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        # cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        # cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
