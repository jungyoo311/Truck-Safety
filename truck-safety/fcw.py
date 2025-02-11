# my version
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo, sys, time
# Adjust path to the new location of hailo-apps-infra
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hailo-apps-infra")))

from hailo_rpi_common_my import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline_my import GStreamerDetectionApp

import argparse
from lucas_kanade_tracker import LucasKanadeTracker
tracker = LucasKanadeTracker()
DANGER_ACCELERATION_THRESHOLD = -5 
# ---------
# argument parser for  cmd line inputs
# ---------
# parser = argparse.ArgumentParser(description="Run object detction on rpi cam")
# parser.add_argument("--input", type=str, default="rpi", help="Input source: 'rpi' for Raspberry Pi camera")
# parser.add_argument("--camera-index", type=int, default=0, help="Select Raspberry Pi camera (0: Noir, 1: Wide)")
# args = parser.parse_args()
# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function

# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "
# -----------------------------------------------------------------------------------------------
# app_callback: given example
# -----------------------------------------------------------------------------------------------

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
    print(f"_*_*format: {format}, width: {width}, height: {height}")
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
    print(frame)
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback_gpt(pad, info, user_data):
    user_data.use_frame = True
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    # print(f"_*_*format: {format}, width: {width}, height: {height}")
    frame = get_numpy_from_buffer(buffer, format, width, height)
    # if frame is not None:
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    print(f"[DEBUG] Total Detections: {len(detections)}")

    # Find the frontmost vehicle
    min_y = float('inf')
    front_vehicle_centroid = None
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()  # Fetch bbox object
        confidence = detection.get_confidence()
        # print(f"label: {label}, bbox: {bbox.xmin}, confidence: {confidence}")

        if bbox is not None and label in ["car", "truck", "motorcycle"] and confidence >= 0.6:
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            # print(f"[DEBUG] Detection: Label={label}, BBox=({x1}, {y1}, {x2}, {y2}), Confidence={confidence:.2f}")
            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2    
            # print(f"[DEBUG] Selected Vehicle Centroid: ({centroid_x}, {centroid_y})")

            if y1 < min_y:  # Closer to top means further ahead
                min_y = y1
                front_vehicle_centroid = (centroid_x, centroid_y)

    if front_vehicle_centroid:
        print(f"[DEBUG] Front vehicle centroid: {front_vehicle_centroid}") #print stmt

        if tracker.prev_pts is None or len(tracker.prev_pts) == 0:
            print("[DEBUG] Tracker was empty, initializing with front vehicle.")
            tracker.initialize(frame, [front_vehicle_centroid])
        else:
            # Track motion and compute speed/acceleration
            frame, speed, acceleration = tracker.track(frame)

            print(f"[DEBUG] Speed: {speed:.2f} px/s, Acceleration: {acceleration:.2f} px/s^2")
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot at centroid

            if speed < 0.1:
                print("[WARNING] Tracking is stuck! The object might not be moving or tracking is failing.")
            
    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        user_data.set_frame(frame)
    # print(f"[DEBUG] Frame Count: {user_data.get_count()}")
    # print(f"[DEBUG] Detected Objects: {len(detections)}")

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()    
    # if args.input == "rpi":
    #     # Use libcamera for Raspberry Pi cameras
    #     camera_index = args.camera_index
    #     input_source = f"libcamera --camera {camera_index}"
    # else:
    #     input_source = args.input  # Other sources can be defined


    app = GStreamerDetectionApp(app_callback_gpt, user_data)
    app.run()
