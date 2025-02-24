# my version
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo, sys, time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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
        
        #plotting values
        self.speeds = []
        self.times = []
        self.start_time = time.time()
        self.log_filename = "speed_log.csv"

    def new_function(self):  # New function example
        return "The meaning of life is: "
# -----------------------------------------------------------------------------------------------
# app_callback: given example
# -----------------------------------------------------------------------------------------------
    def visualize(self, speed):
        current_time = time.time() - self.start_time
        self.speeds.append(speed)
        self.times.append(current_time)

        # Log speed to CSV
        with open(self.log_filename, "a") as f:
            f.write(f"{current_time}, {speed}\n")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

'''
 * To add this filter to your pipeline, you would typically include the following in your code:
 * remover_so_path = os.path.join(self.current_path, '../resources/libremove_labels.so')
 * # adjust the path to the labels_to_remove.txt file
 * labels_to_remove_path = os.path.join(self.current_path, '../resources/labels_to_remove.txt')
 * 'hailofilter name=remover so-path=remover_so_path config-path=labels_to_remove_path qos=false ! '
'''

def app_callback_custom(pad, info, user_data):
    
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
    front_vehicle_detection = None
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()  # Fetch bbox object
        confidence = detection.get_confidence()
        # print(f"label: {label}, bbox: {bbox.xmin}, confidence: {confidence}")

        if bbox is not None and label in ["car", "truck"] and confidence >= 0.6:
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            # print(f"[DEBUG] Detection: Label={label}, BBox=({x1}, {y1}, {x2}, {y2}), Confidence={confidence:.2f}")
            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2    
            # print(f"[DEBUG] Selected Vehicle Centroid: ({centroid_x}, {centroid_y})")

            if y1 < min_y:  # Closer to top means further ahead
                min_y = y1
                front_vehicle_centroid = (centroid_x, centroid_y)
                front_vehicle_detection = detection

    #remove all objects from the ROI so hailo_overlay won't draw them
    for det in list(detections):
        print(det)
        roi.remove_object(det) #changed
        print(f"[DEBUG] Frame deleted") #print stmt
    print(f"front_vehicle_centroid: {front_vehicle_centroid}")
    if front_vehicle_detection:
        roi.add_object(front_vehicle_detection)
        print("[DEBUG] FRONT vehicle re-added to ROI")
        print(f"[DEBUG] Front vehicle centroid: {front_vehicle_centroid}") #print stmt

        if tracker.prev_pts is None or len(tracker.prev_pts) == 0:
            print("[DEBUG] Tracker was empty, initializing with front vehicle.")
            tracker.initialize(frame, [front_vehicle_centroid])
        else:
            # Track motion and compute speed/acceleration
            frame, speed, acceleration = tracker.track(frame)
            h, w, _ = frame.shape
            x_text_ = w - 300  
            y_speed_ = 40      
            y_accel_ = 80
            y_warning_ = 120
            print(f"[DEBUG] Speed: {speed:.2f} px/s, Acceleration: {acceleration:.2f} px/s^2")
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot at centroid
            cv2.putText(
                frame,
                f"Speed={speed:.2f} px/s",
                (x_text_, y_speed_),           # <- origin (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2                             
            )
            cv2.putText(
                frame,
                f"Acc={acceleration:.2f} px/s^2",
                (x_text_, y_accel_),           
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2
            )
            #user_data.visualize(speed) #log speed
            if speed < 0.1:
                print("[WARNING] Tracking is stuck! The object might not be moving or tracking is failing.")
                cv2.putText(
                frame,
                f"BRAKE!!!",
                (x_text_, y_warning_),           
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2
            )
            
    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        user_data.set_frame(frame)
    # print(f"[DEBUG] Frame Count: {user_data.get_count()}")
    # print(f"[DEBUG] Detected Objects: {len(detections)}")

    return Gst.PadProbeReturn.OK

############################################################################
# Basic Lane Detection Helpers
############################################################################
def detect_lane_mask(frame_bgr):
    """
    Very basic lane detection mask: 
      1) Focus on bottom half of the image 
      2) Use Canny edges + Hough lines 
      3) Convert lines to a binary mask
    """
    h, w = frame_bgr.shape[:2]

    # Convert to gray, build an ROI focusing on bottom half
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    roi_vertices = np.array([[
        (0, h),
        (0, int(h * 0.5)),
        (w, int(h * 0.5)),
        (w, h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi = cv2.bitwise_and(gray, mask)

    # Edges + Hough lines
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=100)

    # Draw lines on a blank canvas
    line_img = np.zeros_like(frame_bgr)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Convert to a lane_mask
    line_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(line_gray, 1, 255, cv2.THRESH_BINARY)
    return lane_mask

def get_lane_polygon(lane_mask):
    """
    Convert lane_mask into a polygon. 
    For simplicity, we pick the largest contour.
    """
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Choose the largest contour
    largest = max(contours, key=cv2.contourArea)
    smallast = min(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 4, True)
    return polygon

def point_in_polygon(cx, cy, polygon):
    """
    Returns True if (cx, cy) is inside polygon using cv2.pointPolygonTest.
    """
    dist = cv2.pointPolygonTest(polygon, (cx, cy), False)
    # dist >= 0 => inside or on the boundary
    return (dist >= 0)

def app_callback_fcw(pad, info, user_data):
    """
    1. Convert GStreamer buffer to NumPy (RGB).
    2. Run or retrieve lane detection results (e.g. from UFLD).
    3. Draw lane lines on a debug overlay.
    4. Extract Hailo detection results from the buffer.
    5. Convert them to your tracker’s input format (ByteTrack or Lucas-Kanade).
    6. Find front vehicle in-lane, track it, and annotate.
    7. Show final debug frame in user_data.set_frame().
    """
    # -----------
    # Step 1: Extract frame data
    # -----------
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    frame_rgb = get_numpy_from_buffer(buffer, format, width, height)
    if frame_rgb is None:
        return Gst.PadProbeReturn.OK

    # Convert to BGR for normal OpenCV usage
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    debug_frame = frame_bgr.copy()

    # -----------
    # Step 2: Lane Detection (UFLD or simpler approach)
    #    If you have an actual UFLDProcessing object from the lane example, it might look like:
    #       lanes = ufld_processing.get_coordinates(inference_output)
    #       for lane in lanes: draw circles or polylines
    #    We'll do a simpler "detect_lane_mask => largest contour => polygon" for demonstration
    # -----------
    lane_mask = detect_lane_mask(frame_bgr)    # from your simpler code
    lane_polygon = get_lane_polygon(lane_mask) # largest contour => polygon
    if lane_polygon is not None:
        # Draw the polygon on our debug overlay
        cv2.polylines(debug_frame, [lane_polygon], isClosed=True, color=(0, 255, 0), thickness=3)

    # -----------
    # Step 3: Get Hailo Detections
    # -----------
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    # Optionally remove them so Hailo overlay doesn't re-draw everything:
    for det in list(detections):
        roi.remove_object(det)

    # -----------
    # Step 4: Filter for “front vehicle in lane”
    #    If you want ByteTrack: you’d gather all bounding boxes, class IDs, scores into arrays,
    #    then call your ByteTrack update. For demonstration, we’ll just pick the single front vehicle,
    #    as in your original code, but also confirm it’s inside lane_polygon.
    # -----------
    front_det = None
    min_y = float('inf')

    if lane_polygon is not None:
        for det in detections:
            label = det.get_label()
            score = det.get_confidence()
            if label in ["car", "truck"] and score >= 0.6:
                bbox = det.get_bbox()
                x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
                x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Check if the center is inside the lane polygon:
                if point_in_polygon(cx, cy, lane_polygon):
                    # pick the topmost (front) bounding box
                    if y1 < min_y:
                        min_y = y1
                        front_det = det

    if front_det:
        # re‐add front vehicle so hailo_overlay or other modules can show it
        roi.add_object(front_det)

        # -----------
        # Step 5: Tracking (Lucas-Kanade or ByteTrack).
        #    For ByteTrack, you'd do something like:
        #       detections_for_tracker = convert_detections_to_supervision(front_det)
        #       tracked = byte_tracker.update(detections_for_tracker)
        #    or for Lucas-Kanade:
        # -----------
        fbbox = front_det.get_bbox()
        fx1 = int(fbbox.xmin() * width)
        fy1 = int(fbbox.ymin() * height)
        fx2 = int(fbbox.xmax() * width)
        fy2 = int(fbbox.ymax() * height)
        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        h, w, _ = debug_frame.shape
        x_text = w - 300  
        y_speed = 40      
        y_accel = 80

        # Draw a debug dot
        cv2.circle(debug_frame, (fcx, fcy), 5, (0, 0, 255), -1)
        
        # If using your existing Lucas‐Kanade:
        if tracker.prev_pts is None or len(tracker.prev_pts) == 0:
            tracker.initialize(frame_bgr, [(fcx, fcy)])
        else:
            _, speed, accel = tracker.track(frame_bgr)
            #text overlay
            cv2.putText(
                debug_frame,
                f"Speed={speed:.2f} px/s",
                (x_text, y_speed),           # <- origin (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2                             
            )
            cv2.putText(
                debug_frame,
                f"Acc={accel:.2f} px/s^2",
                (x_text, y_accel),           # just another (x, y) point slightly below the speed
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2
            )
            print(f"[TRACK DEBUG] Speed={speed:.2f} Acc={accel:.2f}")
            user_data.visualize(speed) #record time, speed in separate csv file
        

    # -----------
    # Step 6: Send the debug_frame to the display
    # -----------
    if user_data.use_frame:
        user_data.set_frame(debug_frame)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    user_data.use_frame = True



    app = GStreamerDetectionApp(app_callback_fcw, user_data)
    app.run()
