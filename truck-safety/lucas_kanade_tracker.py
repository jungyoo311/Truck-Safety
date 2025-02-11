import numpy as np
import time
import cv2
class LucasKanadeTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.prev_speed = None
        self.findNew = True

    def initialize(self, frame, points=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #shitomashi corner detections. R = min(lambda1, lambda2)
        # 
        if points is not None:
            self.prev_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.prev_gray = gray
        self.prev_time = time.time()
        self.prev_speed = 0
        self.findNew = False

    # def track(self, frame):
    #     if self.prev_pts is None or len(self.prev_pts) == 0:
    #         self.findNew = True

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     if self.findNew:
    #         self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
    #         self.findNew = False  # Reset the flag
    #         self.prev_gray = gray
    #         return frame, 0, 0  # No movement detected yet
    #     new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
    #     print(f"INSIDE of LUCAS, prev_pts is: {self.prev_pts}")
    #     print(f"_*_*new_pts: {new_pts}, status: {status}, err:{err}")
        
    #     speed = 0
    #     acceleration = 0
    #     maxcorners = 4
    #     if new_pts is not None:
    #         good_new = new_pts[status == 1]
    #         good_old = self.prev_pts[status == 1]
    #         # print(f"***INSIDE of LUCAS, good_new is: {good_new}, good_old is: {good_old}")
    #         if len(good_new) < maxcorners: #gpt hard code to 3
    #             self.findNew = True
    #             return frame, 0, 0
    #         movement_distances = [np.linalg.norm(new - old) for new, old in zip(good_new, good_old)]
    #         # print(f"INSIDE of LUCAS, movement_distances is: {movement_distances}")
    #         if movement_distances and np.any(movement_distances):
                
    #             avg_movement = np.mean(movement_distances)
    #             # Time calculations
    #             current_time = time.time()
    #             time_elapsed = current_time - self.prev_time if self.prev_time else 1
                
    #             # Compute relative speed (pixels/sec)
    #             speed = avg_movement / time_elapsed
                
    #             # Compute acceleration (change in speed over time)
    #             if self.prev_speed is not None:
    #                 acceleration = (speed - self.prev_speed) / time_elapsed

    #             self.prev_time = current_time
    #             self.prev_speed = speed

    #         for new in good_new:
    #             a, b = new.ravel()
    #             frame = cv2.circle(frame, (int(a), int(b)), 5, (255, 0, 0), -1)
    #     else:
    #         self.findNew = True # Trigger new feature poin search
    #     self.prev_gray = gray.copy()
    #     if new_pts is not None:
    #         self.prev_pts = new_pts.reshape(-1, 1, 2) if new_pts is not None else None
    #     if new_pts is not None:
    #         self.prev_pts = good_new.reshape(-1, 1, 2)
    #     return frame, speed, acceleration
    def track(self, frame):
            """Track points using Lucas-Kanade Optical Flow."""
            if self.prev_pts is None or self.prev_gray is None:
                print("[DEBUG] Tracker has no previous frame.")
                return frame, 0, 0  # No movement detected

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Lucas-Kanade optical flow
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

            if new_pts is None or status is None:
                print("[ERROR] Optical flow failed.")
                return frame, 0, 0

            # Compute speed & acceleration
            distances = np.linalg.norm(new_pts - self.prev_pts, axis=2).flatten()
            speed = np.mean(distances)  # Average movement
            acceleration = speed - np.mean(distances)  # Change in speed (dummy acceleration formula)

            # Print debug info
            print(f"INSIDE of LUCAS, prev_pts is: {self.prev_pts}")
            print(f"_*_*new_pts: {new_pts}, status: {status}, err:{err}")
            print(f"[DEBUG] Speed: {speed:.2f} px/s, Acceleration: {acceleration:.2f} px/s²")

            # Update tracker state
            self.prev_gray = gray
            self.prev_pts = new_pts[status == 1].reshape(-1, 1, 2)  # Keep only good points

            return frame, speed, acceleration