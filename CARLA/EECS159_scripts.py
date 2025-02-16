import numpy as np
import cv2
import pygame

CAR_WIDTH = 2.0  # Approximate width of the vehicle in meters

def draw_danger_zone_pygame(surface, danger_zone, color=(255, 0, 0), width=2):
    """
    Draws the danger zone polygon onto a pygame surface.

    Args:
        surface (pygame.Surface): The pygame surface to draw on.
        danger_zone (np.ndarray): Polygon of the danger zone as an array of points.
        color (tuple): RGB color of the polygon.
        width (int): Line thickness of the polygon.
    """
    # Convert danger_zone points to pygame-friendly format
    points = [(int(x), int(y)) for x, y in danger_zone]
    
    # Draw the polygon on the pygame surface
    pygame.draw.polygon(surface, color, points, width)
    
def compute_danger_zone(speed, ego_vehicle_transform):
    """
    Computes the danger zone polygon based on the car's speed and position.

    Args:
        speed (float): Speed of the ego vehicle in m/s.
        ego_vehicle_transform (carla.Transform): Transform of the ego vehicle.

    Returns:
        danger_zone (np.ndarray): Polygon of the danger zone in the image plane.
    """
    L = 1.5 * 3.6 * speed  # Length of the danger zone
    W = CAR_WIDTH    # Width of the danger zone

    # Front of the vehicle as the starting point
    vehicle_location = ego_vehicle_transform.location
    yaw = np.radians(ego_vehicle_transform.rotation.yaw)

    # Compute real-world coordinates of the danger zone corners
    front_x = vehicle_location.x + np.cos(yaw) * L
    front_y = vehicle_location.y + np.sin(yaw) * L

    rear_x = vehicle_location.x
    rear_y = vehicle_location.y

    corners = np.array([
        [rear_x - W / 2 * np.sin(yaw), rear_y + W / 2 * np.cos(yaw)],  # Bottom-left
        [rear_x + W / 2 * np.sin(yaw), rear_y - W / 2 * np.cos(yaw)],  # Bottom-right
        [front_x + W / 2 * np.sin(yaw), front_y - W / 2 * np.cos(yaw)],  # Top-right
        [front_x - W / 2 * np.sin(yaw), front_y + W / 2 * np.cos(yaw)]   # Top-left
    ], dtype=np.float32)

    return corners

def draw_danger_zone(frame, danger_zone, color=(255, 0, 0)):
    """
    Draws the danger zone on the video frame.

    Args:
        frame (np.ndarray): The video frame.
        danger_zone (np.ndarray): Danger zone polygon.
        color (tuple): Color for the danger zone outline.
    """
    danger_zone_pts = danger_zone.astype(int)
    cv2.polylines(frame, [danger_zone_pts], isClosed=True, color=color, thickness=2)

def is_intersecting(danger_zone, bbox):
    """
    Checks if a bounding box intersects the danger zone.

    Args:
        danger_zone (np.ndarray): Danger zone polygon.
        bbox (list): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
        bool: True if the bounding box intersects the danger zone, False otherwise.
    """
    bbox_poly = np.array([
        [bbox[0], bbox[1]],  # Top-left
        [bbox[2], bbox[1]],  # Top-right
        [bbox[2], bbox[3]],  # Bottom-right
        [bbox[0], bbox[3]]   # Bottom-left
    ], dtype=np.int32)

    overlap = cv2.intersectConvexConvex(danger_zone.astype(np.int32), bbox_poly)[0]
    return overlap > 0