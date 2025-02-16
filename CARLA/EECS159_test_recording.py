import carla
import time
import pygame
import numpy as np
import math
import cv2
import json
import datetime
from ultralytics import YOLO

model = YOLO("E:\\SeniorDesignProject\\models\\yolov10s.pt")

# Initialize pygame
pygame.init()

# Define display dimensions
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA Vehicle Camera View")
pygame.font.init()
font = pygame.font.Font(None, 50)

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get blueprint library
blueprint_library = world.get_blueprint_library()

vehicle_list = ["vehicle.carlamotors.firetruck", "vehicle.jeep.wrangler_rubicon"]
vehicle_1_model, vehicle_2_model = vehicle_list

spawn_point_1 = carla.Transform(carla.Location(x=-9.91, y=-217.0, z=0.3), carla.Rotation(yaw=90))
spawn_point_2 = carla.Transform(carla.Location(x=-9.63, y=-203.82, z=0.3), carla.Rotation(yaw=90))
spawn_points = [
    carla.Transform(carla.Location(x=-6.28, y=-228.7, z=0.3), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=-6.43, y=-210.25, z=0.3), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=-13.30, y=-193.05, z=0.3), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=-16.91, y=-201.6, z=0.3), carla.Rotation(yaw=90))
]
blueprints = [
    "vehicle.audi.a2",
    "vehicle.bmw.grandtourer",
    "vehicle.citroen.c3",
    "vehicle.audi.etron"
]

vehicle_1_bp = blueprint_library.find(vehicle_1_model)
vehicle_1 = world.try_spawn_actor(vehicle_1_bp, spawn_point_1)

vehicle_2_bp = blueprint_library.find(vehicle_2_model)
vehicle_2 = world.try_spawn_actor(vehicle_2_bp, spawn_point_2)

if not vehicle_1 or not vehicle_2:
    print("Error: Vehicles not spawned")
    exit()

# Spawn additional vehicles using spawn_points.
# Here, we use the vehicle_2 blueprint for traffic vehicles.
spawned_vehicles = []
for i in range(len(spawn_points)):
    veh = world.try_spawn_actor(blueprint_library.find(blueprints[i]), spawn_points[i])
    if veh:
        spawned_vehicles.append(veh)
    else:
        print("Failed to spawn a vehicle at:", spawn_points[i])
        

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", "110")

camera_transform = carla.Transform(carla.Location(x=3.8, z=2.3))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_1)

recording = False
video_writer = None
movement_enabled = False  # Vehicles will not move until 'Q' is pressed
show_bounding_boxes = False
last_inference_time = time.time()
latest_detections = []

def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed_mps * 2.23694  # Convert m/s to MPH

def get_vehicle_speed_mps(vehicle):
    """Return the vehicle speed in m/s."""
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def run_yolo_inference(frame):
    """Run YOLO inference and return detections."""
    results = model(frame)[0]  # Run inference
    detections = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        detections.append((x1, y1, x2, y2))
    return detections

def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes on the image."""
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    return frame

def process_image(image):
    global last_inference_time, latest_detections

    # Convert raw image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    frame = array[:, :, :3]  # Remove alpha channel
    frame = frame[:, :, ::-1]  # Convert BGR to RGB

    # Run YOLO inference every 0.2 seconds
    if time.time() - last_inference_time > 0.2:
        latest_detections = run_yolo_inference(frame)
        last_inference_time = time.time()

    # Draw bounding boxes if enabled
    if show_bounding_boxes:
        frame = draw_bounding_boxes(frame.copy(), latest_detections)

    # Convert frame to a pygame surface and display it
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surface, (0, 0))

    # --- Overlay Danger Zone ---
    # Base (bottom) points remain fixed at (370,720) and (910,720).
    # The top edge moves from the bottom up to (630,365) and (650,365) as speed increases.
    speed_mps = get_vehicle_speed_mps(vehicle_1)
    ratio = min(speed_mps / 10, 1)  # Ratio from 0 to 1 (30 m/s is max)
    # Compute top edge by linear interpolation:
    top_left_x  = 370 + 260 * ratio    # Moves from 370 to 630
    top_right_x = 910 - 260 * ratio    # Moves from 910 to 650
    top_y = 720 - 355 * ratio          # Moves from 720 to 365

    # Define polygon points (ensure integer values)
    danger_zone_points = [
        (370, 720),                     # Bottom left
        (910, 720),                     # Bottom right
        (int(top_right_x), int(top_y)), # Top right
        (int(top_left_x), int(top_y))   # Top left
    ]

    # Create a semi-transparent surface and draw the danger zone polygon
    danger_zone_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(danger_zone_surface, (255, 0, 0, 100), danger_zone_points)
    screen.blit(danger_zone_surface, (0, 0))
    # --- End Danger Zone Overlay ---

    # Display speed (in MPH)
    speed_text = font.render(f"Speed: {get_vehicle_speed(vehicle_1):.1f} MPH", True, (255, 255, 255))
    screen.blit(speed_text, (30, 30))

    pygame.display.flip()

    # Record frame if enabled
    if recording:
        frame_record = pygame.surfarray.array3d(screen)
        frame_record = np.rot90(frame_record, k=3)
        frame_record = np.flip(frame_record, axis=1)
        frame_record = cv2.cvtColor(frame_record, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_record)

camera.listen(lambda image: process_image(image))

def control_vehicle(vehicle, max_speed):
    """Control vehicle to follow waypoints at a target speed."""
    if not movement_enabled:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return
    speed = get_vehicle_speed(vehicle)
    transform = vehicle.get_transform()
    location = transform.location
    waypoint = world.get_map().get_waypoint(location)
    waypoints_ahead = waypoint.next(5.0)
    if not waypoints_ahead:
        return
    next_waypoint = waypoints_ahead[0]
    target_vector = next_waypoint.transform.location - location
    target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))
    yaw_diff = (target_yaw - transform.rotation.yaw + 180) % 360 - 180
    steer_correction = max(-0.3, min(0.3, yaw_diff * 0.005))
    throttle = 1.0 if speed < max_speed else 0.0
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_correction, brake=0.0))

running = True
clock = pygame.time.Clock()

try:
    while running:
        world.tick()
        if movement_enabled:
            control_vehicle(vehicle_1, 30)
            control_vehicle(vehicle_2, 25)
            control_vehicle(spawned_vehicles[0], 30)
            control_vehicle(spawned_vehicles[1], 30)
            control_vehicle(spawned_vehicles[2], 20)
            control_vehicle(spawned_vehicles[3], 20)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    if not recording:
                        print("Recording started")
                        recording = True
                        video_filename = f"carla_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (WIDTH, HEIGHT))
                    else:
                        print("Recording stopped")
                        recording = False
                        video_writer.release()
                elif event.key == pygame.K_q:
                    movement_enabled = True
                    print("Vehicles moving")
                elif event.key == pygame.K_b:
                    show_bounding_boxes = not show_bounding_boxes
                    print(f"Bounding Boxes {'Enabled' if show_bounding_boxes else 'Disabled'}")
                elif event.key == pygame.K_e:
                    print("Mouse position:", pygame.mouse.get_pos())

        clock.tick(30)

except KeyboardInterrupt:
    print("\nStopping script. Destroying vehicles...")

finally:
    camera.stop()
    vehicle_1.destroy()
    vehicle_2.destroy()
    if recording:
        video_writer.release()
    pygame.quit()
    print("Clean exit.")