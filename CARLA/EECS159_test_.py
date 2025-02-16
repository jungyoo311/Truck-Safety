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
# Set up fonts
pygame.font.init()
font = pygame.font.Font(None, 50)  # Font size 50 for speed display

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Enable synchronous mode for stability
settings = world.get_settings()
settings.synchronous_mode = True  # Enables sync mode
settings.fixed_delta_seconds = 0.05  # Ensures consistent updates
world.apply_settings(settings)

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Define vehicle models (change these to your preferred models)
vehicle_list = ['vehicle.carlamotors.carlacola',    # 0
                "vehicle.mercedes.coupe",           # 1
                'vehicle.carlamotors.firetruck'
            ]
vehicle_1_model = vehicle_list[2]
vehicle_2_model = vehicle_list[1]

# Define spawn locations (modify coordinates as needed)
spawn_point_1 = carla.Transform(carla.Location(x=-9.91, y=-217.0, z=0.3), carla.Rotation(yaw=90))
spawn_point_2 = carla.Transform(carla.Location(x=-9.63, y=-203.82, z=0.3), carla.Rotation(yaw=90))

# Spawn vehicle 1
vehicle_1_bp = blueprint_library.find(vehicle_1_model)
vehicle_1 = world.try_spawn_actor(vehicle_1_bp, spawn_point_1)

# Spawn vehicle 2
vehicle_2_bp = blueprint_library.find(vehicle_2_model)
vehicle_2 = world.try_spawn_actor(vehicle_2_bp, spawn_point_2)

# Set up video recording
recording = False
video_writer = None

# Initialize data logging
log_data = {
    "metadata": {
        "simulation_time_start": None,
        "logging_interval": 0.1,  # Log every 0.1s
        "map_name": world.get_map().name,
        "vehicle_type": vehicle_1_model,
        "sensor_config": {
            "camera_resolution": (WIDTH, HEIGHT),
            "camera_fps": 30
        }
    },
    "data": []  # Store vehicle speed, acceleration, and YOLO detections
}

# Get simulation start time
log_data["metadata"]["simulation_time_start"] = datetime.datetime.now().isoformat()

# Function to calculate acceleration
previous_velocity = None
previous_time = None

# Ensure both vehicles spawned correctly
if not vehicle_1:
    print("Error: 1 not spawned")
    exit()

if not vehicle_2:
    print("Error: 2 not spawned")
    exit()

print(f"Spawned {vehicle_1_model} at {spawn_point_1.location}")
print(f"Spawned {vehicle_2_model} at {spawn_point_2.location}")

# Attach a camera to vehicle_1 (First Person Perspective)
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", "110")  # Wide-angle field of view

camera_transform = carla.Transform(carla.Location(x=3.8, z=2.3))  # Slightly above dashboard
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_1)

# Run YOLO detection
def run_yolo_detection(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    results = model(array)  # Run YOLO inference
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = model.names[cls]  # Get class name
            detections.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})

            # Draw bounding boxes on screen
            cv2.rectangle(array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(array, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return array, detections

def get_vehicle_acceleration(vehicle):
    global previous_velocity, previous_time
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    current_time = time.time()
    acceleration = 0.0
    
    if previous_velocity is not None and previous_time is not None:
        acceleration = (current_speed - previous_velocity) / (current_time - previous_time)
    
    previous_velocity = current_speed
    previous_time = current_time
    
    return acceleration

# Function to process images from the camera
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # Convert to 4-channel (RGBA)
    array = array[:, :, :3]  # Remove alpha channel
    array = array[:, :, ::-1]  # Convert BGR to RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    
    # Display speed in Pygame
    speed_mph = get_vehicle_speed(vehicle_1)
    speed_text = font.render(f"Speed: {speed_mph:.1f} MPH", True, (255, 255, 255))
    screen.blit(speed_text, (30, 30))  # Position the text at the top-left corner
    
    pygame.display.flip()
    # Video recording
    if recording:
        frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        video_writer.write(frame)

# Attach the function to the camera
camera.listen(lambda image: process_image(image))

# Function to log data
def log_vehicle_data():
    global log_data
    speed_mph = get_vehicle_speed(vehicle_1)
    acceleration = get_vehicle_acceleration(vehicle_1)
    
    log_data["data"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "speed_mph": speed_mph,
        "acceleration": acceleration,
        "yolo_detections": []  # Placeholder for YOLO detections
    })

# Function to calculate vehicle speed in MPH
def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Speed in meters per second
    speed_mph = speed_mps * 2.23694  # Convert to miles per hour
    return speed_mph

# Speed limit (adjust this value)
# MAX_SPEED_MPH = 60.0  # Vehicles will not exceed 30 MPH
max_speeds = [40, 45, 50, 55, 60, 65, 70]

# Function to control speed and keep the vehicle inside the lane
def control_vehicle(vehicle, MAX_SPEED_MPH):
    # Get current speed
    speed_mph = get_vehicle_speed(vehicle)

    # Get vehicle transform and waypoint
    transform = vehicle.get_transform()
    location = transform.location
    waypoint = world.get_map().get_waypoint(location)

    # Get a further waypoint to smooth turns
    waypoints_ahead = waypoint.next(5.0)  # Look further ahead
    if not waypoints_ahead:
        return
    next_waypoint = waypoints_ahead[0]

    # Calculate steering based on lane following
    target_vector = next_waypoint.transform.location - location
    target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))

    # Normalize yaw difference to [-180, 180] range to prevent large jumps
    yaw_diff = (target_yaw - transform.rotation.yaw + 180) % 360 - 180

    # Reduce the steering sensitivity
    steer_correction = yaw_diff * 0.005  # Reduced from 0.02 to 0.005 for smooth turns

    # Ensure steering stays in valid range (-1 to 1)
    steer_correction = max(-0.3, min(0.3, steer_correction))  # Limit the range

    # Adjust throttle to maintain speed limit
    if speed_mph < MAX_SPEED_MPH:
        throttle = 1  # Reduce acceleration to avoid jerky movements
    else:
        throttle = 0.0  # Stop accelerating if at speed limit

    # Apply control to vehicle
    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer_correction
    control.brake = 0.0  # No braking unless needed
    vehicle.apply_control(control)

# Main game loop
running = True
clock = pygame.time.Clock()
log_timer = time.time()

try:
    while running:
        world.tick()  # Sync simulation
        control_vehicle(vehicle_1, 60)
        control_vehicle(vehicle_2, 20)

         # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Toggle recording when "R" is pressed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
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

        # Log data at the defined interval
        if time.time() - log_timer > log_data["metadata"]["logging_interval"]:
            log_vehicle_data()
            log_timer = time.time()

        clock.tick(30)  # Limit to 60 FPS

except KeyboardInterrupt:
    print("\nStopping script. Destroying vehicles...")

finally:
    # Clean up actors
    camera.stop()
    vehicle_1.destroy()
    vehicle_2.destroy()
    if recording:
        video_writer.release()

    # Save log data to JSON file
    log_filename = f"carla_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w") as json_file:
        json.dump(log_data, json_file, indent=4)
    pygame.quit()
    print("Vehicles and camera removed. Exiting.")