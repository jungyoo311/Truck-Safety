import carla
import time
import numpy as np
import math
import csv

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get all vehicles
vehicle_list = world.get_actors().filter('vehicle.*')

# Destroy all vehicles
for vehicle in vehicle_list:
    vehicle.destroy()

print(f"Destroyed {len(vehicle_list)} vehicles.")

# Enable synchronous mode for stability
settings = world.get_settings()
settings.synchronous_mode = True  # Enables sync mode
settings.fixed_delta_seconds = 0.05  # Ensures consistent updates
world.apply_settings(settings)

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Define vehicle model
vehicle_model = 'vehicle.carlamotors.firetruck'
vehicle_bp = blueprint_library.find(vehicle_model)

# Spawn location
spawn_point = carla.Transform(carla.Location(x=-9.91, y=-217, z=0.2), carla.Rotation(yaw=90))
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if not vehicle:
    print("Error: Vehicle not spawned")
    exit()

print(f"Spawned {vehicle_model} at {spawn_point.location}")

# Function to get vehicle speed in m/s
def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

# Function to control vehicle
def control_vehicle(vehicle, throttle=1.0, steer=0.0, brake=0.0):
    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer
    control.brake = brake
    vehicle.apply_control(control)

# Define test parameters
initial_speeds = [10, 20, 30]  # Initial speeds in m/s (convert to ~MPH by multiplying 2.23694)
vehicle_masses = [10000, 20000, 30000]  # Vehicle weights in kg
brake_force_values = [1]  # Braking force in Newtons

# Data storage
for mass in vehicle_masses:
    for initial_speed in initial_speeds:
        for brake_force in brake_force_values:
            # Respawn vehicle for each test case
            vehicle.destroy()
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if not vehicle:
                print("Error: Vehicle not spawned")
                continue
            # Change vehicle mass dynamically
            physics_control = vehicle.get_physics_control()
            physics_control.mass = mass
            vehicle.apply_physics_control(physics_control)
            print(f"Testing: Mass={mass} kg, Speed={initial_speed} m/s, Brake Force={brake_force}")
            
            # Set vehicle to the desired speed
            yaw_rad = math.radians(spawn_point.rotation.yaw)
            velocity = carla.Vector3D(initial_speed * math.cos(yaw_rad), initial_speed * math.sin(yaw_rad), 0)
            vehicle.set_target_velocity(velocity)
            world.tick()
            time.sleep(0.5)
            
            # Start braking and measure stopping time and distance
            start_time = time.time()
            start_location = vehicle.get_transform().location
            speed_data = []
            
            while get_vehicle_speed(vehicle) > 5:
                speed = get_vehicle_speed(vehicle)
                elapsed_time = time.time() - start_time
                distance_traveled = math.sqrt((vehicle.get_transform().location.x - start_location.x)**2 + (vehicle.get_transform().location.y - start_location.y)**2)
                speed_data.append([elapsed_time, speed, distance_traveled])
                for i in range(3):
                    control_vehicle(vehicle, throttle=0.0, brake=brake_force)
                world.tick()
                time.sleep(0.05)
            
            # Save data to CSV file
            filename = f"braking_test_mass{mass}_speed{initial_speed}_brake{brake_force}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Speed (m/s)", "Distance (m)"])
                writer.writerows(speed_data)
            
            print(f"Data saved to {filename}")

# Cleanup
vehicle.destroy()
world.apply_settings(settings)
