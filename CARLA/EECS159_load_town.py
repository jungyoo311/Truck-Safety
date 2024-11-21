import carla

def load_town_and_set_camera(town_name):
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    try:
        # Load the specified town
        client.load_world(town_name)
        print(f"Successfully loaded {town_name}")

        # Get the world and the spectator camera
        world = client.get_world()
        spectator = world.get_spectator()

        # Set a new location and rotation for the camera
        # Adjust these values based on your town to get a better view
        # transform = carla.Transform(
        #     carla.Location(x=-1000, y=3000, z=350),  # Position above the ground
        #     carla.Rotation(pitch=-90, yaw=0, roll=0)  # Looking straight down
        # )
        # spectator.set_transform(transform)
        # print("Camera position set.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Specify the town name you want to load
    town_name = 'Town04'  # Example: Town01, Town02, Town03, etc.
    load_town_and_set_camera(town_name)