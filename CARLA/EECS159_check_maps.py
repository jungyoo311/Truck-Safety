import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

available_maps = client.get_available_maps()
print("available maps:", available_maps)