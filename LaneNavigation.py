# Lane Navigation for Self Driving Car - CARLA 0.9.13

import glob
import os
import sys
import random
import time
import numpy as np 
import cv2
import math
import argparse
import matplotlib.pyplot as plt
from collections import deque
import queue

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Camera (data) dimensions:
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Getting data from camera sensor:
def process_img(image):
    i = np.array(image.raw_data)
    # Image: H, W, RGBA:
    i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    # Only taking the rgb values:
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3 / 255.0

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

class VehiclePIDController():
    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.vehicle = vehicle

        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)

    # Update steering, throttle, brake:
    def run_step(self, target_speed, waypoint):
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()

        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_brake)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        if current_steering >= self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1

        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        return control

# Longitudinal control:
class PIDLongitudinalControl():
    def __init__(self, vehicle, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.03):
        self.vehicle = vehicle
        self.Kd = Kd
        self.Ki = Ki
        self.Kp = Kp
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen = 10)

    def pid_controller(self, target_speed, current_speed):
        # Calculating errors:
        error = target_speed - current_speed
        self.errorBuffer.append(error)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
        return np.clip(self.Kp * error + self.Kd * de + self.Ki * ie, -1.0, 1.0)

    # Perform one step and updating speed:
    def run_step(self, target_speed):
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)

# Lateral control:
class PIDLateralControl():
    def __init__(self, vehicle, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.03):
        self.vehicle = vehicle
        self.Kd = Kd
        self.Ki = Ki
        self.Kp = Kp
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen = 10)

    # Getting velocities 
    def pid_controller(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)), y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        dot = math.acos(np.clip(np.dot(w_vec, v_vec) / np.linalg.norm(w_vec) * np.linalg.norm(v_vec), -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)

        if cross[2] < 0:
            dot *= -1

        self.errorBuffer.append(dot)

        # Calculating errors:
        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt       
        else:
            de = 0.0
            ie = 0.0
        return np.clip((self.Kp * dot) + (self.Ki * ie) + (self.Kd * de), -1.0, 1.0)

    # Perform one step:
    def run_step(self, waypoint):        
        return self.pid_controller(waypoint, self.vehicle.get_transform())


def main():
    actorList = []

    try:
        # Setting the environment:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town02')
        carlaMap = world.get_map()

        # Creating vehicle (Tesla Cybertruck)
        blueprintLibrary = world.get_blueprint_library()
        vehicle_bp = blueprintLibrary.filter('cybertruck')[0]

        # spawn_point = random.choice(carlaMap.get_spawn_points())
        spawn_point = carla.Transform(carla.Location(x=140.0, y=195.0, z=30.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actorList.append(vehicle)
        print('Created %s' % vehicle.type_id)

        # Creating camera sensor and attaching to the car:
        cam_bp = blueprintLibrary.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
        cam_bp.set_attribute("fov", "110")
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
        actorList.append(sensor)
        sensor.listen(lambda data: process_img(data))

        # Applying PID controller:
        control_vehicle = VehiclePIDController(vehicle, args_lateral={'Kp': 1.0, 'Kd':0.0, 'Ki': 0.0}, args_longitudinal={'Kp': 1.0, 'Kd':0.0, 'Ki': 0.0})

        # Keep running..
        while True:
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            waypoint = np.random.choice(waypoints.next(0.3))
            control_signal = control_vehicle.run_step(5, waypoint)
            vehicle.apply_control(control_signal)

    finally:
        print('Deleted actor list!')
        client.apply_batch([carla.command.DestroyActor(x) for x in actorList])   


if __name__ == '__main__':
    main()