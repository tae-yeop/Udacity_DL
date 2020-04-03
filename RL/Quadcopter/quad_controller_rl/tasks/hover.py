""" Hover task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """lift off the ground to a target height and move around (x,y) space while maintaining the target height"""
    
    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        print("Hover(): action_space = {}".format(self.action_space))

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.max_error_position = 8.0  # distance units
        self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at
        self.weight_position = 0.55 # weight for position error
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.15 # weight for orientation error
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.30 # weight for velocity error

    def reset(self):
        # Reset episodic-specific variables
        self.last_timestamp = None
        self.last_position = None

        # Return initial condition
        p = self.target_position + np.random.normal(0.5, 0.1, size=3)  # slight random position around the target
        return Pose(
                position=Point(*p),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )
    
    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector 
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero

        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Track the deviations for position, orientation and veloctiy.
        error_position = np.linalg.norm(self.target_position - state[0:3]) #Euclidian distance from target position
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7]) #Euclidian distance from target orientation
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10]) #Euclidian distance from target velocity

        # Compute reward / penalty and check if this episode is complete
        done = False
        # Basic penalty
        reward = -(self.weight_position * error_position + self.weight_orientation * error_orientation + self.weight_velocity * error_velocity)
        # Extra penalty for z-axis
        z_axis_penalty = (self.target_position[2] - pose.position.z)**2 + abs(self.target_velocity[2]-state[9])
        reward -= z_axis_penalty
        if error_position > self.max_error_position:
            reward -= 50 * np.sqrt(np.exp(abs(self.max_duration-timestamp)))  # extra penalty, agent strayed too far
            done = True
        elif timestamp > self.max_duration:
            reward += 100.0  # extra reward, agent made it to the end
            done = True
        else:
            # extra reward, agent hovers around target position
            # weighted by difference between max_error_position and error_position
            reward += (30.0 * np.log(self.max_error_position - error_position)) 
        
        action = self.agent.step(state, reward, done)

        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
