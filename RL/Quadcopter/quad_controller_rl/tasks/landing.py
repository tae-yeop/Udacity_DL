""" Landing task."""
import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
class Landing(BaseTask):
    """lift off the ground to a certain height and land down to the target height nearby ground"""
    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0 # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
        np.array([- cube_size / 2, - cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
        np.array([ cube_size / 2, cube_size / 2, cube_size, 1.0, 1.0, 1.0, 1.0]))
        print("Hover(): observation_space = {}".format(self.observation_space)) # [debug]
        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
        np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
        np.array([ max_force, max_force, max_force, max_torque, max_torque, max_torque]))
        print("Hover(): action_space = {}".format(self.action_space))

        # Task-specific parameters
        self.max_duration = 5.0 # secs
        self.target_z = 0.1 # target height (z position)
        self.target_xy = np.array([0.0, 0.0]) # target (x,y) position

        self.start_position = 13.0 # start height 
        self.target_position = np.array([0.0, 0.0, 0.0]) # target (x,y,z) position

        self.target_accel = np.array([0.0, 0.0, 0.0]) # target acceleration
        self.accel_error_weight = 15
        self.stray_error_weight = 22
        self.velocity_z_weight = 2
        
    def reset(self):
        # Reset episodic-specific variables
        self.last_timestamp = None
        self.last_position = None
        # Return initial condition
        return Pose(
            position=Point(0.0, 0.0, self.start_position + np.random.normal(0.5, 0.1)), # slight random position around the target
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
            )
    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose, linear_acceleration only; ignore angular_velocity)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        acceleration = np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03) # prevent divide by zero
        
        state = np.concatenate([position, orientation, velocity, acceleration]) # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False
        # Basic penalty
        error_position = np.linalg.norm(self.target_position - state[0:3])
        reward = -error_position
        # Extra reward/penalty on z-axis position and velocity.
        error_position_z = np.clip(pose.position.z, 0, 20)**2
        error_velocity_z = np.clip(state[9],-20, 20)**2
        reward = 20 -np.sqrt(error_position_z + self.velocity_z_weight*error_velocity_z)
    
        # Track the (x,y) space deviation
        error_position_xy = self.stray_error_weight * np.linalg.norm(self.target_xy - state[0:2]) #Euclidian distance from target position
        # Extra penalty for straying on (x,y) space
        reward -= error_position_xy

        # Track the linear acceleartion error and extra penalty
        error_acceleration = self.accel_error_weight * np.linalg.norm(self.target_accel - state[10:13])
        reward -= error_acceleration
        
        # If Agent make it to target_z
        if pose.position.z <= self.target_z:
            reward += 50.0
            done = True
        elif timestamp > self.max_duration: # agent has run out of time
            reward -= 35.0 # penalty
            done = True
        # Reward clipping to avoid istability
        reward = np.clip(reward, -1.0, 1.0)

        action = self.agent.step(state, reward, done)
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high) # flatten, clamp to action space limits
            return Wrench(
            force=Vector3(action[0], action[1], action[2]),
            torque=Vector3(action[3], action[4], action[5])
            ), done
        else:
            return Wrench(), done

