"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")
args.use_gpu_pipeline = True

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 100.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../resources/robots/"
asset_file = "biped/urdf/biped.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.replace_cylinder_with_capsule = True
asset_options.fix_base_link = False
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 1.73, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment
env = gym.create_env(sim, env_lower, env_upper, 2)
robot = gym.create_actor(env, robot_asset, initial_pose, 'robot', 0, 1)
# Configure DOF properties
# props = gym.get_actor_dof_properties(env, robot)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
# props["stiffness"] = (5000.0, 5000.0)
# props["damping"] = (100.0, 100.0)
# gym.set_actor_dof_properties(env0, cartpole0, props)
# # Set DOF drive targets
# cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')
# pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole')
# gym.set_dof_target_position(env0, cart_dof_handle0, 0)
# gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)


# # Look at the first env
# cam_pos = gymapi.Vec3(8, 4, 1.5)
# cam_target = gymapi.Vec3(0, 2, 1.5)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
#
# feet_names = ["left_foot", "right_foot"]
# feet_indices = torch.zeros(len(feet_names), dtype=torch.long, requires_grad=False)
# # sensor_pose = gymapi.Transform()
# for i in range(len(feet_names)):
#     # sensor_options = gymapi.ForceSensorProperties()
#     # sensor_options.enable_forward_dynamics_forces = False  # for example gravity
#     # sensor_options.enable_constraint_solver_forces = True  # for example contacts
#     # sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
#     index = gym.find_asset_rigid_body_index(robot_asset, feet_names[i])
#     feet_indices[i] = gym.find_actor_rigid_body_handle(env, robot, feet_names[i])
#     # gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
#
# net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
# gym.refresh_net_contact_force_tensor(sim)
# contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(1, -1, 3)


# Simulate
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    # gym.simulate(sim)
    gym.fetch_results(sim, True)

    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # gym.refresh_force_sensor_tensor(sim)
    # print(contact_forces[:, feet_indices, 1].sum())

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
