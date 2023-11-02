from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BipedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8192
        n_scan = 187
        n_priv = 3 + 3 + 3
        n_priv_latent = 1 + 3 + 1 + 6 + 6  # [rand_mass, rand_com, friction_coeffs, motor_strength_kps, motor_strength_kds]
        n_proprio = 3 + 3 + 3 + 6 + 6 + 6 + 2
        history_len = 10

        num_observations = n_proprio + n_scan + n_priv + n_priv_latent + history_len * n_proprio
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 6

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.10, 0, -0.10]  # front camera
        angle = [40, 40]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

    class terrain(LeggedRobotCfg.terrain):
        simplify_grid = False
        horizontal_scale_if_use_camera = 0.1

    #     measure_heights = True
    #     measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05,
    #                          1.2]  # 1mx1.6m rectangle (without center line)
    #     measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]

    #     mesh_type = 'plane'
    #     measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.65]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'abad_L_Joint': 0.0,
            'hip_L_Joint': 0.0,
            'knee_L_Joint': 0.0,

            'abad_R_Joint': 0.0,
            'hip_R_Joint': 0.0,
            'knee_R_Joint': 0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'Joint': 20.}  # [N*m/rad]
        damping = {'Joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/biped/urdf/biped.urdf'
        name = "biped"
        foot_name = 'foot'
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['base_Link']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.6

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_ang_vel = 1.0
            orientation = -0.05
            torques = -5.e-6
            lin_vel_z = -0.5
            feet_air_time = 5.0
            dof_pos_limits = -1.
            no_fly = 0.25
            feet_edge = -1.0

    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-0.6, 0.6]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0, 0]


class BipedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'biped'
        max_iterations = 3000  # number of policy updates

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class depth_encoder:
        if_depth = BipedCfg.depth.use_camera
        depth_shape = BipedCfg.depth.resized
        buffer_len = BipedCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = BipedCfg.depth.update_interval * 24
