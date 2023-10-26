from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BipedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 217
        num_actions = 6

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
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.6

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0
            tracking_ang_vel = 1.0
            torques = -5.e-6
            lin_vel_z = -0.5
            ang_vel_xy = -0.0
            feet_air_time = 5.0
            dof_pos_limits = -1.
            no_fly = 0.25


class BipedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'biped'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
