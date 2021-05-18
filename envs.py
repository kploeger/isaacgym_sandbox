import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil


class IsaacEnv():
    pass


class IsaacJugglingWam4(IsaacEnv):

    n_dof = 4
    p_gains  = np.array([200.0, 300.0, 100.0, 100.0])
    d_gains  = np.array([  7.0,  15.0,   5.0,   2.5])
    max_ctrl = np.array([150.0, 125.0,  40.0,  60.0])
    min_ctrl = -max_ctrl
    dt = 0.002

    q_home = np.array([0., -1.986, 0., 3.146])


    def __init__(self, gym, env, robots, objects):
        self.gym = gym
        self.env = env
        self.robot = robots[0]
        self.balls = objects

        # Configure DOF properties
        props = self.gym.get_actor_dof_properties(self.env, self.robot)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        self.gym.set_actor_dof_properties(self.env, self.robot, props)
        # self.gym.set_rigid_body_color(self.env, balls[0], )

        for ball in self.balls:
            shape_props = gym.get_actor_rigid_shape_properties(self.env, ball)
            for shape_prop in shape_props:
                shape_prop.restitution = 0
                shape_prop.friction = 0.1
                shape_prop.rolling_friction = 0
                shape_prop.torsion_friction = 0
            gym.set_actor_rigid_shape_properties(self.env, ball, shape_props)

        shape_props = gym.get_actor_rigid_shape_properties(self.env, self.robot)
        for shape_prop in shape_props:
            shape_prop.restitution = 0
            shape_prop.friction = 0.1
            shape_prop.rolling_friction = 0
            shape_prop.torsion_friction = 0
        gym.set_actor_rigid_shape_properties(self.env, self.robot, shape_props)



    def apply_action(self, q=None, dq=None, tau=None):
        # state = self.gym.get_actor_dof_states(self.env, self.robot, gymapi.STATE_NONE)
        state = self.gym.get_actor_dof_states(self.env, self.robot, gymapi.STATE_ALL)
        self.pos = state['pos']
        self.vel = state['vel']
        motor_torques = np.zeros(self.n_dof)
        # ...pos if available and 0 vel if no vel availavle
        if not q is None:
            motor_torques += self.p_gains * (q.ravel() - state['pos'])
        if not dq is None:
            motor_torques += self.d_gains * (dq.ravel() - state['vel'])
        if not tau is None:
            motor_torques += tau.ravel()
        # if self.g_comp_ctl is not None:
            # tau_g_comp = self.g_comp_ctl()
            # motor_torques += tau_g_comp

        self.motor_torques = motor_torques
        motor_torques = np.clip(motor_torques, self.min_ctrl, self.max_ctrl, dtype=np.float32)
        self.gym.apply_actor_dof_efforts(self.env, self.robot, motor_torques)

    def reset(self, q=None, dq=None, ball1_pos=None, ball2_pos=None, ball1_vel=None, ball2_vel=None):
        if q is None:
            q = self.q_home
        if dq is None:
            dq = np.zeros_like(self.q_home)
        state = self.gym.get_actor_dof_states(self.env, self.robot, gymapi.STATE_ALL)
        state['pos'] = q
        state['vel'] = dq
        self.gym.set_actor_dof_states(self.env, self.robot, state, gymapi.STATE_ALL)

        # ball1_handle = self.gym.get_rigid_handle(self.env, "Ball1", "base_link")
        # ball2_handle = self.gym.get_rigid_handle(self.env, "Ball2")


        # ball1_pose = gymapi.Transform()
        # ball2_pose = gymapi.Transform()
        # # ball1_pose.p = gymapi.Vec3(0.87, 0.086, 1.07)
        # ball1_pose.p = gymapi.Vec3(0.88, 0.086, 1.1)
        # ball2_pose.p = gymapi.Vec3(0.87, -0.086, 2.0)

        # self.gym.set_rigid_transform(self.env, self.balls[0], ball1_pose)
        # self.gym.set_rigid_transform(self.env, self.balls[0], ball2_pose)
        # self.gym.set_rigid_linear_velocity(self.env, ball1_handle, gymapi.Vec3(0., 0., 0.))
        # self.gym.set_rigid_linear_velocity(self.env, ball2_handle, gymapi.Vec3(0., 0., 0.))

    def get_ee_trafo(self):
        ee_handle = self.gym.get_rigid_handle(self.env, "Robot", "tool_col")
        trafo = self.gym.get_rigid_transform(self.env, ee_handle)
        return trafo

    def get_ball_positions(self):
        # ball1_handle = self.gym.get_rigid_handle(self.env, "Ball1", "base_link")
        # ball2_handle = self.gym.get_rigid_handle(self.env, "Ball2", "base_link")
        # ball1_trafo = self.gym.get_rigid_transform(self.env, ball1_handle)
        # ball2_trafo = self.gym.get_rigid_transform(self.env, ball2_handle)
        # return np.array([ball1_trafo.p, ball2_trafo.p])
        vec = gymapi.Vec3(0, 0, 3)
        return np.array([vec, vec])



class IsaacSim:
    def __init__(self, env_type, num_envs):
        # TODO: how does this work?
        args = gymutil.parse_arguments(
            description="Basketbot Simulaton",
            custom_parameters=[
                {"name": "--render", "action": "store_true", "help": "Render the simulation"},
                {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])
        self.render = args.render

        self.gym = gymapi.acquire_gym()

        # create a simulator
        sim_params = gymapi.SimParams()
        sim_params.substeps = 2
        sim_params.dt = env_type.dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        if args.physics_engine == gymapi.SIM_FLEX:
            raise NotImplementedError("Flex simulation engine is not supported! Choose PhysX.")
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.use_gpu = True
            # sim_params.physx.solver_type = 1

        # sim_params.physx.use_gpu = False

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id,
                                       args.physics_engine, sim_params)

        # Create the ground plane:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Load the robot:
        asset_root = "robot_description"
        asset_file = "urdf/robot/wam_4dof.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Load the environment assets:
        asset_file = "urdf/environment/ball.urdf"
        asset_options = gymapi.AssetOptions()
        # ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ball_asset = self.gym.create_sphere(self.sim, 0.0375, asset_options)

        sphere_asset = self.gym.create_sphere(self.sim, 0.0375, asset_options)

        # Env gird:
        envs_per_row = 3
        envs_per_row = int(np.sqrt(num_envs))
        # env_spacing = 1.2
        env_spacing = 0.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # Create and populate envs:
        self.envs = []
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            default_pose = gymapi.Transform()  # Default pose
            robot = self.gym.create_actor(env, robot_asset, default_pose, "Robot", i)
            ball1_pose = gymapi.Transform()
            ball1_pose.p = gymapi.Vec3(0.87, 0.086, 1.07)
            ball1 = self.gym.create_actor(env, ball_asset, ball1_pose, "Ball1", i)
            shape_props = self.gym.get_actor_rigid_shape_properties(env, ball1)
            print(shape_props)
            shape_props[0].restitution = 0
            self.gym.set_actor_rigid_shape_properties(env, ball1, shape_props)
            ball2_pose = gymapi.Transform()
            ball2_pose.p = gymapi.Vec3(0.87, -0.086, 2.0)
            ball2 = self.gym.create_actor(env, ball_asset, ball2_pose, "Ball2", i)
            shape_props = self.gym.get_actor_rigid_shape_properties(env, ball2)
            print(shape_props)
            shape_props[0].restitution = 0
            self.gym.set_actor_rigid_shape_properties(env, ball2, shape_props)
            # sphere_pose = gymapi.Transform()
            # sphere_pose.p = gymapi.Vec3(0, 0, 2)
            # sphere = self.gym.create_actor(env, sphere_asset, sphere_pose, "Sphere", i)
            self.envs.append(IsaacJugglingWam4(self.gym, env, [robot], [ball1, ball2]))


        # Create viewer (if rendered)
        self.viewer = None
        if self.render:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # Look at the first env
            # cam_pos = gymapi.Vec3(5, 20, 2)
            cam_pos = gymapi.Vec3(1.7, 1.7, 1.5)
            cam_target = gymapi.Vec3(0.8, 0, 1.3)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            if self.viewer is None:
                    self.render = False
                    print("*** Failed to create viewer -> rendering deactivated")

        self.time_step = 0


    def __del__(self):
        if self.render:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


    def reset(self, *args, **kwargs):
        self.time_step = 0
        for env in self.envs:
            env.reset(*args, **kwargs)


    def step(self): # step the physics

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        if self.render and self.time_step % 5 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        self.time_step += 1

        return self.time_step



def main():

    sim = IsaacSim(IsaacJugglingWam4, num_envs=24)
    for _ in range(10):
    # while True:
        sim.step()


if __name__ == '__main__':
    main()

