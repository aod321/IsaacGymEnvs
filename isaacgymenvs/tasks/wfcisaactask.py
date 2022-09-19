from enum import Enum
from typing import List
import numpy as np
from .base.vec_task import VecTask
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
from .wfc_env_inz_vec_task_only_empty import WFCEnv
import torch
import numpy as np
import sys
import cv2
import torchvision
import gym.spaces as spaces
from utils.torch3dutils import quaternion_multiply, quaternion_to_matrix, axis_angle_to_quaternion


# State Machine
class State(Enum):
    FIRST = 0
    NEWMAP = 1
    RESET = 2
    WAITACTIONS = 3
    UPDATEPHYCIS = 4
    APPLYTENSOR = 5
    REFRESHTENSOR = 6
    COMPUTEREWARD = 7
    IDLE = 8


class WFCIsaacTask(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.enable_viewer_sync = self.cfg["env"]["enable_viewer_sync"]
        self.max_episode_length = self.cfg["env"]["max_episode_length"]
        self.num_environments = self.cfg["env"]["numEnvs"]
        self.randomize = self.cfg["env"]["randomize"]
        self.env_asset_root = self.cfg["env"]["asset_root"]
        self.width = self.cfg["task"]["agent"]["camera"]["width"]
        self.height = self.cfg["task"]["agent"]["camera"]["height"]
        self.channel = self.cfg["task"]["agent"]["camera"]["channel"]
        self.enable_tensors = self.cfg["enableCameraSensors"]
        # camera sensor properties
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = self.width
        self.camera_properties.height = self.height
        self.camera_properties.enable_tensors = self.enable_tensors
        # self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_distance = self.cfg["env"]["plane"]["distance"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.dt = self.cfg["sim"]["dt"] 
        self.substeps = self.cfg["sim"]["substeps"]
        self.proximity_threshold = self.cfg["env"]["proximity_threshold"]
        self.num_actions = self.cfg["env"]["numActions"]
        self.graphics_device_id = graphics_device_id
        self.wfc_envs = []
        self.envs = []
        self.root_tensor = None
        self.inital_tensor = None
        self.temp_tensor = None
        self.camera_tensors = []
        self.food_indexes = []
        self.agent_indexes = []
        self.interaction = 0
        self._root_tensor = None
        self.rotation_angle_tensor = None
        self.neg_rotation_angle_tensor = None
        self.rotation_angle = 0.0625 * np.pi
        self.all_blocks_index_list = []
        self.not_connect_maps = []
        self.all_index_to_env_id = []
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=self.graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        # self.reset_flags = torch.zeros(self.num_environments).to(self.device)
        self.control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
        self.act_space = spaces.Discrete(self.num_actions)
        self.obs_space = spaces.Box(low=0, high=255, shape=(self.channel, self.width, self.height), dtype=np.uint8)
        self.fast_move_speed = 5
        self.slow_move_speed = 2
        self.current_state = State.FIRST
        self.render_env_ids = None
        self.reset_env_ids = None
        self.last_frame_cnt = 0
        self.step_count = 0
        self.start_time = 0

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        # Ground plane distance from origin
        plane_params.distance = self.plane_distance
        # Coefficient of static friction
        plane_params.static_friction = self.plane_static_friction
        # Coefficient of dynamic friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        # The ratio of the final to initial velocity after the rigid body collides
        # 0 is perfectly inelastic collision(完全非弹性碰撞), 1 is elastic collision(弹性碰撞)
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = self.env_asset_root
        # Load all assets
        # create procedural asset
        tiles_options = gymapi.AssetOptions()
        tiles_options.density = 0.02
        tiles_options.fix_base_link = True
        tiles_options.linear_damping = 0.1
        # Preload all assets from urdf model
        color_list = ["gray", "blue", "orange", "red", "white", "yellow"]
        all_pt = {}
        # for example: pt_cubes = ["PCG/gray_cube.urdf","PCG/blue_cube.urdf","PCG/orange_cube.urdf", "PCG/red_cube.urdf", "PCG/white_cube.urdf", "PCG/yellow_cube.urdf"]
        for name in ["cube", "ramp", "corner"]:
            all_pt[name] = [f"PCG/{c}_{name}.urdf" for c in color_list]
        all_assets = {}
        for name in ["cube", "ramp", "corner"]:
            all_assets[name] = [self.gym.load_asset(self.sim, asset_root, pt, tiles_options) for pt in all_pt[name]]
        
        avatar_options = gymapi.AssetOptions()
        avatar_options.density = 1.0
        avatar_options.fix_base_link = False
        avatar_options.linear_damping = 0.1
        capsule_asset = self.gym.create_capsule(self.sim, 1, 1, avatar_options)
        assets_dict = {
            "cube_assets": all_assets["cube"],
            "ramp_assets": all_assets["ramp"],
            "corner_assets": all_assets["corner"],
            "capsule_asset": capsule_asset
        }
        self.not_connect_maps = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            wfcenv = WFCEnv(rank=i, cfg=self.cfg, assets_dict=assets_dict, camera_properties=self.camera_properties, isaac_env=env_ptr, isaac_sim=self.sim, device=self.device)
            # empty aera
            wfcenv.render_in_sim()
            self.wfc_envs.append(wfcenv)
            self.envs.append(env_ptr)
            self.not_connect_maps.append(wfcenv.not_connect_map)
        self.not_connect_maps = torch.from_numpy(np.array(self.not_connect_maps)).to(self.device) # (N, 9, 9)
        # wrappy env list to numpy array
        self.wfc_envs = np.array(self.wfc_envs)
        for env_id, current_env in enumerate(self.envs):
            wfcenv = self.wfc_envs[env_id]
            agent_idx =self.gym.get_actor_index(current_env, wfcenv.agent_handler, gymapi.DOMAIN_SIM)
            food_idx = self.gym.get_actor_index(current_env, wfcenv.food_handler, gymapi.DOMAIN_SIM)
            block_idx_list = []
            # for block_handler in wfcenv.all_block_handlers:
                # block_idx_list.append(self.gym.get_actor_index(current_env, block_handler, gymapi.DOMAIN_SIM))
            self.all_blocks_index_list.append(block_idx_list)
            self.agent_indexes.append(agent_idx)
            self.food_indexes.append(food_idx)
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, current_env, wfcenv.camera_handler, gymapi.IMAGE_COLOR)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            self.camera_tensors.append(torch_camera_tensor)
        
        self.agent_indexes = torch.tensor(self.agent_indexes, device=self.device)
        self.food_indexes = torch.tensor(self.food_indexes, device=self.device)
        self.gym.prepare_sim(self.sim)
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
        self.rotation_angle_tensor = axis_angle_to_quaternion(torch.tensor([self.rotation_angle, 0.0, 0.0], device=self.device))
        self.neg_rotation_angle_tensor = axis_angle_to_quaternion(torch.tensor([-self.rotation_angle, 0.0, 0.0], device=self.device))
    
    def resetActorsTensor(self, env_id):
        all_block_index = np.array(self.all_blocks_index_list)[env_id].reshape(-1).tolist()
        self.temp_tensor[all_block_index, :] = 0.0
        self.temp_tensor[all_block_index, 2] = -100.0
    
    def render_in_sim(self, env_ids):
        self.render_env_ids = env_ids
        self.current_state = State.NEWMAP

    def mutate_a_new_map(self, env_ids):
        for wfc_env in self.wfc_envs[env_ids]:
            wfc_env.mutate_a_new_map()

    # private method, only called by step statemachine
    def __vec_render_in_sim_tensor(self, env_id):
        self.resetActorsTensor(env_id)
        self.not_connect_maps = []
        for wfc_env in self.wfc_envs[env_id]:
            wfc_env.render_in_sim_tensor(root_tensor=self.temp_tensor)
            self.not_connect_maps.append(wfc_env.not_connect_map)
        self.not_connect_maps = torch.from_numpy(np.array(self.not_connect_maps)).to(self.device) # (N, 9, 9)

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # # step graphics
            # self.gym.step_graphics(self.sim)
            # self.gym.render_all_camera_sensors(self.sim)
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

            if mode == "rgb_array":
                # only show one world
                # self.gym.start_access_image_tensors(self.sim)
                
                # camera_tensor_image = self.camera_tensors[0].detach().cpu().numpy().astype(np.uint8)
                camera_tensor_image = torchvision.utils.make_grid(torch.rot90(torch.stack(self.camera_tensors, dim=0).permute(0, 3, 1, 2), 3, dims=[2,3])[:, :3].cpu())
                # self.gym.end_access_image_tensors(self.sim)
                # rotate 90 degrees counter-clockwise
                # image = np.rot90(camera_tensor_image, -1)
                # camera_tensor_image = Image.fromarray(camera_tensor_image)
                image = camera_tensor_image.permute(1, 2, 0).numpy()
                cv2.imshow("view", cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
                img = self.not_connect_maps[0].long().cpu().numpy().astype(np.uint8) * 255
                x = self.root_tensor[self.agent_indexes][0][0].item()
                y = self.root_tensor[self.agent_indexes][0][1].item()
                x = ((x+1)/18)*180
                y = ((y+1)/18)*180
                cv2.circle(img, (int(y),int(x)), 5, (255,0,0), 0)
                cv2.imshow("connect", img)
                self.interaction = cv2.waitKey(1)
                return self.interaction

    def compute_reward(self):
        if self.current_state != State.WAITACTIONS:
            return
        root_positions = self.root_tensor[:, 0:2]
        agent_position = torch.index_select(root_positions, 0, self.agent_indexes)
        food_position = torch.index_select(root_positions, 0, self.food_indexes)
        assert agent_position.shape == food_position.shape, f"agent_posistion:{agent_position.shape}, food_position:{food_position.shape}"
        rewards_buf = torch.zeros(self.num_envs, device=self.device)
        dist = torch.linalg.norm(agent_position - food_position, dim=1)
        self.rew_buf[:] = torch.where(dist < self.proximity_threshold, torch.ones_like(rewards_buf), rewards_buf)
        # done
        # 1. get reward
        self.reset_buf[:] = torch.where(self.rew_buf == 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # 2. out area
        self.reset_buf[:] = torch.where(((agent_position<0) + (agent_position - 18 >=0)).sum(dim=1)>0, torch.ones_like(self.reset_buf), self.reset_buf)
        # # 3. in not connected area
        if self.not_connect_maps.nelement() != 0:
            agent_pos_x = torch.clamp(((((agent_position[:, 1] + 1) / (9 * 2)) * 180).unsqueeze(1)).floor().long(), min=0, max=179)
            agent_pos_y = torch.clamp(((((agent_position[:, 0] + 1) / (9 * 2)) * 180).unsqueeze(1)).floor().long(), min=0, max=179)
            agent_pos_x = agent_pos_x.repeat(1, self.not_connect_maps.shape[1])
            slice_x = self.not_connect_maps.gather(dim=2, index=agent_pos_x.unsqueeze(2)).squeeze(2)
            final_slice = slice_x.gather(1, agent_pos_y).squeeze(1)
            # print(f"x:{agent_pos_x}, y:{agent_pos_y}, now:{final_slice}")
            self.reset_buf[:] = torch.where(final_slice == 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # 4. time out
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        # satrt op
        # n,w,h,c -> n,c,w,h
        self.obs_buf = torch.stack(self.camera_tensors, dim=0).permute(0,3,1,2) / 255.0 # Shape(num_envs, num_cameras_per_env)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, 4, 84, 84), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict
        
    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self.current_state = State.RESET
            # self.reset_flags[env_ids] = 1
            for env in env_ids:
                self.wfc_envs[env].placeAgentAndFood_tensor(self.temp_tensor)
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0

    def apply_actions(self, in_tensor, in_actions, agent_indexes):
        if len(agent_indexes) > 0:
            actions = in_actions.clone()
            agent_tensor = in_tensor[agent_indexes] # Shape(N, 13)
            temp_tensor = agent_tensor.repeat(7,1,1) # Shape(7, N, 13)
            matrix = quaternion_to_matrix(agent_tensor[:, 3:7])                         # Shape(N, 4)
            y_tensors =torch.matmul(matrix, torch.tensor([0.0, 1.0, 0.0], device=self.device)) # Shape (N, 3)
            unit_vector = torch.linalg.norm(y_tensors, dim=1).unsqueeze(1).repeat(1,3)
            pos_tensors = y_tensors / unit_vector
            # if action < 0 or action > 7, no action applied
            no_action = (actions < 0) +  (actions > 7)
            no_action_idxes = torch.nonzero(no_action).view(-1)
            if len(no_action_idxes) > 0:
                actions[no_action] = 0
            speed = 10
            # action ==0, up
            # temp_tensor[0, :, 7:9] = -pos_tensors[:, 0:2]
            temp_tensor[0, :, 0:2] = -pos_tensors[:, 0:2] * speed * self.dt
            # action ==1, down
            # temp_tensor[1, :, 7:9] = pos_tensors[:, 0:2]
            temp_tensor[1, :, 0:2] = pos_tensors[:, 0:2] * speed * self.dt
            # action ==2, left
            # temp_tensor[2, :, 7:9] = torch.stack([pos_tensors[:, 1], -pos_tensors[:, 0]], dim=1)
            temp_tensor[2, :, 0:2] = torch.stack([pos_tensors[:, 1], -pos_tensors[:, 0]], dim=1)  * speed * self.dt
            # action ==3, right
            # temp_tensor[3, :, 7:9] = torch.stack([-pos_tensors[:, 1], pos_tensors[:, 0]], dim=1)
            temp_tensor[3, :, 0:2] = torch.stack([-pos_tensors[:, 1], pos_tensors[:, 0]], dim=1)  * speed * self.dt
            # action ==4, turn left
            temp_tensor[4, :, 3:7] = quaternion_multiply(agent_tensor[:, 3:7], self.rotation_angle_tensor)
            # action ==5, turn right
            temp_tensor[5, :, 3:7] = quaternion_multiply(agent_tensor[:, 3:7], self.neg_rotation_angle_tensor)
            # action ==6, slow up
            temp_tensor[6, :, 7:9] = (-self.slow_move_speed * y_tensors)[:, 0:2]

            action_tensor = temp_tensor[actions.type(torch.long), torch.arange(len(agent_indexes))] # Shape (N, 13)

            # keep no action
            if len(no_action_idxes) > 0:
                action_tensor[no_action_idxes] = agent_tensor[no_action_idxes]

            assert action_tensor.shape == torch.Size((agent_tensor.shape[0], 13)), action_tensor.shape
            in_tensor[agent_indexes] = action_tensor
        return in_tensor

    def pre_physics_step(self, actions):
        if self.current_state == State.FIRST:
            # skip first to wait temp_tensor to be initialized after the first time simulation
            return

        if self.current_state == State.APPLYTENSOR:
            # 将temp_tensor中的数值写入物理引擎
            self.root_tensor[:] = self.temp_tensor[:]
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
            self.current_state = State.REFRESHTENSOR

        elif self.current_state == State.NEWMAP:
            self.__vec_render_in_sim_tensor(self.render_env_ids)
            self.current_state = State.WAITACTIONS

        elif self.current_state == State.RESET:
            all_reset_ids = torch.cat([self.agent_indexes[self.reset_env_ids],
                                       self.food_indexes[self.reset_env_ids]],
                                      dim=0)
            self.root_tensor[all_reset_ids, :] = self.temp_tensor[all_reset_ids, :]
            self.current_state = State.WAITACTIONS

        if self.current_state == State.WAITACTIONS:
            self.actions = actions.clone().to(self.device)
            # 关于agent部分的数值根据动作计算, 该计算结果替换物理引擎相应的结果
            self.temp_tensor = self.apply_actions(in_tensor=self.temp_tensor, in_actions=self.actions, agent_indexes=self.agent_indexes)
            self.current_state = State.APPLYTENSOR

    def post_physics_step(self):
        
        if self.current_state == State.FIRST:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            # init temp_tensor
            self.temp_tensor = self.root_tensor.clone()
            # 设定初始线速度为0,防止初始重叠造成的飞跳
            self.temp_tensor[:, 7:10] = 0
            self.current_state = State.APPLYTENSOR

        elif self.current_state == State.REFRESHTENSOR:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.current_state = State.UPDATEPHYCIS
        
        if self.current_state == State.UPDATEPHYCIS:
            # 从物理引擎中读取数值更新到temp_tensor中,仅取agent和food的位置及线速度部分,以恢复基本的摩擦,重力等效果的同时防止旋转倾倒等其他问题
            self.temp_tensor[self.agent_indexes, 0:3] = self.root_tensor[self.agent_indexes, 0:3]
            self.temp_tensor[self.agent_indexes, 7:10] = self.root_tensor[self.agent_indexes, 7:10]
            self.current_state = State.WAITACTIONS

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_env_ids = env_ids
            self.reset_idx(env_ids)

        self.progress_buf += 1
        self.gym.start_access_image_tensors(self.sim)
        self.compute_observations()
        self.compute_reward()
        self.gym.end_access_image_tensors(self.sim)
        # 统计Simulation的FPS
        # end_time = time.time()
        # self.step_count += self.num_envs
        # if (end_time - self.start_time) > 1:
        #     print("FPS: %.2f" % ((self.step_count - self.last_frame_cnt)))
        #     self.last_frame_cnt = self.step_count
        #     self.start_time = time.time()
