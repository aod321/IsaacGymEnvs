from isaacgym import gymapi
from utils.torch3dutils import *
from pcgworker.PCGWorker import *
import torch
from gym import spaces
import random
import einops
import numpy as np


class Block(object):
    def __init__(self, handler, used, type, tag):
        self.handler = handler
        self.used = used
        self.type = type
        self.tag = tag

    def set_handler(self, handler, type, tag):
        self.handler = handler
        self.tyep = type
        self.used = False
        self.tag = tag

    def is_used(self):
        self.used = True

    def __str__(self) -> str:
        return self.tag


class WFCEnv(object):
    def __init__(self, rank, cfg, isaac_env, isaac_sim, camera_properties, assets_dict, device=None):
        super(WFCEnv, self).__init__()
        self.device = device
        self.cfg = cfg["task"]
        self.group = 0
        self.wfc_size = self.cfg["wfc_size"]
        self.prefab_size = self.cfg["prefab_size"]
        self.height_scale = self.cfg["height_scale"]
        prefab_height = self.cfg["prefab_height"]
        self.num_actions = cfg["env"]["numActions"]
        self.width = self.cfg["agent"]["camera"]["width"]
        self.height = self.cfg["agent"]["camera"]["height"]
        self.channel = self.cfg["agent"]["camera"]["channel"]
        self.prefab_height = prefab_height * self.height_scale
        self.camera_properties = camera_properties
        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.channel),
                                            dtype=np.uint8)
        torch.manual_seed(3407)
        self.env = isaac_env
        self.sim = isaac_sim
        self.pcgworker_ = PCGWorker(self.wfc_size, self.wfc_size)
        self.wave = self.pcgworker_.build_wave()
        # all empty tile
        self.all_empty_tile = np.ones((self.wfc_size * self.wfc_size, 1, 2)).astype(np.int32)
        self.seed = self.all_empty_tile
        self.gym = gymapi.acquire_gym()
        avatar_options = gymapi.AssetOptions()
        avatar_options.density = self.cfg["agent"]["density"]
        avatar_options.fix_base_link = self.cfg["agent"]["fix_base_link"]
        avatar_options.linear_damping = self.cfg["agent"]["linear_damping"]
        self.avatar_capsule_asset = self.gym.create_capsule(self.sim, 1, 1, avatar_options)
        food_options = gymapi.AssetOptions()
        food_options.density = self.cfg["food"]["density"]
        food_options.fix_base_link = self.cfg["food"]["fix_base_link"]
        food_options.linear_damping = self.cfg["food"]["linear_damping"]
        avatar_options = gymapi.AssetOptions()
        avatar_options.density = self.cfg["agent"]["density"]
        avatar_options.fix_base_link = self.cfg["agent"]["fix_base_link"]
        avatar_options.linear_damping = self.cfg["agent"]["linear_damping"]
        self.avatar_capsule_asset = self.gym.create_capsule(self.sim, 1, 1, avatar_options)
        food_options = gymapi.AssetOptions()
        food_options.density = self.cfg["food"]["density"]
        food_options.fix_base_link = self.cfg["food"]["fix_base_link"]
        food_options.linear_damping = self.cfg["food"]["linear_damping"]
        self.food_capsule_asset = self.gym.create_capsule(self.sim, 1, 1, food_options)
        self.cube_assets = assets_dict["cube_assets"]
        self.ramp_assets = assets_dict["ramp_assets"]
        self.corner_assets = assets_dict["corner_assets"]
        self.cfg["food"]["color"] = "green"
        self.color_name_list = ["gray", "blue", "orange", "red", "white", "yellow"]
        # object pool indexed by grid
        self.grid_tile_handler = {}
        self.shape_props2 = gymapi.RigidShapeProperties()
        self.shape_props2.friction = 1
        self.shape_props2.rolling_friction = -1
        self.shape_props2.torsion_friction = -1
        self.shape_props2.compliance = 0
        self.shape_props2.restitution = 0
        self.initial_height = 11
        self.agent_handler = None
        self.food_handler = None
        self.camera_handler = None
        self.attractor_handle_agent = None
        self.agent_space = None
        self.food_space = None
        self.agent_idx = None
        self.food_idx = None
        self.space = None
        self.not_connect_map = None
        self.rank = rank
        self.all_cubes_handler = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        self.all_ramps_handler = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        self.all_corners_handler = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        # index for tensor version
        self.all_cubes_index = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        self.all_ramps_index = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        self.all_corners_index = {
            "gray": [],
            "blue": [],
            "orange": [],
            "red": [],
            "white": [],
            "yellow": []
        }
        self.all_blocks_idx = []
        self.all_block_handlers = []
        self.tile_to_position = []
        self.grid_tile_blocks_index = {}
        self.initial_pos = gymapi.Transform()
        self.initial_pos.p = gymapi.Vec3(0, 0, -100)
        self.initial_pos.r = gymapi.Quat(0, 1, 0, 1)
        self.num_cubes = 81
        self.num_ramps = 0
        self.num_corners = 0
        # predefine all blocks
        self.__set_tile_position()
        self.__create_all_actors()

    def set_wave(self, wave):
        self.wave = wave
        result_seed, succeed = wave.get_result()
        if succeed:
            self.seed = np.array(result_seed).astype(np.int32)
        else:
            self.seed = self.all_empty_tile

    def get_space_from_wave(self, wave=None):
        if not wave:
            wave = self.wave
        mask, _ = self.pcgworker_.connectivity_analysis(wave=wave, visualize_=False, to_file=False)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilate = cv2.dilate(mask, kernel, 1)
        # reduce mask to 9x9 for processing
        self.not_connect_map = (dilate != np.argmax(np.bincount(dilate.reshape(-1)))).astype(int)
        reduced_map = einops.reduce(mask, "(h a) (w b) -> h w", a=20, b=20, reduction='max')
        flatten_reduced_map = reduced_map.reshape(-1)
        # use maxium playable area as probility space
        return np.flatnonzero(flatten_reduced_map == np.argmax(np.bincount(flatten_reduced_map))).astype(np.int32)

    def mutate_a_new_map(self, base_wave=None, size=81):
        if base_wave is None:
            base_wave = self.wave
        self.wave = self.pcgworker_.mutate(base_wave, size)
        result_seed, succeed = self.wave.get_result()
        if succeed:
            self.seed = np.array(result_seed).astype(np.int32)
        else:
            self.seed = self.all_empty_tile
        return self.wave

    def render_in_sim(self, wave=None):
        if wave is None:
            wave = self.wave
        result_seed, succeed = wave.get_result()
        if succeed:
            self.seed = np.array(result_seed).astype(np.int32)
        else:
            self.seed = self.all_empty_tile
        self.space = self.get_space_from_wave()
        # create actors based on WFC seed
        self.__resetActors()
        self.__gridRender(self.seed)
        self.__placeAgentAndFood()

    def render_in_sim_tensor(self, root_tensor, wave=None):
        self.resetActors_tensor(root_tensor=root_tensor)
        if wave is None:
            wave = self.wave
        result_seed, succeed = wave.get_result()
        if succeed:
            self.seed = np.array(result_seed).astype(np.int32)
        else:
            self.seed = self.all_empty_tile
        self.space = self.get_space_from_wave()
        # create actors based on WFC seed
        self.__gridRender(self.seed, istensor=True, root_tensor=root_tensor)
        self.placeAgentAndFood_tensor(root_tensor=root_tensor)
        all_list = self.all_blocks_idx.copy()
        all_list.append(self.agent_idx)
        all_list.append(self.food_idx)
        return root_tensor

    def resetActors_tensor(self, root_tensor):
        root_tensor[self.all_blocks_idx] = 0.0
        root_tensor[self.all_blocks_idx, 2] = -100.0
        return root_tensor

    def placeAgentAndFood_tensor(self, root_tensor):
        space = self.space.copy()
        self.agent_space = list(space)
        assert len(space) > 1, len(space)
        # choose a place for agent
        random_cell = np.random.choice(self.agent_space)
        cell_top_tile_index = self.grid_tile_blocks_index[random_cell][-1]
        # reset agent's pose
        # set all speed to zero 
        root_tensor[self.agent_idx, 7:13] = 0.0
        # set agent's position
        root_tensor[self.agent_idx, 0:3] = root_tensor[cell_top_tile_index, 0:3] + torch.tensor([0, 0, self.prefab_height * 1.5], dtype=torch.float32, device=root_tensor.device)
        # set agent's rotation
        # root_tensor[self.agent_idx, 3:7] = torch.tensor([1, 0, 1, 0], dtype=torch.float32, device=root_tensor.device)
        # choose a place for food
        self.food_space = self.agent_space.copy()
        self.food_space.remove(random_cell)
        random_cell = np.random.choice(list(self.food_space))
        cell_top_tile_index = self.grid_tile_blocks_index[random_cell][-1]
        # reset food's pose
        # set all speed to zero
        root_tensor[self.food_idx, 7:13] = 0.0
        # set food's position
        root_tensor[self.food_idx, 0:3] = root_tensor[cell_top_tile_index, 0:3] + torch.tensor([0, 0, self.prefab_height * 1.5], dtype=torch.float32, device=root_tensor.device)
        # set food's rotation
        # root_tensor[self.food_idx, 3:7] = torch.tensor([1, 0, 1, 0], dtype=torch.float32, device=root_tensor.device)
        return root_tensor

    """
     private methods
    """

    def __set_tile_position(self):
        for i in range(self.width):
            for j in range(self.height):
                self.tile_to_position.append([i * self.prefab_height, j * self.prefab_height])
        self.tile_to_position = np.array(self.tile_to_position)

    def __instanteAsset(self, asset, pos, name, colli_group, colli_filter):
        return self.gym.create_actor(self.env, asset, pos, name, colli_group, colli_filter)

    def __moveActor(self, actor, pos):
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.food_handler, gymapi.STATE_POS)
        body_states['pose']['p'].fill((pos.p.x, pos.p.y, pos.p.z))
        body_states['pose']['r'].fill((pos.r.x, pos.r.y, pos.r.z, pos.r.w))
        self.gym.set_actor_rigid_body_states(self.env, actor, body_states, gymapi.STATE_POS)

    # reset all actor in self.grid_tile_blocks to initial position
    def __resetActors(self):
        for color in self.color_name_list:
            for cube in self.all_cubes_handler[color]:
                if cube.used:
                    cube.used = False
                    self.__moveActor(cube.handler, self.initial_pos)
            for ramp in self.all_ramps_handler[color]:
                if ramp.used:
                    ramp.used = False
                    self.__moveActor(ramp.handler, self.initial_pos)
            for corner in self.all_corners_handler[color]:
                if corner.used:
                    corner.used = False
                    self.__moveActor(corner.handler, self.initial_pos)
        if self.agent_handler:
            self.__moveActor(self.agent_handler, self.initial_pos)
        if self.food_handler:
            self.__moveActor(self.food_handler, self.initial_pos)

    def __create_all_actors(self):
        # create all cubes, and store them regularly
        for i, color in enumerate(self.color_name_list):
            cube_asset = self.cube_assets[i]
            for j in range(self.num_cubes):
                handler = self.__instanteAsset(asset=cube_asset, pos=self.initial_pos, name=f"{color}_cube_{j}",
                                               colli_group=self.rank, colli_filter=0)
                self.all_cubes_handler[color].append(
                    Block(handler=handler, used=False, type="cube", tag=f"{color}_cube_{j}"))
                idx = self.gym.get_actor_index(self.env, handler, gymapi.DOMAIN_SIM)
                self.all_cubes_index[color].append(idx)
                self.all_blocks_idx.append(idx)
                self.all_block_handlers.append(handler)

        # create all ramps, and store them regularly
        for i, color in enumerate(self.color_name_list):
            ramp_asset = self.ramp_assets[i]
            for j in range(self.num_ramps):
                handler = self.__instanteAsset(asset=ramp_asset, pos=self.initial_pos, name=f"{color}_ramp_{j}",
                                               colli_group=self.rank, colli_filter=0)
                self.all_ramps_handler[color].append(
                    Block(handler=handler, used=False, type="ramp", tag=f"{color}_ramp__{j}"))
                self.all_ramps_index[color].append(self.gym.get_actor_index(self.env, handler, gymapi.DOMAIN_SIM))
                self.all_blocks_idx.append(idx)
                self.all_block_handlers.append(handler)
        # create all corners, and store them regularly
        for i, color in enumerate(self.color_name_list):
            corner_asset = self.corner_assets[i]
            for j in range(self.num_corners):
                handler = self.__instanteAsset(asset=corner_asset, pos=self.initial_pos, name=f"{color}_corner_{j}",
                                               colli_group=self.rank, colli_filter=0)
                self.all_corners_handler[color].append(
                    Block(handler=handler, used=False, type="corner", tag=f"{color}_corner_{j}"))
                self.all_corners_index[color].append(self.gym.get_actor_index(self.env, handler, gymapi.DOMAIN_SIM))
                self.all_blocks_idx.append(idx)
                self.all_block_handlers.append(handler)
        self.__createActorAndFood()

    @staticmethod
    def __rotateAsMark(mark, name="cube"):
        if mark > 3:
            mark = 0
            print("Rotate data is out of range 0-3, fallback to 0")
        base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        base_axis = gymapi.Vec3(0, 0, 1)
        if name == "corner":
            base_axis = gymapi.Vec3(1, 0, 0)
            base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5 * np.pi)
        elif name == "ramp":
            base_axis = gymapi.Vec3(0, 1, 0)
            base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * np.pi) * gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), -0.5 * np.pi)
            mark = -mark
        rotation = base_angle * gymapi.Quat.from_axis_angle(base_axis, mark * 0.5 * np.pi)
        return rotation

    def __rotateAsMark_tensor(self, mark, name="cube"):
        if mark > 3:
            mark = 0
            print("Rotate data is out of range 0-3, fallback to 0")
        base_angle = axis_angle_to_quaternion(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device))
        base_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        if name == "corner":
            quat1 = axis_angle_to_quaternion(
                torch.tensor([0.0, 0.5 * np.pi, 0.0], dtype=torch.float32, device=self.device))
            quat2 = axis_angle_to_quaternion(torch.tensor([-0.5 * np.pi, 0.0, 0.0], dtype=torch.float32))
            base_angle = quaternion_multiply(quat1, quat2)
            mark = -mark
        elif name == "ramp":
            # quat1 = axis_angle_to_quaternion(torch.tensor([0.5 * np.pi, 0, 0.0], dtype=torch.float32)) 
            quat1 = axis_angle_to_quaternion(torch.tensor([0, 0, 0.5 * np.pi], dtype=torch.float32))
            quat2 = axis_angle_to_quaternion(torch.tensor([-0.5 * np.pi, 0, 0.0], dtype=torch.float32))
            base_angle[:] = quaternion_multiply(quat1, quat2)
            # base_angle[:] = axis_angle_to_quaternion(torch.tensor([0.5 * np.pi, 0.0, 0.0], dtype=torch.float32)) 
            mark = -mark
        rotation = quaternion_multiply(base_angle, axis_angle_to_quaternion(base_axis * 0.5 * np.pi * mark))
        return rotation

    def __placeAgentAndFood(self, connectivity_map=None):
        # space = self.get_space_from_wave()
        space = self.space.copy()
        assert len(space) > 1, len(space)
        self.agent_space = list(space)
        # choose a place for agent
        random_cell = np.random.choice(self.agent_space)
        cell_top_tile_handler = self.grid_tile_blocks[random_cell][-1].handler
        # reset agent's pose
        cell_body_states = self.gym.get_actor_rigid_body_states(self.env, cell_top_tile_handler, gymapi.STATE_ALL)
        actor_cell_pose = gymapi.Transform()
        actor_cell_pose.p = gymapi.Vec3(cell_body_states["pose"]["p"][0][0], cell_body_states["pose"]["p"][0][1],
                                        cell_body_states["pose"]["p"][0][2] + self.prefab_height * 1.5)
        actor_cell_pose.r = gymapi.Quat(0, 1, 0, 1)
        # reset agent's pose
        self.__moveActor(self.agent_handler, actor_cell_pose)
        # choose a place for food
        self.food_space = self.agent_space.copy()
        self.food_space.remove(random_cell)
        random_cell = np.random.choice(list(self.food_space))
        cell_top_tile_handler = self.grid_tile_blocks[random_cell][-1].handler
        cell_body_states = self.gym.get_actor_rigid_body_states(self.env, cell_top_tile_handler, gymapi.STATE_ALL)
        food_cell_pose = gymapi.Transform()
        food_cell_pose.p = gymapi.Vec3(cell_body_states["pose"]["p"][0][0], cell_body_states["pose"]["p"][0][1],
                                       cell_body_states["pose"]["p"][0][2] + self.prefab_height * 1.5)
        food_cell_pose.r = gymapi.Quat(0, 1, 0, 1)
        # reset food's pose
        self.__moveActor(self.food_handler, food_cell_pose)

    def __applytoActor(self, tile_type, pos, cell_index, handler_list):
        # get a not used handle and use it
        select_cube = None
        for handle in handler_list:
            select_handle = None
            if not handle.used:
                select_handle = handle
                handle.used = True
                self.__moveActor(handle.handler, pos)
                idx = self.gym.get_actor_index(self.env, handle.handler, gymapi.DOMAIN_SIM)
                # add handler to objects pool
                self.grid_tile_blocks[cell_index].append(handle)
                self.grid_tile_blocks_index[cell_index].append(idx)
                break
        if select_handle is None:
            raise Exception(f"No more {tile_type} available")

    def __applytoActor_tensor(self, tile_type, pose_ij, cell_index, root_tensor, used_object_idx, index_list):
        select_index = None
        for idx in index_list:
            if idx not in used_object_idx:
                select_index = idx
                used_object_idx.append(idx)
                self.grid_tile_blocks_index[cell_index].append(idx)
                break
        if select_index is None:
            raise Exception(f"No more {tile_type} available")
        root_tensor[select_index, 0:3] = pose_ij[0]
        root_tensor[select_index, 3:7] = pose_ij[1]

    def __getPos(self, i, j, istensor=False):
        if not istensor:
            pose_ij = gymapi.Transform()
            # y-axis up, same as Unity3D and the original model's coordinal system 
            pose_ij.p = gymapi.Vec3(i * self.prefab_size, j * self.prefab_size, 0)
            pose_ij.r = gymapi.Quat(0, 0, 0, 1)
        else:
            pose_ij = [
                torch.tensor([i * self.prefab_size, j * self.prefab_size, 0], dtype=torch.float32, device=self.device),
                torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)]
        return pose_ij

    def __gridRender(self, seed, istensor=False, root_tensor=None):
        if root_tensor is None:
            istensor = False
        # create actors based on WFC seed
        cell_index = 0
        self.grid_tile_blocks = {}
        self.grid_tile_blocks_index = {}
        used_object_idx = []
        for i in range(0, 9):
            for j in range(0, 9):
                self.grid_tile_blocks[cell_index] = []
                self.grid_tile_blocks_index[cell_index] = []
                tile_ = seed[i * 9 + j][0][0]
                rot = seed[i * 9 + j][0][1]
                pose_ij = self.__getPos(i, j, istensor)
                # 1 - 5 stacked cubes
                if 0 < tile_ <= 5:
                    if istensor:
                        for q in range(tile_):
                            pose_ij[0][2].fill_(q * self.prefab_height)
                            pose_ij[1][:] = self.__rotateAsMark_tensor(rot)
                            self.__applytoActor_tensor(tile_type="cube", pose_ij=pose_ij, cell_index=cell_index,
                                                       root_tensor=root_tensor, used_object_idx=used_object_idx,
                                                       index_list=self.all_cubes_index[self.color_name_list[q]])
                    else:
                        for q in range(tile_):
                            pose_ij.p.z = q * self.prefab_height
                            pose_ij.r = self.__rotateAsMark(rot)
                            self.__applytoActor(tile_type="cube", pos=pose_ij, cell_index=cell_index,
                                                handler_list=self.all_cubes_handler[self.color_name_list[q]])
                # 6 -9 corners
                elif 6 <= tile_ <= 9:
                    if istensor:
                        for q in range(tile_ - 5):
                            pose_ij[0][2].fill_(q * self.prefab_height)
                            pose_ij[1][:] = self.__rotateAsMark_tensor(rot)
                            self.__applytoActor_tensor(tile_type="cube", pose_ij=pose_ij, cell_index=cell_index,
                                                       root_tensor=root_tensor, used_object_idx=used_object_idx,
                                                       index_list=self.all_cubes_index[self.color_name_list[q]])
                        # Instantiate the corners at the position
                        pose_ij[0][2].fill_((tile_ - 5) * self.prefab_height)
                        pose_ij[1][:] = self.__rotateAsMark_tensor(rot, name="corner")
                        self.__applytoActor_tensor(tile_type="corner", pose_ij=pose_ij, cell_index=cell_index,
                                                   root_tensor=root_tensor, used_object_idx=used_object_idx,
                                                   index_list=self.all_corners_index[self.color_name_list[tile_ - 5]])
                    else:
                        # Instantiate base cube at the position for high layer corners
                        # The bottom layer is a cube, so the corners tile will never be the bottom layer
                        for q in range(tile_ - 5):
                            pose_ij.p.z = q * self.prefab_height
                            pose_ij.r = self.__rotateAsMark(rot)
                            self.__applytoActor(tile_type="cube", pos=pose_ij, cell_index=cell_index,
                                                handler_list=self.all_cubes_handler[self.color_name_list[q]])
                        # Instantiate the corners at the position
                        pose_ij.p.z = (tile_ - 5) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="corner")
                        self.__applytoActor(tile_type="corner", pos=pose_ij, cell_index=cell_index,
                                            handler_list=self.all_corners_handler[self.color_name_list[tile_ - 5]])

                elif 10 <= tile_ <= 13:
                    if istensor:
                        for q in range(tile_ - 9):
                            pose_ij[0][2].fill_(q * self.prefab_height)
                            pose_ij[1][:] = self.__rotateAsMark_tensor(rot)
                            self.__applytoActor_tensor(tile_type="cube", pose_ij=pose_ij, cell_index=cell_index,
                                                       root_tensor=root_tensor, used_object_idx=used_object_idx,
                                                       index_list=self.all_cubes_index[self.color_name_list[q]])
                        # Instantiate the ramps at the position
                        pose_ij[0][2].fill_((tile_ - 9) * self.prefab_height)
                        pose_ij[1][:] = self.__rotateAsMark_tensor(rot, name="ramp")
                        self.__applytoActor_tensor(tile_type="ramp", pose_ij=pose_ij, cell_index=cell_index,
                                                   root_tensor=root_tensor, used_object_idx=used_object_idx,
                                                   index_list=self.all_ramps_index[self.color_name_list[tile_ - 9]])
                    else:
                        # Instantiate bottom cubes first
                        for q in range(tile_ - 9):
                            pose_ij.p.z = q * self.prefab_height
                            pose_ij.r = self.__rotateAsMark(rot)
                            self.__applytoActor(tile_type="cube", pos=pose_ij, cell_index=cell_index,
                                                handler_list=self.all_cubes_handler[self.color_name_list[q]])
                        # Instantiate the ramps at the position
                        pose_ij.p.z = (tile_ - 9) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="ramp")
                        self.__applytoActor(tile_type="ramp", pos=pose_ij, cell_index=cell_index,
                                            handler_list=self.all_ramps_handler[self.color_name_list[tile_ - 9]])
                else:
                    raise Exception("The tile Data is out of the range: 0 to 13")
                cell_index += 1

    def __createActorAndFood(self):
        # create capsule_asset actor in the environment
        color_red = gymapi.Vec3(1, 0, 0)
        color_green = gymapi.Vec3(0, 1, 0)
        # generate random x,y coordinates in range [0,18] for the actor
        x = random.uniform(0, 16)
        y = random.uniform(0, 16)
        facing_ = 0.25 * (float)(random.randint(0, 8))
        r_ = gymapi.Quat(0, 1, 0, 1) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), facing_ * np.pi)
        # r_ = gymapi.Quat(0,1,0,1)
        startpose = gymapi.Transform()
        startpose.p = gymapi.Vec3(x, y, self.initial_height)
        startpose.r = gymapi.Quat(0, 1, 0, 1)
        cap_handle = self.gym.create_actor(self.env, self.avatar_capsule_asset, startpose, 'agent', self.rank, 0)
        self.gym.set_actor_rigid_shape_properties(self.env, cap_handle, [self.shape_props2])
        self.gym.set_actor_scale(self.env, cap_handle, 0.4)
        self.gym.set_rigid_body_color(self.env, cap_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_red)
        self.agent_handler = cap_handle
        # set random position for food
        x = random.uniform(0, 16)
        y = random.uniform(0, 16)
        # create capsule_asset food in the environment
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(x, y, self.initial_height)
        pose.r = gymapi.Quat(0, 1, 0, 1)
        food_handle = self.gym.create_actor(self.env, self.food_capsule_asset, pose, 'food', self.rank, 0)
        self.gym.set_actor_rigid_shape_properties(self.env, food_handle, [self.shape_props2])
        self.gym.set_actor_scale(self.env, food_handle, 0.6)
        # self.food_handles.append(food_handle)
        self.gym.set_rigid_body_color(self.env, food_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_green)
        self.food_handler = food_handle
        # keep agent and food stand
        att_pose = gymapi.Transform()
        att_pose.p = gymapi.Vec3(x, y, self.initial_height)
        att_pose.r = r_
        attractor_properties_ = gymapi.AttractorProperties()
        attractor_properties_.stiffness = 5e10
        attractor_properties_.damping = 5e10
        attractor_properties_.axes = gymapi.AXIS_SWING_1 | gymapi.AXIS_SWING_2  # 48
        # attractor_properties_.axes = 0
        attractor_properties_.target = att_pose
        attractor_properties_.rigid_handle = cap_handle
        attractor_handle_ = self.gym.create_rigid_body_attractor(self.env, attractor_properties_)
        self.attractor_handle_agent = attractor_handle_

        attractor_properties_food = gymapi.AttractorProperties()
        attractor_properties_food.stiffness = 5e10
        attractor_properties_food.damping = 5e10
        attractor_properties_food.axes = gymapi.AXIS_SWING_1 | gymapi.AXIS_SWING_2
        attractor_properties_food.target = pose
        attractor_properties_food.rigid_handle = food_handle

        h1 = self.gym.create_camera_sensor(self.env, self.camera_properties)
        # camera_offset = gymapi.Vec3(-2, 0, 0)
        camera_offset = gymapi.Vec3(-2, 0, 0)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.4 * np.pi)
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        # camera_rotation = gymapi.Quat(0,,0,1)
        body_handle = self.gym.get_actor_rigid_body_handle(self.env, cap_handle, 0)
        self.gym.attach_camera_to_body(h1, self.env, body_handle, gymapi.Transform(camera_offset, camera_rotation),
                                       gymapi.FOLLOW_TRANSFORM)
        self.camera_handler = h1
        self.agent_idx = self.gym.get_actor_index(self.env, self.agent_handler, gymapi.DOMAIN_SIM)
        self.food_idx = self.gym.get_actor_index(self.env, self.food_handler, gymapi.DOMAIN_SIM)
