from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.roomgrid import reject_next_to
from num2words import num2words
import itertools as itt, random
import numpy as np
from copy import deepcopy
from gym import spaces

################################################################################
# Heleper functions
################################################################################

def reject_dist_3(env, pos):
    """
    Function to filter out object positions that are 3 away from
    the agent's starting positon
    """

    sx, sy = env.agent_pos
    x, y = pos
    d = abs(sx - x) + abs(sy - y)
    return d <= 3

def too_close(pos, entities, min_dist):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """
    for entity_pos in entities:
        sx, sy = entity_pos
        x, y = pos
        d = abs(sx - x) + abs(sy - y)
        if d <= min_dist:
            return True
    return False

################################################################################
# Synthetic instruction generation
################################################################################

AVOID_OBJ = {
    0: 'lava',
    1: 'grass',
    2: 'water'
}

OBJ_TO_NUMBER = {
    'lava': -3,
    'grass': -2,
    'water': 1
}

OBJ_TO_CON = {
    'lava':0,
    'grass':1,
    'water':2,
    None:-1
}
AVOID_OBJ_VALS = list(AVOID_OBJ.values())


NEG = ['do not', 'don\'t', 'never']
VNOP = ['cross', 'touch']
VPROP = ['move', 'go', 'travel', 'pass', 'walk']
VNOP_GERUND = ['crossing', 'touching']
VPROP_GERUND = ['moving', 'going', 'traveling', 'passing', 'walking']
PROP = ['through', 'on', 'upon']
VGOAL = ['go to', 'reach', 'move to']

def num_to_str(num):
    if num==1:
        return 'once'
    elif num==2:
        return 'twice'
    else:
        return num2words(num)+' times'

def constraint1(avoid, nu):
    ne = random.choice(NEG)
    vn = random.choice(VNOP)
    return f'{ne} {vn} {avoid} more than {nu}'


def constraint2(avoid, nu):
    ne = random.choice(NEG)
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f'{ne} {vp} {pr} {avoid} more than {nu}'


def constraint3(avoid, nu):
    vn = random.choice(VNOP)
    return f'{vn} {avoid} less than {nu}'


def constraint4(avoid, nu):
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f'{vp} {pr} {avoid} less than {nu}'

CONSTRAINTS = {
    0: constraint1,
    1: constraint2,
    2: constraint3,
    3: constraint4
}

def make_budgetary_constraint(avoid_obj, hc):
    if hc == 1:
        ne = random.choice(NEG)
        vn = random.choice(VNOP)
        return f'{ne} {vn} {avoid_obj}'
    opt = np.random.choice(4)
    if opt==0 or opt==1:
        nu = num_to_str(hc-1)
    else: 
        nu = num_to_str(hc)
    return CONSTRAINTS[opt](avoid_obj, nu)

def make_sequential_constraint(first_obj, avoid_obj):
    sequential_str=[
        f"After walking through {first_obj}, please avoid stepping on {avoid_obj}.",
        f"After walking past {first_obj}, don't go back to {avoid_obj}.",
        f"Once you've walked past {first_obj}, make sure you don't set foot on {avoid_obj}.",
    ]
    return random.choice(sequential_str)


def make_logical_constrain(total_blood):
    logical_str=[
        f"You only have {total_blood} HP, please don't die.",
        f"You're left with {total_blood} HP, make every hit count and stay alive!",
        f"With only {total_blood} HP remaining, survival becomes critical. Stay vigilant!",
        f"With {total_blood} HP left, your fate hangs by a thread. Don't let it snap!"
    ]
    return random.choice(logical_str)


################################################################################
# HazardWorld base
################################################################################
OBJECT_TO_PIX={
    'grass':[11,1,0],
    'water':[12,2,0],
    'lava':[9,0,0]
}
class HazardWorldBase(MiniGridEnv):
    """
    Pick up 3 objects while avoiding many potential hazards.
    Potential hazards are specified by the constraint stored in the mission
    field. The base HazardWorld environment contains budgetary constraints.
    """
    def __init__(self, size=13, seed=None):
        super().__init__(
            grid_size=size,
            max_steps=199,
            see_through_walls=True,
            seed=None,
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10,
            high=20,
            shape=(7, 7, 3),
            dtype='int32'
        )

    def isEmpty(self, i, j):
        return self.grid.get(i, j) == None

    def step(self, action):
        # if not isinstance(action, int):
        #     action = np.argmax(action)
        obs, reward, _, info = super().step(action)
        terminated=False
        truncated=False
        cost=0
        if action == self.actions.pickup:
            if self.carrying != None:
                if self.carrying.type == 'ball':
                    reward += self._reward()
                    self.objs_collected += 1
                elif self.carrying.type == 'box':
                    reward += self._reward() * 2
                    self.objs_collected += 1
                elif self.carrying.type == 'key':
                    reward += self._reward() * 3
                    self.objs_collected += 1
                self.carrying = None
        curr_cell = self.grid.get(*self.agent_pos)
        celltype=None
        if curr_cell != None:
            celltype=curr_cell.type
        info['constrain']=OBJ_TO_CON[celltype]
        if self.objs_collected == 3:
            terminated = True
        obs = self.gen_obs()
        if celltype!=None:
            obs['image'][3,6]=OBJECT_TO_PIX[celltype]
        if self.step_count >= self.max_steps:
            truncated=True
        obs['image']=np.array(obs['image'],dtype=np.int32)
        for row in obs['image']:
            for item in row:
                if item[0]==11:
                    item[2]=-2
                elif item[0]==12:
                    item[2]=1
                elif item[0]==9:
                    item[2]=-3
        if self.agent_pos[0] == self.prev_agent_pos[0] and self.agent_pos[1] == self.prev_agent_pos[1]:
            self.stay_count += 1
        else:
            self.stay_count = 0
        info['penalty'] = 0
        if self.stay_count > 5:
            info['penalty'] = -0.5
            
        self.prev_agent_pos = deepcopy(self.agent_pos)
        
        if not self.is_violation:
            if self.env_type=="budgetary":
                if curr_cell != None and curr_cell.type == self.avoid_obj:
                    self.violations += 1
                if self.violations>=self.hc:
                    cost=1.0
                    # terminated=True
                    self.is_violation=True
            elif self.env_type=="sequential":
                if curr_cell != None and curr_cell.type == self.first_obj:
                    self.is_second_stage = True
                if self.is_second_stage and curr_cell != None and curr_cell.type == self.second_obj:
                    cost=1.0
                    self.is_violation=True
            elif self.env_type=="logical":
                if curr_cell != None:
                    self.total_blood+=OBJ_TO_NUMBER[curr_cell.type]
                if self.total_blood<=0:
                    cost=1.0
                    # terminated=True
                    self.is_violation=True
            else:
                raise NotImplementedError
            
        if cost==1.0:
            terminated=True
        
        return obs['image'], reward, cost, terminated, truncated, info

    def reset(self):
        obs = super().reset()
        curr_cell = self.grid.get(*self.agent_pos)
        self.prev_agent_pos = deepcopy(self.agent_pos)
        self.stay_count = 0
        celltype=None
        if curr_cell != None:
            celltype=curr_cell.type
        self.is_violation=False
        info={}
        info['penalty'] = 0
        info['constrain']=OBJ_TO_CON[celltype]
        info['mission']=self.mission
        if celltype!=None:
            obs['image'][3,6]=OBJECT_TO_PIX[celltype]
        obs['image']=np.array(obs['image'],dtype=np.int32)
        for row in obs['image']:
            for item in row:
                if item[0]==11:
                    item[2]=-2
                elif item[0]==12:
                    item[2]=1
                elif item[0]==9:
                    item[2]=-3  
        return obs['image'],info

    def _gen_grid(self, width, height, sparsity=0.25):
        assert width % 2 == 1 and height % 2 == 1
        # HazardWorld grid size must be odd
        self.grid = Grid(width, height)
        # pick a cost entity
        self.objs_collected = 0
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # The child class must place reward entities


################################################################################
# LavaWall 
################################################################################

class HazardWorldLavaWall(HazardWorldBase):

    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        # add obstacles
                # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0
        
        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: 1]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(Lava(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.place_agent()

        self.env_type="budgetary"
        # add 3 reward entities
        self.place_obj(Ball('red'), reject_fn=reject_next_to)
        self.place_obj(Box('yellow'), reject_fn=reject_next_to)
        self.place_obj(Key('blue'), reject_fn=reject_next_to)

        self.avoid_obj = 'lava'
        self.hc = random.choice([1,2,3])
        self.violations = 0
        self.mission = make_budgetary_constraint(self.avoid_obj, self.hc)


register(
    id='MiniGrid-HazardWorld-LavaWall-v0',
    entry_point='gym_minigrid.envs:HazardWorldLavaWall'
)


################################################################################
# Budgetary Constraints
################################################################################

class HazardWorldBudgetary(HazardWorldBase):

    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        # add obstacles
        for i in range(1, height-1):
            for j in range(1, width-1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)

        self.place_agent()
        self.env_type="budgetary"
        # add 3 reward entities
        self.place_obj(Ball('red'), reject_fn=reject_next_to)
        self.place_obj(Box('yellow'), reject_fn=reject_next_to)
        self.place_obj(Key('blue'), reject_fn=reject_next_to)
        
        self.avoid_obj = random.choice(AVOID_OBJ_VALS)
        self.hc = random.choice([3,5,8,10])
        self.violations = 0
        
        self.mission = make_budgetary_constraint(self.avoid_obj, self.hc)
        

register(
    id='MiniGrid-HazardWorld-B-v0',
    entry_point='gym_minigrid.envs:HazardWorldBudgetary'
)

################################################################################
# Sequential Constraints
################################################################################

class HazardWorldSequential(HazardWorldBase):

    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        # avoid object is chosen for free in super class
        self.first_obj = random.choice(AVOID_OBJ_VALS)
        self.avoid_obj_vals = [obj for obj in AVOID_OBJ_VALS if obj != self.first_obj]
        self.second_obj = random.choice(self.avoid_obj_vals)
        self.mission = make_sequential_constraint(self.first_obj, self.second_obj)
        self.is_second_stage=False
        self.env_type="sequential"
        for i in range(1, height-1):
            for j in range(1, width-1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)

        self.place_agent()

        # add 3 reward entities
        self.place_obj(Ball('red'), reject_fn=reject_next_to)
        self.place_obj(Box('yellow'), reject_fn=reject_next_to)
        self.place_obj(Key('blue'), reject_fn=reject_next_to)

register(
    id='MiniGrid-HazardWorld-S-v0',
    entry_point='gym_minigrid.envs:HazardWorldSequential'
)

################################################################################
# Logical Constraints
################################################################################

class HazardWorldLogical(HazardWorldBase):

    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
            # avoid object is chosen for free in super class
        self.env_type="logical"
        self.total_blood = random.choice([20,25,30])
        self.mission = make_logical_constrain(self.total_blood)
        for i in range(1, height-1):
            for j in range(1, width-1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)

        self.place_agent()
        # add 3 reward entities
        self.place_obj(Ball('red'), reject_fn=reject_next_to)
        self.place_obj(Box('yellow'), reject_fn=reject_next_to)
        self.place_obj(Key('blue'), reject_fn=reject_next_to)

register(
    id='MiniGrid-HazardWorld-L-v0',
    entry_point='gym_minigrid.envs:HazardWorldLogical'
)
