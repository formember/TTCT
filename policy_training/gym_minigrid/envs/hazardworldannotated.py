from gym_minigrid.minigrid import *
from gym_minigrid.envs import *

import json, pkgutil

class HazardWorldBudgetaryAnnotated(HazardWorldBudgetary):
    def __init__(self, mode='train'):
        data = pkgutil.get_data(__name__, f"data/budgetary-{mode}.json")
        self.annotation_dict = json.loads(data.decode('utf-8'))
        super().__init__()
        
    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        mission_id = self.avoid_obj + str(self.hc-1)
        self.mission = random.choice(self.annotation_dict[mission_id])

register(
    id='MiniGrid-HazardWorld-BA-v0',
    entry_point='gym_minigrid.envs:HazardWorldBudgetaryAnnotated'
)

class HazardWorldRelationalAnnotated(HazardWorldLogical):
    def __init__(self, mode='train'):
        data = pkgutil.get_data(__name__, f"data/relational-{mode}.json")
        self.annotation_dict = json.loads(data.decode('utf-8'))
        super().__init__()
        
    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        mission_id = self.avoid_obj + str(self.min_dist)
        self.mission = random.choice(self.annotation_dict[mission_id])

register(
    id='MiniGrid-HazardWorld-RA-v0',
    entry_point='gym_minigrid.envs:HazardWorldRelationalAnnotated'
)

class HazardWorldSequentialAnnotated(HazardWorldSequential):
    def __init__(self, mode='train'):
        data = pkgutil.get_data(__name__, f"data/sequential-{mode}.json")
        self.annotation_dict = json.loads(data.decode('utf-8'))
        super().__init__()
        
    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        if self.isBefore:
            mission_id = 'b' + self.first_obj + self.avoid_obj
        else:
            mission_id = 'a' + self.first_obj + self.second_obj
        self.mission = random.choice(self.annotation_dict[mission_id])

register(
    id='MiniGrid-HazardWorld-SA-v0',
    entry_point='gym_minigrid.envs:HazardWorldSequentialAnnotated'
)

