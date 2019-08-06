import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.core.simulator import Observations
import numpy as np
import quaternion as q
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.follower = ShortestPathFollower(self.habitat_env.sim, 0.25, False)
        #self.config= config

    def reset_curr_episode(self) -> Observations:

        self._env._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"

        self._env.sim.reconfigure(self._env._config.SIMULATOR)

        observations = self.habitat_env._sim.reset()
        observations.update(
            self._env.task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        self._env._task.measurements.reset_measures(episode=self._env.current_episode)

        return observations

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_best_actions(self):
        act = self.follower.get_next_action(
            self.habitat_env.current_episode.goals[0].position
        )
        return act

    def get_agent_states(self):
        curr_state = self.habitat_env.sim.get_agent_state()
        return [curr_state.position, q.as_float_array(curr_state.rotation)]

    def get_episode_over(self):
        return self.habitat_env.episode_over

    def get_goal(self):
        return self.habitat_env.current_episode.goals[0].position

    def reconfigure(self,config):
        return self.habitat_env.sim.reconfigure(config.SIMULATOR)



def make_env_fn(config_env, rank):
    env = SimpleRLEnv(config=config_env)
    env.seed(rank * 1000)
    return env

def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def run_env(envs, episode_length, observations, run_mode = 'expert', actor = None):

    datas = [{'rgb': [], 'position': [], 'rotation': [], 'action': []},
             {'rgb': [], 'position': [], 'rotation': [], 'action': []},
             {'rgb': [], 'position': [], 'rotation': [], 'action': []}]

    paused = [False] * 3
    env_ind_states = np.arange(3)
    for t in range(episode_length):
        best_actions = envs.get_best_actions()#np.array(envs.get_best_actions())
        alive_indices = np.where(np.array(paused) == False)
        past_obs = observations

        curr_states = envs.get_agent_states()

        #best_actions[np.where(best_actions > 0)] = best_actions[np.where(best_actions > 0)] + 5
        #best_actions[np.where(envs.get_episode_overs()) == 1] = 0
        if best_actions > 0 : best_actions += 5

        if run_mode is not 'expert' and actor is not None :
            actions = actor.predict_action(observations)
            actions[np.where(actions > 0)] = actions[np.where(best_actions > 0)] + 5
            actions[np.where(envs.get_episode_overs()) == 1] = 0
        #outputs = envs.step(best_actions) if run_mode is 'expert' else envs.step(actions)
        #observations, rewards, dones, infos = [
        #    list(x) for x in zip(*outputs)
        #]
        observations, rewards, dones, infos = envs.step(best_actions) if run_mode is 'expert' else envs.step(actions)

        if dones : break
        for i, j in enumerate(alive_indices[0]):
            if dones == 1 : #if dones[i] == 1:
                ind = np.where(env_ind_states == j)
                #envs.pause_at(ind[0][0])
                #env_ind_states = np.delete(env_ind_states, ind)
                #paused[j] = True
                continue
            # print(i, best_actions[0],best_actions[1],best_actions[2])
            # if best_actions[i] == 0 : continue
            datas[j]['rgb'].append(past_obs['rgb'])
            datas[j]['position'].append(curr_states[0])
            datas[j]['rotation'].append(curr_states[1])
            datas[j]['action'].append(best_actions)


    return datas

