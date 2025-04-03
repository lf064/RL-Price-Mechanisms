from config import simple_two_agent_config
from main_env import WelfareSPMEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    config = simple_two_agent_config
    env = make_vec_env(lambda: WelfareSPMEnv(config), n_envs=1)
    obs = env.reset()
    print(obs) 