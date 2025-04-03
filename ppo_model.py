from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from config import simple_two_agent_config
from main_env import WelfareSPMEnv

# Step 1: Load config and build env
config = simple_two_agent_config
env = make_vec_env(lambda: WelfareSPMEnv(config), n_envs=1)

# Step 2: Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=0)
model.learn(total_timesteps=10)

# Step 3: Evaluate on 5 episodes
obs = env.reset()
for episode in range(10):
    episode_reward = 0
    done = False
    prices_offered = []

    while not done:
        action, _ = model.predict(obs)
        action = action[0]  # Unpack from VecEnv batch

        # Convert action to price
        price_bin_idx = action
        price = float(price_bin_idx * (config['max_price'] / (config['pricing_levels'] - 1)))
        prices_offered.append(price)

        obs, reward, done, _ = env.step([action])  # wrap back in list for VecEnv
        episode_reward += reward[0]  # unpack reward from vector

    # After episode is done
    print(f"Episode {episode + 1}")
    
    if len(prices_offered) > 1:
        print(f"Prices offered: First = ${prices_offered[0]:.2f}, Second = ${prices_offered[1]:.2f}")
    else:
        print(f"Price offered: ${prices_offered[0]:.2f}")

    
    print(f"Total episode reward: {episode_reward:.2f}")
    agent_vals = env.get_attr("agents")[0]
    clean_vals = [int(v) for v in agent_vals]  
    print(f"Agent valuations this episode: {clean_vals}\n")

    obs = env.reset()

print(obs)