import gymnasium 
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from Main_env2 import WelfareSPMEnv 


config = {
    "num_agents": 2,
    "num_items": 1,
    "max_price": 2.0,
    "pricing_levels": 5,
    "valuation_space": [1, 2]

}

# --- Create environment instance ---
env = WelfareSPMEnv(config)

# (Optional) check that your env follows Gym API
check_env(env, warn=True)

# --- Train PPO agent ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    ent_coef=0.02  # 🔥 Higher entropy → more exploration
)

# Train for 10,000 timesteps (adjust as needed)
model.learn(total_timesteps=100_000)

# Save the model
model.save("ppo_welfare_max")

print(" Training complete. Model saved as 'ppo_welfare_max'.")

from stable_baselines3 import PPO
from Main_env2 import WelfareSPMEnv

# Reload env and model
config = {"num_agents": 2, "num_items": 1, "max_price": 2.0, "pricing_levels": 5, "valuation_space": [1, 2]}
env = WelfareSPMEnv(config)
model = PPO.load("ppo_welfare_max")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    total_reward += reward

print(f" Total Reward (Normalized Welfare): {total_reward:.2f}")

price_counts = {price: 0 for price in env.price_mapping}

for ep in range(100):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        price = env.price_mapping[action]
        price_counts[price] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

print("\n PPO Price Distribution:")
for price, count in price_counts.items():
    print(f"Price {price:.2f}: {count} times")


import numpy as np

# Run PPO Evaluation
ppo_rewards = []
ppo_prices = []

print("\n Evaluating PPO Strategy:")
for ep in range(100):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs)
        price = env.price_mapping[action]
        ppo_prices.append(price)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
    ppo_rewards.append(ep_reward)

# Manual Strategy Evaluation
manual_rewards = []
manual_prices = []

print("\n Evaluating Manual Strategy (1.5 → 0.5):")
for ep in range(100):
    env.reset()
    env.current_agent_idx = 0
    done = False
    reward = 0.0

    # Step 1: Offer 1.5 to Agent 0
    action = np.where(env.price_mapping == 1.5)[0][0]
    obs, reward, terminated, truncated, _ = env.step(action)
    manual_prices.append(1.5)

    if not (terminated or truncated):
        # Step 2: Offer 0.5 to Agent 1
        action = np.where(env.price_mapping == 0.5)[0][0]
        obs, reward, terminated, truncated, _ = env.step(action)
        manual_prices.append(0.5)

    manual_rewards.append(reward)

#  Results Summary
print("\n=====================  RESULTS =====================")
print(f" PPO Avg Reward:    {np.mean(ppo_rewards):.3f}")
print(f" PPO Success Rate:  {np.sum(np.array(ppo_rewards) > 0) / len(ppo_rewards) * 100:.1f}%")

print(f"\n Manual Avg Reward: {np.mean(manual_rewards):.3f}")
print(f" Manual Success Rate: {np.sum(np.array(manual_rewards) > 0) / len(manual_rewards) * 100:.1f}%")

from collections import Counter

ppo_price_counts = Counter(ppo_prices)
manual_price_counts = Counter(manual_prices)

print("\n PPO Price Distribution:")
for p, c in sorted(ppo_price_counts.items()):
    print(f"Price {p:.2f}: {c} times")

print("\n Manual Price Distribution:")
for p, c in sorted(manual_price_counts.items()):
    print(f"Price {p:.2f}: {c} times")

print("======================================================")
