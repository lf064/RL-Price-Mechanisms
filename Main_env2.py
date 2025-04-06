import gymnasium 
from gymnasium import spaces
import numpy as np

class WelfareSPMEnv(gymnasium.Env):
    def __init__(self, config):
        super(WelfareSPMEnv, self).__init__()

        self.num_agents = config["num_agents"]       #2
        self.num_items = config["num_items"]         #1
        self.max_price = config["max_price"]         # 2
        self.pricing_levels = config["pricing_levels"] #: 0, 0.5, 1.0, 1.5, 2.0
        self.current_agent_idx = 0 # chosen agent index, hardcoded
        self.valuation_space = config["valuation_space"]


        # --- ACTION SPACE ---
        self.action_space = spaces.Discrete(self.pricing_levels) #index for choice of pricing options

        # --- OBSERVATION SPACE SETUP ---
        total_size = 1 + self.num_agents + self.num_items + 2 * self.num_agents * self.num_items  # creates total size of observation vector: 8: current agent idx, remaining agent, remaining items, allocated items, price history. All binary
        low = np.zeros(total_size, dtype=np.float32) # lowest possible value --> size 8. 
        high = np.concatenate([
            np.array([self.num_agents - 1], dtype=np.float32),
            np.ones(self.num_agents),
            np.ones(self.num_items),
            np.ones(self.num_agents * self.num_items),
            np.full(self.num_agents * self.num_items, self.max_price)
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) # defines upper bound of observation vector. 

        # --- INIT STATE VARIABLES ---
        self.remaining_agents = np.ones(self.num_agents, dtype=np.int32) # binary vector size of 2 number of agents
        self.remaining_items = np.ones(self.num_items, dtype=np.int32) # binary 1 until sold
        self.allocated_items = np.zeros((self.num_agents, self.num_items), dtype=np.float32) #agent 0 gets it then [1, 0]
        self.price_history = np.zeros((self.num_agents, self.num_items), dtype=np.float32) # agent 0 offered price 1.5 [1.]

        self.price_mapping = np.linspace(0, self.max_price, self.pricing_levels) # maps index of action to a price

    def get_observation(self):
        return np.concatenate([
            self.remaining_agents.astype(np.float32),
            self.remaining_items.astype(np.float32),
            self.allocated_items.flatten(),
            self.price_history.flatten(),
            np.array([self.current_agent_idx / self.num_agents], dtype=np.float32)  # 👈 Added step info
        ])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.remaining_agents = np.ones(self.num_agents, dtype=np.int32)
        self.remaining_items = np.ones(self.num_items, dtype=np.int32)
        self.allocated_items = np.zeros((self.num_agents, self.num_items), dtype=np.float32)
        self.price_history = np.zeros((self.num_agents, self.num_items), dtype=np.float32)
        self.valuations = np.random.choice(self.valuation_space, size=self.num_agents).astype(np.float32)
        self.max_social_welfare = max(self.valuations)
        self.current_agent_idx = 0

        return self.get_observation(), {}  # gymnasium requires (obs, info)

    
    def step(self, action):
        price = self.price_mapping[action]
        reward = 0.0
        terminated = False
        truncated = False

        agent_idx = self.current_agent_idx
        valuation = self.valuations[agent_idx]

        print("\n===============================")
        print(f"📌 Agent Index: {agent_idx}")
        print(f"💵 Valuation: {valuation}")
        print(f"🏷️  Offered Price: {price:.2f}")

        accepted = False

        if valuation > price:
            print("✅ Accepted: Valuation > Price")
            self.allocated_items[agent_idx][0] = 1
            self.price_history[agent_idx][0] = price
            self.remaining_items[0] = 0
            self.remaining_agents[agent_idx] = 0
            reward = valuation / self.max_social_welfare
            terminated = True
            accepted = True
        else:
            print("❌ Rejected: Valuation <= Price")
            self.remaining_agents[agent_idx] = 0
            self.current_agent_idx += 1

            # If all agents have been tried or no items left
            if self.current_agent_idx >= self.num_agents or self.remaining_items.sum() == 0:
                terminated = True
                reward = 0.0

        if terminated:
            print(f"🏁 Episode Done — Final Reward: {reward:.2f}")
            print(f"📈 Welfare Achieved: {valuation if accepted else 0}")
            print(f"🏆 Max Social Welfare: {self.max_social_welfare}")

        print(f"🎯 Remaining Agents: {self.remaining_agents}")
        print(f"📦 Remaining Items: {self.remaining_items}")
        print(f"📊 Allocation Matrix:\n{self.allocated_items}")
        print(f"💰 Price History:\n{self.price_history}")
        print(f"🛑 Done: {terminated}")
        print("===============================\n")

        return self.get_observation(), float(reward), terminated, truncated, {}



        
