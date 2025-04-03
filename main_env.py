import gym
import numpy as np
from gym import spaces
import config


class WelfareSPMEnv(gym.Env):
    def __init__(self, config):
        super(WelfareSPMEnv, self).__init__()

        self.num_agents = config["num_agents"]
        self.num_items = config["num_items"]
        self.delta = config["delta"]
        self.valuation_space = config["valuation_space"]
        self.pricing_levels = config["pricing_levels"]
        self.objective = config["objective"]
        self.max_price = config["max_price"]

        self.remaining_agents = list(range(self.num_agents))
        self.remaining_items = list(range(self.num_items))
        self.allocated_items = {}
        self.price_history = np.zeros((self.num_agents, self.num_items))
        self.action_space = spaces.Discrete(self.pricing_levels)

        self.observation_space = spaces.Dict({
            "remaining_agents": spaces.MultiBinary(self.num_agents),
            "remaining_items": spaces.MultiBinary(self.num_items),
            "allocated_items": spaces.Box(low=0, high=1, shape=(self.num_agents, self.num_items), dtype=np.float32),
            "price_history": spaces.Box(low=0, high=1, shape=(self.num_agents, self.num_items), dtype=np.float32),
            "agent_valuations": spaces.MultiDiscrete([int(max(self.valuation_space) + 1)] * self.num_agents)
        })

    def _get_allocated_items_matrix(self):
        matrix = np.zeros((self.num_agents, self.num_items), dtype=np.float32)
        for agent_id, items in self.allocated_items.items():
            for item_id in items:
                matrix[agent_id, item_id] = 1.0
        return matrix

    def reset(self):
        self.remaining_agents = list(range(self.num_agents))
        self.remaining_items = list(range(self.num_items))
        self.allocated_items = {i: [] for i in range(self.num_agents)}
        self.price_history = np.zeros((self.num_agents, self.num_items))
        self._generate_valuations()
        self.current_agent_idx = 0
        print("\n=== ENVIRONMENT RESET ===")
        print(f"Agents Valuations: {self.agents}")
        return self._get_observation()

    def _generate_valuations(self):
        self.agents = [np.random.choice(self.valuation_space) for _ in range(self.num_agents)]

        
    def _get_observation(self):
        obs = {
            "remaining_agents": np.array([1 if i in self.remaining_agents else 0 for i in range(self.config["num_agents"])]),
            "remaining_items": np.array([1 if i in self.remaining_items else 0 for i in range(self.config["num_items"])]),
            "allocated_items": self._get_allocated_items_matrix(),
            "price_history": self.price_history.copy(),
            "agent_valuations": np.array(self.agents)
        }

        print(f"\n=== OBSERVATION SENT TO PPO ===")
        print(f"Remaining Agents Mask: {obs['remaining_agents']}")
        print(f"Remaining Items Mask: {obs['remaining_items']}")
        print(f"Allocated Items:\n{obs['allocated_items']}")
        print(f"Price History:\n{obs['price_history']}")
        print(f"Agent Valuations: {obs['agent_valuations']}")
        
        return obs



    def step(self, action):
    # Print the incoming action
        print(f"\n=== PPO ACTION RECEIVED ===")
        print(f"Action (price index): {action}")

        price = action * (self.config["max_price"] / (self.config["pricing_levels"] - 1))
        print(f"Converted price: {price:.2f}")

        # Assuming you select agent in order (no randomness)
        agent_idx = self.remaining_agents[0]
        print(f"Selected Agent ID: {agent_idx}")
        print(f"Agent Valuation: {self.agents[agent_idx]}")

        accepted = self.agents[agent_idx] >= price
        print(f"Agent Accepted? {accepted}")

    # Apply the environment logic...


        reward = 0
        if accepted:
            if self.objective == "welfare":
                reward = valuation - price
                
            elif self.objective == "revenue":
                reward = price
        
            self.remaining_items.pop(0)  # Item sold
            self.allocated_items[agent_idx].append(0)
            self.price_history[agent_idx, 0] = price

        self.remaining_agents.remove(agent_idx)

        done = len(self.remaining_items) == 0 or len(self.remaining_agents) == 0

        print(f"Agent {agent_idx} offered price: ${price:.2f} | Valuation: ${valuation:.2f} | {'Accepted' if accepted else 'Rejected'}")

        return self._get_observation(), reward, done, {}

    def render(self, mode="human"):
        print(f"Agent valuations: {self.agents}")
        print(f"Remaining Agents: {self.remaining_agents}")
        print(f"Remaining Items: {self.remaining_items}")
        print(f"Current agent index: {self.current_agent_idx}")

    def seed(self, seed=None):
        np.random.seed(seed)

