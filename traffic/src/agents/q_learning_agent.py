"Q-learning agent for traffic light control."
import numpy as np
from collections import defaultdict
class QLearningAgent:
    "This agent learns how to control one traffic light."
    def __init__(self, intersection_id, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.intersection_id = intersection_id
        # learning_rate shows how fast the agent updates its knowledge.
        # 0.1 means it learns slowly and does not change the Q-values too aggressively.
        self.alpha = learning_rate
        # discount_factor shows how much the agent cares about future rewards.
        # 0.99 means future rewards are very important, not only the current reward.
        self.gamma = discount_factor
        # epsilon controls exploration.
        # 0.1 means the agent chooses a random action 10% of the time while training.
        self.epsilon = epsilon
        # Here I create the Q-table. For every traffic state, the agent has two choices:
        # 0 means keep the same light phase, and 1 means switch to the other phase.
        self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.last_state = None
        self.last_action = None
    def select_action(self, state, training=True):
        "In this part, the agent decides what action to take. If it is training, sometimes it chooses randomly to explore. Otherwise, it chooses the action with the highest Q-value."
        if training and np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        q_vals = self.q_table[state]
        max_q = max(q_vals.values())
        # If both actions have the same Q-value, I randomly choose one of them.
        best_actions = [action for action, q in q_vals.items() if q == max_q]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        " Here I update the Q-table after the agent takes an action. The new value depends on the reward and the best possible future action."
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q

        current_q = self.q_table[state][action]
        # This formula moves the old Q-value closer to the better updated value.
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self, decay_rate=0.995):
        " After some training, I reduce epsilon. This means the agent explores less and starts using what it has learned."
        self.epsilon *= decay_rate
        self.epsilon = max(self.epsilon, 0.01)

    def get_q_table_size(self):
        "This shows how many states the agent has learned so far."
        return len(self.q_table)

    def save_q_table(self, filepath):
        "Here I save the Q-table so the learned data can be used later."
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filepath):
        "Here I load a saved Q-table instead of starting learning from zero."
        import pickle
        with open(filepath, "rb") as f:
            self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0}, pickle.load(f))
