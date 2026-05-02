"Training script for Q-learning agent."

import sys
import os
import json
from pathlib import Path
# Here I add the project root to the path so this file can import project modules correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from traffic_rl.src.env import TrafficGridEnv
from traffic_rl.src.agents import QLearningAgent
from traffic_rl.src.utils import save_agents, save_metrics

def train_q_learning(num_episodes=100, episode_length=3600, arrival_rate=0.1,
                    learning_rate=0.1, epsilon_decay=0.995, checkpoint_dir='checkpoints'):
    "This function trains Q-learning agents in the traffic simulation environment."
    # First I create the traffic environment.
    # It is a 2x2 grid, so there are four intersections in total.
    # The seed makes the simulation results reproducible.
    env = TrafficGridEnv(rows=2, cols=2, arrival_rate=arrival_rate,
                        episode_length=episode_length, seed=42)
    # Here I create one Q-learning agent for each intersection.
    # Each agent controls only its own traffic light.
    agents = {}
    for iid in env.intersections.keys():
        agents[iid] = QLearningAgent(iid, learning_rate=learning_rate)
    # These lists store the training progress.
    # episode_rewards keeps total reward for each episode.
    # episode_metrics keeps values like average wait time and queue length.
    episode_rewards = []
    episode_metrics = []
    print(f"Starting training for {num_episodes} episodes...")
    print(f"  Arrival rate: {arrival_rate}")
    print(f"  Episode length: {episode_length}s")
    print(f"  Learning rate: {learning_rate}\n")

    for episode in range(num_episodes):
        # At the start of every episode, I reset the traffic environment.
        obs = env.reset()
        total_reward = 0

        for step in range(episode_length):
            # In this part, every agent looks at its current state and selects an action.
            # Action 0 means keep the current phase, and action 1 means switch phase.
            actions = {}
            for iid, agent in agents.items():
                actions[iid] = agent.select_action(obs[iid], training=True)
            # Here the environment applies all selected actions.
            # It returns the next states, rewards, and whether the episode is finished.
            obs_next, rewards, done, info = env.step(actions)
            # After the action result is known, each agent updates its Q-table.
            # This is where learning actually happens.
            for iid, agent in agents.items():
                agent.update(obs[iid], actions[iid], rewards[iid], obs_next[iid], done)
            # I add all intersection rewards to measure total performance in this episode.
            total_reward += sum(rewards.values())
            # The next state becomes the current state for the next step.
            obs = obs_next
            if done:
                break
        # After each episode, epsilon is reduced.
        # This means the agents explore less and use their learned knowledge more.
        for agent in agents.values():
            agent.decay_epsilon(epsilon_decay)
        # Here I collect the final metrics of this episode.
        metrics = env.metrics()
        episode_rewards.append(total_reward)
        episode_metrics.append(metrics)
        # Every 10 episodes, I print progress to see whether learning is improving.
        if (episode + 1) % 10 == 0:
            avg_wait = sum(m['avg_wait'] for m in episode_metrics[-10:]) / 10
            avg_queue = sum(m['avg_queue'] for m in episode_metrics[-10:]) / 10
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg reward: {total_reward/episode_length:.3f}")
            print(f"  Avg wait time (last 10): {avg_wait:.2f}s")
            print(f"  Avg queue length (last 10): {avg_queue:.2f}")
            print(f"  Exploration rate: {agents[(0,0)].epsilon:.4f}")
            print()
    # Here I save the trained agents.
    # Their Q-tables can be reused later during evaluation.
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_agents(agents, checkpoint_dir)
    # I also save the training history so results can be used for reports or plots.
    history = {
        'episode_rewards': episode_rewards,
        'episode_metrics': episode_metrics,
        'config': {
            'num_episodes': num_episodes,
            'episode_length': episode_length,
            'arrival_rate': arrival_rate,
            'learning_rate': learning_rate,
            'epsilon_decay': epsilon_decay
        }
    }
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! Checkpoints saved to {checkpoint_dir}")
    print(f"Final average wait time: {episode_metrics[-1]['avg_wait']:.2f}s")
    print(f"Final average queue length: {episode_metrics[-1]['avg_queue']:.2f}")
    return agents, history
if __name__ == '__main__':
    train_q_learning(num_episodes=200, episode_length=3600, arrival_rate=0.15)
