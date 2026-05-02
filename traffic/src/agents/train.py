"""Training script for Q-learning agent."""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_rl.src.env import TrafficGridEnv
from traffic_rl.src.agents import QLearningAgent
from traffic_rl.src.utils import save_agents, save_metrics


def train_q_learning(num_episodes=100, episode_length=3600, arrival_rate=0.1,
                    learning_rate=0.1, epsilon_decay=0.995, checkpoint_dir='checkpoints'):
    """
    Train Q-learning agents for traffic control.
    
    Args:
        num_episodes: Number of training episodes
        episode_length: Steps per episode (seconds)
        arrival_rate: Vehicle arrival rate (Poisson lambda)
        learning_rate: Q-learning alpha parameter
        epsilon_decay: Decay rate for exploration
        checkpoint_dir: Directory to save checkpoints
    """
    
    # Create environment
    env = TrafficGridEnv(rows=2, cols=2, arrival_rate=arrival_rate, 
                        episode_length=episode_length, seed=42)
    
    # Create agents (one per intersection)
    agents = {}
    for iid in env.intersections.keys():
        agents[iid] = QLearningAgent(iid, learning_rate=learning_rate)
    
    # Training metrics
    episode_rewards = []
    episode_metrics = []
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"  Arrival rate: {arrival_rate}")
    print(f"  Episode length: {episode_length}s")
    print(f"  Learning rate: {learning_rate}\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        for step in range(episode_length):
            # Select actions for each agent
            actions = {}
            for iid, agent in agents.items():
                actions[iid] = agent.select_action(obs[iid], training=True)
            
            # Step environment
            obs_next, rewards, done, info = env.step(actions)
            
            # Update agents
            for iid, agent in agents.items():
                agent.update(obs[iid], actions[iid], rewards[iid], obs_next[iid], done)
            
            total_reward += sum(rewards.values())
            obs = obs_next
            
            if done:
                break
        
        # Decay exploration
        for agent in agents.values():
            agent.decay_epsilon(epsilon_decay)
        
        # Collect metrics
        metrics = env.metrics()
        episode_rewards.append(total_reward)
        episode_metrics.append(metrics)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_wait = sum(m['avg_wait'] for m in episode_metrics[-10:]) / 10
            avg_queue = sum(m['avg_queue'] for m in episode_metrics[-10:]) / 10
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg reward: {total_reward/episode_length:.3f}")
            print(f"  Avg wait time (last 10): {avg_wait:.2f}s")
            print(f"  Avg queue length (last 10): {avg_queue:.2f}")
            print(f"  Exploration rate: {agents[(0,0)].epsilon:.4f}")
            print()
    
    # Save checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_agents(agents, checkpoint_dir)
    
    # Save training history
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
