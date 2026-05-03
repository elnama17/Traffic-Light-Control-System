"""Utility functions for training and evaluation."""

import os
import pickle
import json
from pathlib import Path


def save_metrics(metrics, output_path):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath):
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_agents(agents, output_dir):
    """Save all agents' Q-tables."""
    os.makedirs(output_dir, exist_ok=True)
    for iid, agent in agents.items():
        agent.save_q_table(os.path.join(output_dir, f"q_table_{iid[0]}_{iid[1]}.pkl"))


def load_agents(agents, checkpoint_dir):
    """Load agents' Q-tables from checkpoint."""
    for iid, agent in agents.items():
        q_path = os.path.join(checkpoint_dir, f"q_table_{iid[0]}_{iid[1]}.pkl")
        if os.path.exists(q_path):
            agent.load_q_table(q_path)


def compute_episode_stats(episode_data):
    """
    Compute statistics for an episode.
    
    Args:
        episode_data: Dict with keys 'rewards', 'wait_times', 'queue_lengths'
        
    Returns:
        Dict with average rewards, wait times, queue lengths
    """
    stats = {
        'avg_reward': sum(episode_data['rewards']) / len(episode_data['rewards']),
        'avg_queue': sum(episode_data['queue_lengths']) / len(episode_data['queue_lengths']),
    }
    return stats


def format_results(results):
    """Format results for display."""
    output = []
    for algo_name, metrics in results.items():
        output.append(f"\n{algo_name}:")
        output.append(f"  Avg Wait Time: {metrics['avg_wait']:.2f}s")
        output.append(f"  Avg Queue Length: {metrics['avg_queue']:.2f}")
        output.append(f"  Throughput: {metrics['throughput']} vehicles")
        output.append(f"  Total Vehicles: {metrics['total_vehicles']}")
    return "\n".join(output)
