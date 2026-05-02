import json


def plot_training_progress(history_file, output_file=None):
    # this function just plots graphs from training history

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return

    # load saved data from training
    with open(history_file, 'r') as f:
        history = json.load(f)

    episode_rewards = history['episode_rewards']
    episode_metrics = history['episode_metrics']

    # create 4 graphs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Q-Learning Training Progress', fontsize=14)

    # rewards over time (should improve)
    ax = axes[0, 0]
    ax.plot(episode_rewards, label='Cumulative Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Rewards Over Training')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # waiting time (lower is better)
    ax = axes[0, 1]
    avg_waits = [m['avg_wait'] for m in episode_metrics]
    ax.plot(avg_waits, label='Avg Wait Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wait Time (seconds)')
    ax.set_title('Average Waiting Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # queue length (lower is better)
    ax = axes[1, 0]
    avg_queues = [m['avg_queue'] for m in episode_metrics]
    ax.plot(avg_queues, label='Avg Queue Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Queue Length (vehicles)')
    ax.set_title('Average Queue Length')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # throughput (higher is better)
    ax = axes[1, 1]
    throughputs = [m['throughput'] for m in episode_metrics]
    ax.plot(throughputs, label='Throughput')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Vehicles Exited')
    ax.set_title('Throughput Over Training')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # save or show plot
    if output_file:
        plt.savefig(output_file, dpi=100)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def analyze_training_history(history_file):
    # this prints summary of training performance

    with open(history_file, 'r') as f:
        history = json.load(f)

    episode_rewards = history['episode_rewards']
    episode_metrics = history['episode_metrics']
    config = history.get('config', {})

    print("\n" + "=" * 60)
    print("TRAINING ANALYSIS")
    print("=" * 60)

    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # calculate total steps
    episode_length = config.get('episode_length', 0)
    total_steps = episode_length * len(episode_rewards)

    print(f"\nTotal Episodes: {len(episode_rewards)}")
    print(f"Total Training Steps: {total_steps:,}")

    # compare start vs end of training
    comparison_window = min(10, len(episode_metrics))

    early_episodes = episode_metrics[:comparison_window]
    late_episodes = episode_metrics[-comparison_window:]

    # average wait time
    early_avg_wait = sum(m['avg_wait'] for m in early_episodes) / len(early_episodes)
    late_avg_wait = sum(m['avg_wait'] for m in late_episodes) / len(late_episodes)

    # improvement %
    improvement = ((early_avg_wait - late_avg_wait) / early_avg_wait) * 100 if early_avg_wait else 0

    # average queue
    early_avg_queue = sum(m['avg_queue'] for m in early_episodes) / len(early_episodes)
    late_avg_queue = sum(m['avg_queue'] for m in late_episodes) / len(late_episodes)

    print("\nPerformance Comparison:")

    print(f"  Early episodes (first {comparison_window}):")
    print(f"    Avg wait time: {early_avg_wait:.2f}s")
    print(f"    Avg queue length: {early_avg_queue:.2f}")

    print(f"  Late episodes (last {comparison_window}):")
    print(f"    Avg wait time: {late_avg_wait:.2f}s")
    print(f"    Avg queue length: {late_avg_queue:.2f}")

    print(f"\n  Improvement: {improvement:+.1f}%")

    # show last episode result
    final_metrics = episode_metrics[-1]

    print("\nFinal Episode Metrics:")
    for key, val in final_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")


def compare_algorithms_table(results):
    # print table comparing all algorithms

    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON TABLE")
    print("=" * 80)

    metrics = ['avg_wait', 'avg_queue', 'throughput', 'total_vehicles']

    # header row
    print(f"{'Algorithm':<20}", end='')
    for metric in metrics:
        print(f"{metric:<15}", end='')
    print()

    print("-" * 80)

    # each algorithm row
    for algo_name, algo_metrics in results.items():
        print(f"{algo_name:<20}", end='')

        for metric in metrics:
            val = algo_metrics.get(metric, 0)

            if isinstance(val, float):
                print(f"{val:<15.2f}", end='')
            else:
                print(f"{val:<15}", end='')

        print()

    # compare vs fixed-time
    if 'Fixed-Time' in results and len(results) > 1:
        print("\n" + "-" * 80)
        print("IMPROVEMENTS OVER FIXED-TIME BASELINE")
        print("-" * 80)

        fixed_baseline = results['Fixed-Time']

        for algo_name, algo_metrics in results.items():
            if algo_name == 'Fixed-Time':
                continue

            wait_improvement = (
                (fixed_baseline['avg_wait'] - algo_metrics['avg_wait']) /
                fixed_baseline['avg_wait'] * 100
            ) if fixed_baseline['avg_wait'] else 0

            queue_improvement = (
                (fixed_baseline['avg_queue'] - algo_metrics['avg_queue']) /
                fixed_baseline['avg_queue'] * 100
            ) if fixed_baseline['avg_queue'] else 0

            print(f"\n{algo_name}:")
            print(f"  Wait time: {wait_improvement:+.1f}%")
            print(f"  Queue length: {queue_improvement:+.1f}%")


if __name__ == '__main__':
    import sys

    # simple CLI usage
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            history_file = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/training_history.json'
            plot_training_progress(history_file)

        elif sys.argv[1] == 'analyze':
            history_file = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/training_history.json'
            analyze_training_history(history_file)

    else:
        print("Usage:")
        print("  python analyze.py plot [history_file]")
        print("  python analyze.py analyze [history_file]")