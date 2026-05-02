import os
import pickle

from traffic.src.env import TrafficGridEnv
from traffic.src.agents.q_learning_agent import QLearningAgent
from traffic.src.agents.rule_based_agent import RuleBasedAgent


def format_results(results):
    # make result output readable
    lines = []

    for name, metrics in results.items():
        lines.append(f"\n{name}")
        lines.append(f"  Average wait time: {metrics['avg_wait']:.2f}s")
        lines.append(f"  Average queue length: {metrics['avg_queue']:.2f}")
        lines.append(f"  Throughput: {metrics['throughput']:.1f} vehicles")
        lines.append(f"  Total vehicles: {metrics['total_vehicles']:.1f}")

    return "\n".join(lines)


def load_agents(agents, checkpoint_dir):
    # load saved Q-tables
    for iid, agent in agents.items():
        filename = f"q_table_{iid[0]}_{iid[1]}.pkl"
        filepath = os.path.join(checkpoint_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing checkpoint file: {filepath}")

        with open(filepath, "rb") as f:
            agent.q_table.update(pickle.load(f))


def make_env(arrival_rate, episode_length):
    # create environment with fixed seed
    return TrafficGridEnv(
        rows=2,
        cols=2,
        arrival_rate=arrival_rate,
        episode_length=episode_length,
        seed=123
    )


class FixedTimeAgent:
    # baseline controller that switches every fixed number of steps

    def __init__(self, iid, switch_interval=10):
        self.iid = iid
        self.switch_interval = switch_interval
        self.step_count = 0

    def reset(self):
        # reset counter before each episode
        self.step_count = 0

    def select_action(self, state):
        # switch when interval is reached
        self.step_count += 1

        if self.step_count >= self.switch_interval:
            self.step_count = 0
            return 1

        return 0


def evaluate_agents(agents, agent_name, env, num_episodes=10):
    # store metric totals before averaging
    total_metrics = {
        "avg_wait": 0,
        "avg_queue": 0,
        "throughput": 0,
        "total_vehicles": 0
    }

    print(f"\nEvaluating {agent_name}")

    for _ in range(num_episodes):
        obs = env.reset()

        # reset agents that remember internal state
        for agent in agents.values():
            if hasattr(agent, "reset"):
                agent.reset()

        for _ in range(env.episode_length):
            actions = {}

            # choose action for each intersection
            for iid, agent in agents.items():
                if isinstance(agent, QLearningAgent):
                    actions[iid] = agent.select_action(obs[iid], training=False)
                else:
                    actions[iid] = agent.select_action(obs[iid])

            # move simulation forward
            obs_next, rewards, done, info = env.step(actions)
            obs = obs_next

            if done:
                break

        # collect metrics from this episode
        metrics = env.metrics()

        for key in total_metrics:
            total_metrics[key] += metrics[key]

    # average metrics over episodes
    for key in total_metrics:
        total_metrics[key] /= num_episodes

    print(f"  Average wait time: {total_metrics['avg_wait']:.2f}s")
    print(f"  Average queue length: {total_metrics['avg_queue']:.2f}")
    print(f"  Throughput: {total_metrics['throughput']:.0f} vehicles")

    return total_metrics


def compare_algorithms(
    arrival_rate=0.1,
    episode_length=3600,
    num_eval_episodes=10,
    checkpoint_dir="checkpoints"
):
    print(f"\nTraffic light control comparison")
    print(f"Arrival rate: {arrival_rate} vehicles/second")
    print(f"Episode length: {episode_length}s")
    print(f"Evaluation episodes: {num_eval_episodes}")

    results = {}

    # evaluate Q-learning controller
    env = make_env(arrival_rate, episode_length)
    q_agents = {}

    for iid in env.intersections.keys():
        q_agents[iid] = QLearningAgent(iid)

    try:
        load_agents(q_agents, checkpoint_dir)
        print(f"\nLoaded Q-learning agents from '{checkpoint_dir}'")
    except Exception as e:
        print(f"\nCould not load Q-learning checkpoints: {e}")
        print("Using untrained Q-learning agents")

    results["Q-Learning"] = evaluate_agents(
        q_agents,
        "Q-Learning",
        env,
        num_eval_episodes
    )

    # evaluate rule-based controller
    env = make_env(arrival_rate, episode_length)
    rule_agents = {}

    for iid in env.intersections.keys():
        rule_agents[iid] = RuleBasedAgent(iid)

    results["Rule-Based"] = evaluate_agents(
        rule_agents,
        "Rule-Based",
        env,
        num_eval_episodes
    )

    # evaluate fixed-time controller
    env = make_env(arrival_rate, episode_length)
    fixed_agents = {}

    for iid in env.intersections.keys():
        fixed_agents[iid] = FixedTimeAgent(iid)

    results["Fixed-Time"] = evaluate_agents(
        fixed_agents,
        "Fixed-Time",
        env,
        num_eval_episodes
    )

    print("\nResults summary")
    print(format_results(results))

    print("\nImprovement compared with fixed-time baseline")

    fixed_wait = results["Fixed-Time"]["avg_wait"]
    fixed_queue = results["Fixed-Time"]["avg_queue"]

    for algo_name in ["Q-Learning", "Rule-Based"]:
        if fixed_wait > 0:
            wait_improvement = (
                (fixed_wait - results[algo_name]["avg_wait"]) / fixed_wait
            ) * 100
        else:
            wait_improvement = 0.0

        if fixed_queue > 0:
            queue_improvement = (
                (fixed_queue - results[algo_name]["avg_queue"]) / fixed_queue
            ) * 100
        else:
            queue_improvement = 0.0

        print(f"\n{algo_name}")
        print(f"  Wait time improvement: {wait_improvement:+.1f}%")
        print(f"  Queue length improvement: {queue_improvement:+.1f}%")

    return results


if __name__ == "__main__":
    scenarios = [
        ("Light", 0.05),
        ("Medium", 0.10),
        ("Heavy", 0.15)
    ]

    for label, rate in scenarios:
        print(f"\n{label} traffic, arrival_rate={rate}")
        compare_algorithms(
            arrival_rate=rate,
            num_eval_episodes=10
        )