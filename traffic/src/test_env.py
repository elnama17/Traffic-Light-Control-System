"""
Unit tests for the traffic simulation environment (env.py).

Tests cover:
- Environment initialization
- Vehicle creation and movement
- Signal phase transitions and timing
- Queue discharge mechanics
- Reward calculation
- Metrics aggregation
"""

import numpy as np
from env import (
    TrafficGridEnv, Intersection, Vehicle, Intersection,
    GRID_ROWS, GRID_COLS, APPROACHES, PHASE_APPROACHES,
    MIN_PHASE, YELLOW_STEPS
)


class TestVehicle:
    """Test Vehicle class."""

    def test_vehicle_creation(self):
        v = Vehicle(vid=0, spawn_time=10)
        assert v.vid == 0
        assert v.spawn_time == 10
        assert v.wait_time == 0

    def test_vehicle_wait_time_accumulation(self):
        v = Vehicle(vid=1, spawn_time=5)
        v.wait_time += 1
        assert v.wait_time == 1
        v.wait_time += 5
        assert v.wait_time == 6


class TestIntersection:
    """Test Intersection class."""

    def test_intersection_initialization(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        assert inter.id == (0, 0)
        assert inter.position == (0, 0)
        assert inter.phase == 0
        assert inter.phase_duration == 0
        assert inter.yellow_remaining == 0
        assert len(inter.queues) == 4

    def test_queue_lengths(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Initially all empty
        lengths = inter.queue_lengths()
        assert all(l == 0 for l in lengths.values())
        
        # Add vehicles
        inter.queues['N'].append(Vehicle(0, 0))
        inter.queues['N'].append(Vehicle(1, 0))
        inter.queues['S'].append(Vehicle(2, 0))
        
        lengths = inter.queue_lengths()
        assert lengths['N'] == 2
        assert lengths['S'] == 1
        assert lengths['E'] == 0
        assert lengths['W'] == 0

    def test_total_queue(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        assert inter.total_queue() == 0
        
        inter.queues['N'].append(Vehicle(0, 0))
        inter.queues['S'].append(Vehicle(1, 0))
        inter.queues['E'].append(Vehicle(2, 0))
        
        assert inter.total_queue() == 3

    def test_get_state_discretization(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Empty queues + phase 0
        state = inter.get_state()
        assert state == (0, 0, 0, 0, 0)
        
        # Add 2 vehicles to N (should bin to 1)
        inter.queues['N'].append(Vehicle(0, 0))
        inter.queues['N'].append(Vehicle(1, 0))
        state = inter.get_state()
        assert state[0] == 1  # N bins to 1 (1-3 range)
        
        # Add 5 more to E (should bin to 2, 5 is in 4-7 range)
        for i in range(5):
            inter.queues['E'].append(Vehicle(10+i, 0))
        state = inter.get_state()
        assert state[2] == 2  # E bins to 2 (4-7 range)
        
        # Add 10 to W (should bin to 3, >7)
        for i in range(10):
            inter.queues['W'].append(Vehicle(20+i, 0))
        state = inter.get_state()
        assert state[3] == 3  # W bins to 3 (>7)

    def test_can_switch_logic(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Initially at phase 0 with duration 0 -> cannot switch
        assert not inter.can_switch()
        
        # Advance phase_duration to MIN_PHASE
        inter.phase_duration = MIN_PHASE
        assert inter.can_switch()
        
        # Start yellow transition -> cannot switch
        inter.yellow_remaining = 1
        assert not inter.can_switch()

    def test_apply_action_keep_phase(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Action 0 = keep phase (should not trigger switch)
        inter.phase_duration = MIN_PHASE
        inter.apply_action(0)
        assert inter.yellow_remaining == 0
        assert inter.phase == 0

    def test_apply_action_switch_phase(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Action 1 = switch, but only if can_switch()
        inter.phase_duration = MIN_PHASE
        inter.apply_action(1)
        assert inter.yellow_remaining == YELLOW_STEPS
        assert inter.pending_phase == 1  # toggled from 0

    def test_apply_action_switch_blocked_by_yellow(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        inter.yellow_remaining = 1
        inter.phase_duration = MIN_PHASE
        inter.apply_action(1)
        # Should not switch
        assert inter.yellow_remaining == 1  # unchanged

    def test_tick_signal_phase_duration_increment(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        inter.phase = 0
        inter.phase_duration = 0
        inter.yellow_remaining = 0
        
        inter.tick_signal()
        assert inter.phase_duration == 1
        assert inter.phase == 0

    def test_tick_signal_yellow_countdown(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        inter.phase = 0
        inter.yellow_remaining = YELLOW_STEPS
        inter.pending_phase = 1
        
        # Tick through yellow period
        for i in range(YELLOW_STEPS):
            inter.tick_signal()
            if i < YELLOW_STEPS - 1:
                assert inter.yellow_remaining == YELLOW_STEPS - i - 1
                assert inter.phase == 0  # still old phase
            else:
                assert inter.yellow_remaining == 0
                assert inter.phase == 1  # switched to pending phase

    def test_is_discharging(self):
        inter = Intersection(iid=(0, 0), position=(0, 0))
        # Not discharging during yellow
        inter.yellow_remaining = 1
        assert not inter.is_discharging()
        
        # Discharging when not in yellow
        inter.yellow_remaining = 0
        assert inter.is_discharging()


class TestTrafficGridEnv:
    """Test TrafficGridEnv class."""

    def test_env_initialization(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.1, episode_length=100, seed=42)
        assert env.rows == 2
        assert env.cols == 2
        assert len(env.intersections) == 4
        assert env.t == 0
        assert env.next_vid == 0

    def test_reset_clears_state(self):
        env = TrafficGridEnv(rows=2, cols=2, seed=42)
        # Manually add a vehicle to an intersection
        env.intersections[(0, 0)].queues['N'].append(Vehicle(0, 0))
        env.t = 50
        env.next_vid = 10
        
        obs = env.reset(seed=42)
        assert env.t == 0
        assert env.next_vid == 0
        assert env.intersections[(0, 0)].total_queue() == 0
        assert all(inter.phase == 0 for inter in env.intersections.values())

    def test_get_observations(self):
        env = TrafficGridEnv(rows=2, cols=2, seed=42)
        obs = env.get_observations()
        assert len(obs) == 4
        for iid in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            assert iid in obs
            assert isinstance(obs[iid], tuple)
            assert len(obs[iid]) == 5  # 4 queue bins + phase

    def test_boundary_entry_approaches(self):
        env = TrafficGridEnv(rows=2, cols=2)
        entries = list(env._boundary_entry_approaches())
        
        # Should have: 2 cols (N) + 2 cols (S) + 2 rows (W) + 2 rows (E) = 8
        assert len(entries) == 8
        
        # Check specific boundary entries exist
        assert ((0, 0), 'N') in entries  # top-left, entering from north
        assert ((1, 1), 'S') in entries  # bottom-right, entering from south
        assert ((0, 0), 'W') in entries  # top-left, entering from west
        assert ((1, 1), 'E') in entries  # bottom-right, entering from east

    def test_next_hop_movement(self):
        env = TrafficGridEnv(rows=2, cols=2)
        
        # Vehicle heading south from (0,0) -> goes to (1,0) as 'N'
        nxt = env._next_hop((0, 0), 'N')
        assert nxt == ((1, 0), 'N')
        
        # Vehicle heading south from (1,0) -> exits (row 2 out of bounds)
        nxt = env._next_hop((1, 0), 'N')
        assert nxt is None
        
        # Vehicle heading east from (0,0) -> goes to (0,1) as 'W'
        nxt = env._next_hop((0, 0), 'W')
        assert nxt == ((0, 1), 'W')
        
        # Vehicle heading east from (0,1) -> exits (col 2 out of bounds)
        nxt = env._next_hop((0, 1), 'W')
        assert nxt is None

    def test_step_basic_flow(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        # Manually inject a vehicle at (0,0) in N queue
        v = Vehicle(0, 0)
        env.intersections[(0, 0)].queues['N'].append(v)
        
        # Execute step with action 0 (keep phase) on all intersections
        actions = {iid: 0 for iid in env.intersections.keys()}
        obs, rewards, done, info = env.step(actions)
        
        # Vehicle should still be in queue (needs green N/S phase and discharge)
        # At t=0 in phase 0, vehicle should discharge
        assert env.t == 1
        assert isinstance(rewards, dict)
        assert len(rewards) == 4

    def test_step_vehicle_discharge_and_progression(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        # Phase 0 is NS green; inject vehicle in N queue at (0,0)
        v = Vehicle(0, 0)
        env.intersections[(0, 0)].queues['N'].append(v)
        
        initial_queue = env.intersections[(0, 0)].total_queue()
        assert initial_queue == 1
        
        # Step with keep actions
        actions = {iid: 0 for iid in env.intersections.keys()}
        obs, rewards, done, info = env.step(actions)
        
        # Vehicle should have discharged and moved to (1,0) N queue
        assert env.intersections[(0, 0)].total_queue() == 0
        assert len(env.intersections[(1, 0)].queues['N']) == 1

    def test_step_rewards_based_on_queue(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        # Add 3 vehicles to (0,0) N queue
        for i in range(3):
            env.intersections[(0, 0)].queues['N'].append(Vehicle(i, 0))
        
        actions = {iid: 0 for iid in env.intersections.keys()}
        obs, rewards, done, info = env.step(actions)
        
        # Reward should be negative of queue size
        # Queue is 2 (one discharged) = reward -2
        assert rewards[(0, 0)] == -2.0

    def test_step_wait_time_accumulation(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        v = Vehicle(0, 0)
        env.intersections[(0, 0)].queues['N'].append(v)
        
        # Step 1: vehicle discharges to next intersection
        actions = {iid: 0 for iid in env.intersections.keys()}
        env.step(actions)
        
        # Check that vehicle accumulated wait time = 1
        v_next = env.intersections[(1, 0)].queues['N'][0]
        assert v_next.wait_time == 1

    def test_step_episode_termination(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=10, seed=42)
        env.reset(seed=42)
        
        actions = {iid: 0 for iid in env.intersections.keys()}
        
        done = False
        for step_count in range(10):
            obs, rewards, done, info = env.step(actions)
            if step_count < 9:
                assert not done
            else:
                assert done

    def test_metrics_empty_network(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=10, seed=42)
        env.reset(seed=42)
        
        actions = {iid: 0 for iid in env.intersections.keys()}
        for _ in range(10):
            env.step(actions)
        
        metrics = env.metrics()
        assert metrics['throughput'] == 0
        assert metrics['avg_wait'] == 0.0
        assert metrics['avg_queue'] == 0.0
        assert metrics['total_vehicles'] == 0

    def test_metrics_with_completed_trips(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        # Inject a vehicle that will exit the network
        env.intersections[(1, 0)].queues['N'].append(Vehicle(0, 0))
        
        actions = {iid: 0 for iid in env.intersections.keys()}
        
        # Step until vehicle exits
        for _ in range(10):
            env.step(actions)
        
        metrics = env.metrics()
        assert metrics['throughput'] >= 0
        assert metrics['avg_wait'] >= 0
        assert metrics['total_vehicles'] >= 0

    def test_phase_switching_timing(self):
        env = TrafficGridEnv(rows=2, cols=2, arrival_rate=0.0, episode_length=100, seed=42)
        env.reset(seed=42)
        
        inter = env.intersections[(0, 0)]
        initial_phase = inter.phase
        
        # Keep current phase for MIN_PHASE steps
        actions = {iid: 0 for iid in env.intersections.keys()}
        for _ in range(MIN_PHASE):
            env.step(actions)
        
        assert inter.phase == initial_phase
        assert inter.phase_duration == MIN_PHASE
        
        # Now request phase switch
        actions[(0, 0)] = 1
        env.step(actions)
        
        # Should be in yellow transition
        assert inter.yellow_remaining > 0
        
        # Wait for yellow to clear
        actions[(0, 0)] = 0
        for _ in range(YELLOW_STEPS):
            env.step(actions)
        
        # Phase should have switched
        assert inter.phase != initial_phase


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [TestVehicle, TestIntersection, TestTrafficGridEnv]
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                passed_tests += 1
                print(f"✓ {test_class.__name__}.{test_method}")
            except Exception as e:
                failed_tests.append((test_class.__name__, test_method, e))
                print(f"✗ {test_class.__name__}.{test_method}")
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_all_tests()
