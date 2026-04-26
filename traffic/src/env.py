"""
Traffic simulation environment: 2x2 grid of signalized intersections.

Simulation model
----------------
- Discrete time steps (1 step = 1 second).
- Each intersection has 4 approach queues: N, S, E, W. An 'N' queue holds
  vehicles coming from the north (heading south) and waiting to pass through.
- Two signal phases per intersection:
    phase 0 -> NS green (N and S queues can discharge)
    phase 1 -> EW green (E and W queues can discharge)
- Switching phase triggers a 2-step all-red (yellow) transition during which
  no queue discharges.
- Minimum green time of MIN_PHASE steps before switching is allowed.
- Saturation flow = 1 vehicle / step per green approach.
- Vehicles travel straight through the grid (no turns) for tractability.
- External Poisson arrivals at the grid boundary; after passing through an
  intersection, a vehicle either joins the next intersection's inbound queue
  or exits the network (if it has reached the opposite boundary).
"""

from collections import deque
import numpy as np


GRID_ROWS = 2
GRID_COLS = 2

# Approach directions
APPROACHES = ('N', 'S', 'E', 'W')

# Which approaches are allowed to discharge in each phase
PHASE_APPROACHES = {
    0: ('N', 'S'),  # NS green
    1: ('E', 'W'),  # EW green
}

MIN_PHASE = 5       # minimum green duration before switching
YELLOW_STEPS = 2    # all-red transition length


class Vehicle:
    __slots__=('vid','spawn_time','wait_time')
    def __init__(self,vid,spawn_time):
        self.vid=vid 
        self.spawn_time=spawn_time
        self.wait_time=0


class Intersection:
    """A single signalized intersection with 4 approach queues."""

    def __init__(self, iid, position):
        self.id = iid
        self.position = position  # (row, col)
        self.queues = {d: deque() for d in APPROACHES}
        self.phase = 0
        self.phase_duration = 0
        self.yellow_remaining = 0  # >0 means currently in all-red transition
        self.pending_phase = 0     # phase to switch to after yellow finishes

    def queue_lengths(self):
        return {d: len(self.queues[d]) for d in APPROACHES}

    def total_queue(self):
        return sum(len(q) for q in self.queues.values()) # all of the cars waiting in total

    def get_state(self):
        """Return discretized state tuple for Q-learning."""
        def bin_q(n):
            if n == 0:
                return 0
            elif n <= 3:
                return 1
            elif n <= 7:
                return 2
            else:
                return 3
        q = self.queue_lengths()
        return (bin_q(q['N']), bin_q(q['S']), bin_q(q['E']), bin_q(q['W']),
                self.phase)

    def can_switch(self):
        if self.yellow_remaining > 0:
            return False
        return self.phase_duration >= MIN_PHASE

    def apply_action(self, action):
        if action == 1 and self.can_switch(): #action-keep or switch for 1
            self.pending_phase = 1 - self.phase #simply stores the next phase in the pending_phase variable like a flag if 1 0 if 0 1
            self.yellow_remaining = YELLOW_STEPS

    def tick_signal(self):
        """Advance internal signal timing by one step (called every step)."""
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0:
                self.phase = self.pending_phase #switch to the new phase that was queued up
                self.phase_duration = 0
        else:
            self.phase_duration += 1

    def is_discharging(self): # can cars leave the queue?
        """Queues are discharging only when not in yellow transition."""
        return self.yellow_remaining == 0


class TrafficGridEnv:
    """
    2D grid of intersections. Agents (one per intersection) observe local
    queue state and choose a phase action each step. The environment handles
    vehicle arrivals, discharge, movement between intersections, and metrics.
    """

    def __init__(self, rows=GRID_ROWS, cols=GRID_COLS,
                 arrival_rate=0.1, episode_length=3600, seed=None):
        self.rows = rows
        self.cols = cols
        self.arrival_rate = arrival_rate  # Poisson lambda per boundary lane per step
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed) #fixed seed, same sequence of numbers will be generated

        self.intersections = {}
        for r in range(rows):
            for c in range(cols):
                iid = (r, c)
                self.intersections[iid] = Intersection(iid, iid)

        self.reset(seed=seed) # calling reset to ensure fresh start

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for inter in self.intersections.values():
            for d in APPROACHES:
                inter.queues[d].clear()
            inter.phase = 0
            inter.phase_duration = 0
            inter.yellow_remaining = 0
            inter.pending_phase = 0
        self.t = 0
        self.next_vid = 0
        # Metric accumulators
        self.completed_trips = 0
        self.total_wait_completed = 0
        self.cumulative_queue = 0
        return self.get_observations()

    # ---- movement helpers -------------------------------------------------

    def _boundary_entry_approaches(self): # is designed to identify the points
    # at which external vehicles can enter the grid network
        """
        Yield (intersection_id, approach_dir) pairs where external vehicles
        may enter the network from the outside.
        """
        for c in range(self.cols):
            yield ((0, c), 'N')           # top edge, entering heading south
            yield ((self.rows - 1, c), 'S')  # bottom edge, entering heading north
        for r in range(self.rows):
            yield ((r, 0), 'W')           # left edge, entering heading east
            yield ((r, self.cols - 1), 'E')  # right edge, entering heading west

    def _next_hop(self, iid, approach):
        """
        Given a vehicle that just discharged from `approach` queue of
        intersection `iid`, return the (next_iid, next_approach) it should
        join next, or None if it exits the network.
        Vehicles go straight through. This is what makes the grid feel like a connected town:
        cars don't disappear when they leave one intersection — they become inputs to the next.
        """
        r, c = iid
        if approach == 'N':   # was heading south; next intersection below
            nr, nc = r + 1, c
            if nr >= self.rows:
                return None
            return ((nr, nc), 'N')
        if approach == 'S':   # heading north
            nr, nc = r - 1, c
            if nr < 0:
                return None
            return ((nr, nc), 'S')
        if approach == 'W':   # heading east
            nr, nc = r, c + 1
            if nc >= self.cols:
                return None
            return ((nr, nc), 'W')
        if approach == 'E':   # heading west
            nr, nc = r, c - 1
            if nc < 0:
                return None
            return ((nr, nc), 'E')
        return None

    # ---- main step --------------------------------------------------------

    def step(self, actions):
        """
        actions: dict {intersection_id: 0 or 1}
        Returns: observations, rewards (dict per intersection), done, info
        """
        # 1. Apply actions (may start yellow transitions).
        for iid, a in actions.items():
            self.intersections[iid].apply_action(a)

        # 2. External arrivals at boundary approaches.
        for iid, d in self._boundary_entry_approaches():
            n_arrivals = self.rng.poisson(self.arrival_rate)
            for _ in range(n_arrivals):
                v = Vehicle(self.next_vid, self.t)
                self.next_vid += 1
                self.intersections[iid].queues[d].append(v)

        # 3. Discharge vehicles from green approaches.
        # Collect hops first, then apply, so a vehicle can't traverse two
        # intersections in one step.
        hops = []
        for iid, inter in self.intersections.items():
            if not inter.is_discharging():
                continue
            green_dirs = PHASE_APPROACHES[inter.phase]
            for d in green_dirs:
                if inter.queues[d]:
                    v = inter.queues[d].popleft()
                    nxt = self._next_hop(iid, d)
                    hops.append((v, nxt))

        for v, nxt in hops:
            if nxt is None:
                # Vehicle exits network
                self.completed_trips += 1
                self.total_wait_completed += v.wait_time
            else:
                next_iid, next_dir = nxt
                self.intersections[next_iid].queues[next_dir].append(v)

        # 4. Every remaining queued vehicle accrues one second of wait time.
        step_total_queue = 0
        for inter in self.intersections.values():
            for d in APPROACHES:
                for v in inter.queues[d]:
                    v.wait_time += 1
                step_total_queue += len(inter.queues[d])
        self.cumulative_queue += step_total_queue

        # 5. Compute per-agent rewards (negative of local queued vehicles).
        rewards = {}
        for iid, inter in self.intersections.items():
            rewards[iid] = -float(inter.total_queue())

        # 6. Advance signal timing.
        for inter in self.intersections.values():
            inter.tick_signal()

        self.t += 1
        done = self.t >= self.episode_length
        obs = self.get_observations()
        info = {'step_total_queue': step_total_queue}
        return obs, rewards, done, info

    def get_observations(self):
        return {iid: inter.get_state()
                for iid, inter in self.intersections.items()}

    # ---- metrics ---------------------------------------------------------

    def metrics(self):
        """Final metrics over the episode."""
        # Include currently-queued vehicles in the wait total too, so the
        # metric isn't artificially good just because trips are slow to
        # complete in heavy traffic.
        total_wait = self.total_wait_completed
        queued_vehicles = 0
        for inter in self.intersections.values():
            for d in APPROACHES:
                for v in inter.queues[d]:
                    total_wait += v.wait_time
                    queued_vehicles += 1
        total_vehicles = self.completed_trips + queued_vehicles
        avg_wait = total_wait / max(total_vehicles, 1)
        avg_queue = self.cumulative_queue / max(self.t, 1)
        throughput = self.completed_trips  # vehicles that exited the network
        return {
            'avg_wait': avg_wait,
            'avg_queue': avg_queue,
            'throughput': throughput,
            'total_vehicles': total_vehicles,
        }
