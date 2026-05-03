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

# Independent agents in same environment

class Vehicle:
    __slots__=('vid','spawn_time','wait_time') # Makes simulation faster by avoiding dynamic attribute creation (dictionary) and using less memory per vehicle, since we expect many vehicles in the system at once. By defining __slots__, we tell Python to only allocate space for these three attributes and not use a dynamic dict for each instance.
    def __init__(self,vid,spawn_time):
        self.vid=vid 
        self.spawn_time=spawn_time # Can be useful for metrics like total trip time, but not strictly necessary for the environment dynamics
        self.wait_time=0 # Every step the vehicle spends waiting in a queue, it accrues one second of wait time. This is important for reward calculation and performance metrics, as it captures the delay experienced by vehicles in the system.


class Intersection:

    def __init__(self, iid, position):
        self.id = iid
        self.position = position  # (row, col)
        self.queues = {d: deque() for d in APPROACHES} # O(1) complexity of deque
        self.phase = 0
        self.phase_duration = 0
        self.yellow_remaining = 0  # >0 means currently in all-red transition
        self.pending_phase = 0     # phase to switch to after yellow finishes

    def queue_lengths(self):
        return {d: len(self.queues[d]) for d in APPROACHES} # Find the number of cars waiting in the each approach for the given intersection.

    def total_queue(self):
        return sum(len(q) for q in self.queues.values()) # all of the cars waiting in total: for this intersection self get the total sum of vehicles in its each queue q.

    def get_state(self):
        """Return discretized state tuple for Q-learning.""" # used dsicretization to reduce the state space for Q-learning, since the number of vehicles in each queue can grow indefinitely, which would make the state space too large to learn effectively. By binning the queue lengths into categories (0, 1-3, 4-7, 8+), we can capture the general level of congestion without needing to track exact counts.
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
                self.phase) # Get queue lengths for each approach, bin them, and include the current phase in the state representation.

    def can_switch(self):
        if self.yellow_remaining > 0:
            return False
        return self.phase_duration >= MIN_PHASE # ensure it is allowed to switch

    def apply_action(self, action):
        if action == 1 and self.can_switch(): # action-keep (0) or switch for 1
            self.pending_phase = 1 - self.phase #simply stores the next phase in the pending_phase variable like a flag if 1 0 if 0 1
            self.yellow_remaining = YELLOW_STEPS

    def tick_signal(self):
        """Advance internal signal timing by one step (called every step)."""
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0:
                self.phase = self.pending_phase # Switch to the new phase that was queued up
                self.phase_duration = 0 #restart the light timer
        else:
            self.phase_duration += 1

    def is_discharging(self): # can cars leave the queue?
        """Queues are discharging only when not in yellow transition."""
        return self.yellow_remaining == 0


class TrafficGridEnv:

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

    def reset(self, seed=None): # New episode, clean slate, no cars, fresh traffic lights
        if seed is not None: # If there is no seed it make it random every time, if there is a seed it will be the same every time, which is useful for debugging and reproducibility.
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

    # MOVEMENT HElPERS

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

    # MAIN STEP

    def step(self, actions):
        # 1. apply actions (may start yellow transitions).
        for iid, a in actions.items(): # what does agent want to do at each intersection? Keep current phase (0) or switch (1)
            self.intersections[iid].apply_action(a)

        # 2. external arrivals at boundary approaches.
        for iid, d in self._boundary_entry_approaches():
            n_arrivals = self.rng.poisson(self.arrival_rate)
            for _ in range(n_arrivals):
                v = Vehicle(self.next_vid, self.t)
                self.next_vid += 1
                self.intersections[iid].queues[d].append(v)

        # 3. Discharge vehicles from green approaches.
        # Collect hops first, then apply changes in the queues, so a vehicle can't traverse two
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
                self.completed_trips += 1 #number of vehicles that leaves the traffic
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

        self.t += 1 # Move simulation forward one step
        done = self.t >= self.episode_length # provided in training script
        obs = self.get_observations()
        info = {'step_total_queue': step_total_queue}
        return obs, rewards, done, info

    def get_observations(self):
        return {iid: inter.get_state()
                for iid, inter in self.intersections.items()}

    # METRICS

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
