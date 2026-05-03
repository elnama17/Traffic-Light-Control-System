from collections import deque
import numpy as np


GRID_ROWS = 2
GRID_COLS = 2

# Approach directions which are North, South, East, West
APPROACHES = ('N', 'S', 'E', 'W')

# Which approaches are allowed to discharge in each phase
PHASE_APPROACHES = {
    0: ('N', 'S'),  # North South green
    1: ('E', 'W'),  # East West green
}

MIN_PHASE = 5       # minimum green duration before switching
YELLOW_STEPS = 2    # all-red transition length


class Vehicle:
    __slots__=('vid','spawn_time','wait_time') # Makes simulation faster by avoiding dynamic attribute creation per vehicle
    def __init__(self,vid,spawn_time):
        self.vid=vid # vehicle id
        self.spawn_time=spawn_time # when vehicle entered the network
        self.wait_time=0 # waiting time of the vehicle, important for performance metrics and reward calculation.

class Intersection:

    def __init__(self, iid, position):
        self.id = iid
        self.position = position  # (row, col)
        self.queues = {d: deque() for d in APPROACHES} # We use deque because it has O(1) complexity when adding or removing cars
        self.phase = 0
        self.phase_duration = 0
        self.yellow_remaining = 0  # if this variable is greater than zero, it means intersection iscurrently in all-red transition
        self.pending_phase = 0     # phase to switch to after yellow finishes

    def queue_lengths(self):
        return {d: len(self.queues[d]) for d in APPROACHES} # This finds the number of cars waiting in the each approach for the given intersection.

    def total_queue(self):
        return sum(len(q) for q in self.queues.values()) # all of the cars waiting in total: for this intersection self get the total sum of vehicles in its each queue q.

    def get_state(self):
        """Returns the current traffic state in a simplified - discretized form for state space reduction""" 
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
                self.phase) # we use discretization to reduce the state space size

    def can_switch(self):
        """ensure Traffic Light is allowed to switch""" 
        if self.yellow_remaining > 0:
            return False
        return self.phase_duration >= MIN_PHASE

    def apply_action(self, action):
        if action == 1 and self.can_switch(): # action-keep (0) or switch for 1
            self.pending_phase = 1 - self.phase #simply stores the next phase in the pending_phase variable like a flag if 1 0 if 0 1
            self.yellow_remaining = YELLOW_STEPS

    def tick_signal(self):
        """Increase internal signal timing by one step since it's called once per step"""
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0:
                self.phase = self.pending_phase # Switch to the new phase that was queued up
                self.phase_duration = 0 # Restart the light timer
        else:
            self.phase_duration += 1

    def is_discharging(self):
        """Answers can cars leave the queue? Queues are discharging only when not in yellow transition."""
        return self.yellow_remaining == 0


class TrafficGridEnv:
    """
    This class represents 2x2 grid of the traffic network. In one step the agent of the intersection observes local
    queue state and choose a phase action.
    """

    def __init__(self, rows=GRID_ROWS, cols=GRID_COLS,
                 arrival_rate=0.1, episode_length=3600, seed=None):
        self.rows = rows # rows and columns of the grid
        self.cols = cols
        self.arrival_rate = arrival_rate  # Poisson lambda per boundary lane in each step
        self.episode_length = episode_length # number of steps in one episode
        self.rng = np.random.default_rng(seed) # Fixed seed is used which means same sequence of numbers will be generated

        self.intersections = {} # storing all intersections with their id as key and Intersection object as value
        for r in range(rows):
            for c in range(cols):
                iid = (r, c)
                self.intersections[iid] = Intersection(iid, iid)

        self.reset(seed=seed) # calling reset to ensure fresh start

    def reset(self, seed=None): # This function resets the episode, New episode, clean state, no cars, fresh traffic lights
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

    # MOVEMENT HElPERS

    def _boundary_entry_approaches(self): 
        """ This function identifies points at which external vehicles can enter the grid network """
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
        # Firstly, apply actions (may start yellow transitions).
        for iid, a in actions.items(): # what does agent want to do at each intersection? Keep current phase (0) or switch (1)
            self.intersections[iid].apply_action(a)

        # Second, external arrivals at boundary approaches.
        for iid, d in self._boundary_entry_approaches():
            n_arrivals = self.rng.poisson(self.arrival_rate)
            for _ in range(n_arrivals):
                v = Vehicle(self.next_vid, self.t)
                self.next_vid += 1
                self.intersections[iid].queues[d].append(v)

        # Then discharge vehicles from green approaches.
        # Firstly, we collect hops, then apply changes in the queues, so a vehicle can't traverse two
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
                self.completed_trips += 1 # number of vehicles that leaves
                self.total_wait_completed += v.wait_time
            else:
                next_iid, next_dir = nxt
                self.intersections[next_iid].queues[next_dir].append(v)

        # Every remaining queued vehicle's waiting time calculated by this:
        step_total_queue = 0
        for inter in self.intersections.values():
            for d in APPROACHES:
                for v in inter.queues[d]:
                    v.wait_time += 1
                step_total_queue += len(inter.queues[d])
        self.cumulative_queue += step_total_queue
        # We compute rewards for each agent which is negative of number of vehicles in one queue:
        rewards = {}
        for iid, inter in self.intersections.items():
            rewards[iid] = -float(inter.total_queue())

        # Then we trigger signal
        for inter in self.intersections.values():
            inter.tick_signal()

        self.t += 1 # moving simulation forward one step
        done = self.t >= self.episode_length # provided in training script and also config file
        obs = self.get_observations()
        info = {'step_total_queue': step_total_queue}
        return obs, rewards, done, info
    
    def get_observations(self): # at last we collect the current state of every intersection and return it to the agents.
        return {iid: inter.get_state()
                for iid, inter in self.intersections.items()}

    # METRICS

    def metrics(self):
        """This function computes final performance statistics for one episode which means one full simulation"""
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
