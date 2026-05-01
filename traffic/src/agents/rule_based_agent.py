"Rule-based baseline agent for traffic light control."
class RuleBasedAgent:
    "This agent controls the traffic light using simple fixed rules."
    def __init__(self, intersection_id, switch_threshold=5, min_phase=5):
        self.intersection_id = intersection_id
        # switch_threshold shows after how much traffic the agent should think about switching.
        # In this simple version, the queue values are already grouped into bins.
        self.switch_threshold = switch_threshold
        # min_phase means the light should stay in one phase for at least this much time.
        # It is stored here, but this state does not include phase duration yet.
        self.min_phase = min_phase
    def select_action(self, state):
        " Here I decide whether to keep the current traffic light phase or switch it. The rule is simple: if the opposite direction has enough waiting cars,then the agent switches the light to that direction."
        n_q, s_q, e_q, w_q, current_phase = state
        # If the current green light is for North-South traffic,
        # I check whether East-West traffic is getting crowded.
        if current_phase == 0:
            max_ew_queue = max(e_q, w_q)
            # Queue bin 2 means around 4-7 vehicles, so it is worth switching.
            if max_ew_queue >= 2:
                return 1
        # If the current green light is for East-West traffic,
        # I check whether North-South traffic is getting crowded.
        else:
            max_ns_queue = max(n_q, s_q)
            # If North or South has enough waiting cars, I switch the phase.
            if max_ns_queue >= 2:
                return 1
        # If the opposite direction is not crowded enough, I keep the current phase.
        return 0
