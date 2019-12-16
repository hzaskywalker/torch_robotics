PERIOD = {}
PERIOD_N = {}

class period:
    def __init__(self, func, times, name, goal=0):
        if name not in PERIOD:
            PERIOD[name] = -1
            PERIOD_N[name] = times
        else:
            assert times == PERIOD_N[name]
        if goal < 0:
            goal = times + goal

        self.name = name
        self.func = func
        self.goal = goal
        PERIOD[name] = (PERIOD[name] + 1)%times
    
    def __call__(self, *args, **kwargs):
        if PERIOD[self.name] == self.goal:
            return self.func(*args, **kwargs)
        return None