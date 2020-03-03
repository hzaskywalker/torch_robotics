class Constraint:
    def __init__(self, priority, perpetual=True):
        self.priority = priority
        self.perpetual = perpetual

    def prerequisites(self, sim):
        return True

    def preprocess(self, sim):
        pass

    def postprocess(self, sim):
        # 强行 set huiqu
        pass

    def cost(self, sim_t, s, t):
        # we assume
        return 0
