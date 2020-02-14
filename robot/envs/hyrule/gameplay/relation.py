class Relation:
    # constraints are similar to pytorch functions..
    # we maintain
    def __call__(self, object):
        return self.parse(object)

    def parse(self, object):
        raise NotImplementedError
