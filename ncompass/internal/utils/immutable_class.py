class ImmutableClass(object):
    def __init__(self):
        self.attrWasSet = []

    def mutate(self, name, value):
        idx = self.attrWasSet.index(name)
        self.attrWasSet.pop(idx)
        self.__setattr__(name, value)
    
    def __setattr__(self, name, value):
        if name == 'attrWasSet':        super().__setattr__(name, value)
        elif name in self.attrWasSet:   raise RuntimeError('Cannot change state once created')
        else:
            self.attrWasSet.append(name)
            super().__setattr__(name, value)
