class RegionalizationState:
    '''
    A state of your dynamic programme.
    I know it is a bit annoying, but this class *MUST* implement the methods
    __eq__ and __hash__ at the very least
    '''

    def __init__(self, regions, edges, h):
        self.regions = regions
        self.n_regions = len(regions)
        self.edges = edges
        self.h = h
        self.transition_costs = {}

    def __eq__(self, other):
        if not isinstance(other, RegionalizationState):
            return False
        elif self.n_regions != other.n_regions:
            return False
        else:
            return self.edges == other.edges

    def __hash__(self):
        hash = 0
        for i,v in enumerate(self.edges.values()):
            hash += (i^3)*v
        return hash