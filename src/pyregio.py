from regiostate import RegionalizationState
import numpy as np
import queue


p = 100000

class Regionalization:
    '''
    The problem definition
    '''

    def __init__(self, vertex, k, neighbors, edges):
        self.vertex = vertex
        self.k = k
        self.neighbors = neighbors
        self.edges = edges
        self.H = self.compute_h(list(self.vertex.index))

    def nb_variables(self):
        return self.k-1

    def initial_state(self):
        regions = [list(self.vertex.index)]
        h = [self.H]
        edges = {}
        for edge in self.edges:
            edges[edge] = True
        return RegionalizationState(regions, edges, h)

    def initial_value(self):
        return -int(round(self.H*p))

    def compute_h(self,region):
        mean = np.array(self.vertex.loc[region].mean())
        h = 0.
        for vertice in region:
            h += np.sum(np.square(np.array(self.vertex.loc[vertice])-mean))
        return h


    def dfs_connected_vertex(self, vertice, edges):
        visited = set()
        q = queue.Queue()
        visited.add(vertice)
        q.put(vertice)
        while not q.empty():
            v = q.get()
            for n in self.neighbors[v]:
                e = tuple(sorted((v, n)))
                if edges[e] and n not in visited:
                    visited.add(n)
                    q.put(n)
        return list(visited)

    def transition(self, state, variable, value):
        edge = self.edges[value]
        new_edges = state.edges.copy()
        new_edges[edge] = False
        new_regions = []
        new_h = []
        for i,region in enumerate(state.regions):
            if edge[0] in region and edge[1] in region:
                left_region = self.dfs_connected_vertex(edge[0], new_edges)
                right_region = self.dfs_connected_vertex(edge[1], new_edges)
                left_h = self.compute_h(left_region)
                right_h = self.compute_h(right_region)
                new_regions += [left_region] + [right_region]
                new_h += [left_h] + [right_h]

                state.transition_costs[value] = state.h[i] - left_h - right_h
            else:
                new_regions.append(state.regions[i].copy())
                new_h.append(state.h[i])
        return RegionalizationState(new_regions, new_edges, new_h)

    def transition_cost(self, state, variable, value):
        if value in state.transition_costs:
            return int(round(state.transition_costs[value]*p))
        edge = self.edges[value]
        for i, region in enumerate(state.regions):
            if edge[0] in region and edge[1] in region:
                old_h = state.h[i]
                new_edges = state.edges.copy()
                new_edges[edge] = False
                left_region = self.dfs_connected_vertex(edge[0], new_edges)
                right_region = self.dfs_connected_vertex(edge[1], new_edges)
                left_h = self.compute_h(left_region)
                right_h = self.compute_h(right_region)
                return int(round((old_h - left_h - right_h)*p))
        return None

    def domain(self, variable, state):
        if state.n_regions < self.k:
            return [i for i,edge in enumerate(state.edges.keys()) if state.edges[edge]]
        else:
            return []

    def next_variable(self, depth, next_layer):
        if len(next_layer) == 0:
            return None
        else:
            depth = next_layer[0].n_regions
            if depth < self.k :
                return depth
            else:
                return None


class RegionalizationRelax:
    '''
    This is the problem relaxation
    '''

    def __init__(self, problem):
        self.problem = problem

    def merge(self, states):
        new_regions = []
        new_edges = states[0].edges.copy()
        new_h = []
        edges_removed = set()
        for s in states[1:]:
            for edge,v in s.edges.items():
                if v is False and new_edges[edge] is True:
                    new_edges[edge] = False
                    edges_removed.add(edge)
        for i,region in enumerate(states[0].regions):
            edges_computed = []
            vertex_visited = []
            modified = False
            for edge in edges_removed:
                if edge[0] in region and edge[1] in region:
                    modified = True
                    edges_computed.append(edge)
                    for vertex in edge:
                        if vertex not in vertex_visited:
                            new_region = self.problem.dfs_connected_vertex(vertex, new_edges)
                            vertex_visited += new_region
                            new_regions.append(new_region)
                            new_h.append(self.problem.compute_h(new_region))
            if modified:
                for edge in edges_computed:
                    edges_removed.remove(edge)
            else:
                new_regions.append(region.copy())
                new_h.append(states[0].h[i])
        return RegionalizationState(new_regions, new_edges,new_h)

    def relax(self, edge):
        return edge["cost"]

    def fast_upper_bound(self, state):
        n = self.problem.k - state.n_regions
        h_sorted = sorted(state.h, reverse=True)
        tot = 0.
        for i in range(n):
            tot+= h_sorted[i]
        return int(round(tot*p))



class RegionalizationRanking:
    '''
    An heuristic to discriminate the better states from the worse ones
    '''

    def compare(self, a, b):
        return int(round((sum(b.h) - sum(a.h))*p))

