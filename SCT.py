import time as t
from Cluster import Cluster
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import queue
import time
import pyregiorust as rr



class SCT:
    def __init__(self, vertices, contiguity_matrix, distance_matrix, method = 'full_order_CL',talk=False):
        self.cont_m = contiguity_matrix
        self.dist = distance_matrix
        self.vertices = vertices
        self.talk = talk
        self.times = {}
        self.method = method
        if method == 'full_order_CL':
            self.edges, self.neighbors = self.find_SCT_full_order_CL()
        elif method == 'MST':
            self.edges, self.neighbors = self.find_MST()
        else:
            raise Exception('Wrong SCT method, must be full_order_CL or MST')

    def create_clusters(self):
        clusters = {}
        for i in self.cont_m.index:
            e = {}
            for j in self.cont_m.index:
                if i != j and self.cont_m.loc[i,j] > 0:
                    e[j] = {i: self.dist.loc[i,j]}
            clusters[i] = Cluster(i, e)
        return clusters

    def sorted_full_order_edges(self):
        FullO_E = {}
        indexes = self.dist.index
        for i in range(len(indexes) - 1):
            index1 = indexes[i]
            for j in range(i + 1, len(indexes)):
                index2 = indexes[j]
                dist = self.dist.loc[index1,index2]
                if index1 < index2:
                    FullO_E[(index1, index2)] = dist
                else:
                    FullO_E[(index2, index1)] = dist
        sorted_FullO_E = {k: v for k, v in sorted(FullO_E.items(), key=lambda item: item[1])}
        return sorted_FullO_E

    def merge(self,l,m,clusters,cont_C,dist_C):
        clusters[m].go_off()
        clusters[l].grow(clusters[m])
        for v in clusters[m].get_v():
            clusters[v] = clusters[l]
        cont_C[l] = cont_C[[l,m]].max(axis=1)
        cont_C.loc[l] = cont_C.loc[[l, m]].max(axis=0)
        dist_C[l] = dist_C[[l, m]].max(axis=1)
        dist_C.loc[l] = dist_C.loc[[l, m]].max(axis=0)

    def find_MST(self):
        if self.talk:
            print('start MST')
        t1 = t.time()
        FirstO = self.cont_m.values * self.dist.values
        MST = minimum_spanning_tree(FirstO).toarray()
        neighbors = {v: [] for v in self.vertices.index}
        edges = {}
        for i, row in enumerate(MST):
            for j, el in enumerate(row):
                if el > 0:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges[min(i,j),max(i,j)] = el
        t2 = t.time()
        self.times['SCT'] = t2 - t1
        self.times['sort FuO edges'] = 0.0
        if self.talk:
            print('end MST ' + str(self.times['SCT']))
        return list(edges.keys()), neighbors

    def find_SCT_full_order_CL(self):

        if self.talk:
            print('start sort full_order')
        t1 = t.time()
        FullO_E = self.sorted_full_order_edges()
        t2 = t.time()
        self.times['sort FuO edges'] = t2-t1
        if self.talk:
            print('end sort full order ' + str(self.times['sort FuO edges']))

        if self.talk:
            print('start SCT')
        i = 0
        edges = {}
        neighbors = {v : [] for v in self.vertices.index}
        cont_C = self.cont_m.copy()
        dist_C = self.dist.copy()
        clusters = self.create_clusters()
        t1 = t.time()
        for (u,v) in FullO_E.keys():
            l, m = clusters[u].get_name(), clusters[v].get_name()
            if l != m and cont_C[l][m] != 0 and FullO_E[(u,v)] >= dist_C[l][m]:
                e,cost = clusters[u].find_shortest_edge(clusters[v])
                e = tuple(sorted(e))
                edges[e] = cost
                self.merge(l,m,clusters,cont_C,dist_C) # changing clusters, changing contiguity and changing dist_C
                neighbors[e[0]] += [e[1]]
                neighbors[e[1]] += [e[0]]
                i += 1
        t2 = t.time()
        self.times['SCT'] = t2-t1
        if self.talk:
            print('stop SCT ' + str(self.times['SCT']))
        return list(edges.keys()), neighbors

    def compute_h(self, region):
        mean = np.array(self.vertices.loc[region].mean())
        h = 0.
        for vertice in region:
            h += np.sum(np.square(np.array(self.vertices.loc[vertice]) - mean))
        return h

    def dfs_connected_vertex(self, vertice, edges):
        visited = set()
        q = queue.Queue()
        visited.add(vertice)
        q.put(vertice)
        while not q.empty():
            v = q.get()
            for n in self.neighbors[v]:
                e = tuple(sorted((v,n)))
                if edges[e] and n not in visited:
                    visited.add(n)
                    q.put(n)
        return list(visited)

    def partition(self,k,method,W=5,cutoff=60):
        if self.talk:
            print("== partition started ==")
        h_tot, regions, regions_h, proved_exact, edges_removed = None, None, None, False, None
        partition_time = None
        if method == 'mdd':
            vertices_list = self.vertices.values.tolist()
            if type(vertices_list[0]) != list:
                vertices_list = [[val] for val in vertices_list]
            neighbors_list = [adj for adj in self.neighbors.values()]
            t1 = time.time()
            solution = rr.solve_regionalization(vertices_list, neighbors_list, self.edges, k, W, cutoff)
            partition_time = time.time() - t1
            self.times['mdd_partition'] = partition_time
            #h_tot, proved_exact, edges_removed = solution.h_tot, solution.proved, solution.deleted_edges
        elif method == 'redcap':
            t1 = time.time()
            h_tot, regions, regions_h, edges_removed = self.redcap(k)
            partition_time = time.time() - t1
            self.times['redcap_partition'] = partition_time
        if self.talk:
            print("h_tot  {}".format(h_tot))
            print("time {} sec".format(partition_time))
        return h_tot, regions, regions_h, proved_exact, edges_removed, partition_time

    def redcap(self,k):
        edges = {}
        for edge in self.edges:
            edges[edge] = True
        regions = [list(self.vertices.index)]
        edges_removed = []
        H = [self.compute_h(list(self.vertices.index))]
        while len(regions) < k:
            best_gain = 0.0
            best_edge = None
            region_cut = None
            new_regions = None
            new_h = None
            for e,v in edges.items():
                if v:
                    edges[e] = False
                    r1 = self.dfs_connected_vertex(e[0],edges)
                    h1 = self.compute_h(r1)
                    r2 = self.dfs_connected_vertex(e[1], edges)
                    h2 = self.compute_h(r2)
                    for i,region in enumerate(regions):
                        if e[0] in region and e[1] in region:
                            gain = H[i] - h1 - h2
                            if gain > best_gain:
                                best_gain = gain
                                best_edge = e
                                region_cut = i
                                new_regions = [r1,r2]
                                new_h = [h1,h2]
                    edges[e] = True
            updated_regions = []
            updated_h = []
            edges[best_edge] = False
            edges_removed.append(best_edge)
            for i in range(len(regions)):
                if i == region_cut:
                    updated_regions += new_regions
                    updated_h += new_h
                else:
                    updated_regions.append(regions[i])
                    updated_h.append(H[i])
            regions = updated_regions.copy()
            H = updated_h.copy()
        h_tot = sum(H)
        return h_tot, regions, H, edges_removed

    def del_edges_2_regions(self,del_edges):
        regions = []
        regions_h = []
        edges = {edge: (edge not in del_edges) for edge in self.edges}
        visited = []
        for v in self.vertices.index:
            if v not in visited:
                region = self.dfs_connected_vertex(v,edges)
                visited += region
                regions.append(region)
                regions_h.append(self.compute_h(region))
        return regions, regions_h