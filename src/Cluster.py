class Cluster:
    def __init__(self,index,edges):
        self.name = index
        self.v = [index]
        self.e = edges
        self.on = True

    def get_name(self):
        return self.name

    def get_e(self):
        return self.e

    def get_v(self):
        return self.v

    def find_shortest_edge(self,cluster):
        best_u = self.v[0]
        best_dist = float('inf')
        best_v = cluster.get_v()[0]
        for v in cluster.get_v():
            if v in self.e:
                for u in self.e[v].keys():
                    if self.e[v][u] < best_dist:
                        best_u = u
                        best_v = v
                        best_dist = self.e[v][u]
        return (best_u,best_v),best_dist

    def go_off(self):
        self.on = False

    def grow(self,cluster):
        self.v += cluster.get_v()
        for e in cluster.get_e().keys():
            if e in self.e:
                for u in cluster.get_e()[e].keys():
                    self.e[e][u] = cluster.get_e()[e][u]
            else:
                self.e[e] = cluster.get_e()[e]

    def __eq__(self, other):
        return self.name == other.name()

