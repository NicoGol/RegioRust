import random
import numpy as np
import regiorust as rr

def generate_random_inst(vertices, attributes):
    '''
    returns a matrix [vertices * attributes] of floating point values.
    The values are normalized on a per row basis.
    '''
    denormalized = [
        [random.random() for _ in range(attributes) ]
        for _ in range(vertices)
    ]
    total_w = [ 
        sum([ denormalized[v][a] for a in range(attributes) ]) 
        for v in range(vertices) 
    ]
    normalized = [
        [ denormalized[v][a] / total_w[a] for a in range(attributes) ] 
        for v in range(vertices) 
    ]
    return normalized

def make_spanning_tree(vertices):
    '''
    returns a list of edges to turn a random graph into a tree
    '''
    open   = { i for i in range(1, vertices) }
    closed = { 0 }
    result = []
    while open:
        src = random.choice(list(open))
        dst = random.choice(list(closed))
        a   = min(src, dst)
        b   = max(src, dst)
        result.append((a, b))
        open.remove(src)
        closed.add(src)
    return result

def make_ajdlist(vertices, edges):
    adjlist = [ [] for i in range(vertices) ]
    for (src, dst) in edges:
        adjlist[src].append(dst)
        adjlist[dst].append(src)
    return adjlist

def main():
    random.seed(0xCAFECAFE)
    N_vertices   = 200
    N_attributes = 200
    k            = 10
    vertices     = generate_random_inst(N_vertices, N_attributes)
    edges        = make_spanning_tree(N_vertices)
    neighbors    = make_ajdlist(N_vertices, edges)
    #
    solution     = rr.solve_regionalization(vertices, neighbors, edges, k, 10, 60)
    print("proved {}".format(solution.proved))
    print("h_tot  {}".format(solution.h_tot))

if __name__ == '__main__':
    main()
