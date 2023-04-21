#! /bin/python3

import os

def  solve_regionalization(vertex, neighbors, id2edge, k, w, timeout):
    '''
    '''
    with open('vertices.txt', 'w') as f:
        for j,vs in enumerate(vertex):
            for i,v in enumerate(vs):
                f.write(str(v))
                if i < len(vs) - 1:
                    f.write(' ')
            if j < len(vertex) -1:
                f.write('\n')
    with open('neighbors.txt', 'w') as f:
        for j,ns in enumerate(neighbors):
            for i,n in enumerate(ns):
                f.write(str(n))
                if i < len(ns) - 1:
                    f.write(' ')
            if j < len(neighbors) -1:
                f.write('\n')
    with open('id2edge.txt', 'w') as f:
        for j,(s, d) in enumerate(id2edge):
            f.write(str(s))
            f.write(' ')
            f.write(str(d))
            if j < len(id2edge) -1:
                f.write('\n')
    os.system("./target/release/regiorust -v vertices.txt -n neighbors.txt -i id2edge.txt -k {} -w {} -t {}".format(
        k, w, timeout))
    #return None