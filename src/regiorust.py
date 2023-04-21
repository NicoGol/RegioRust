#! /bin/python3

import os

def  solve_regionalization(vertex, neighbors, id2edge, k, w, timeout):
    '''
    '''
    with open('vertices.txt', 'w') as f:
        for vs in vertex:
            for v in vs:
                f.write(str(v))
                f.write(' ')
            f.write('\n')
    with open('neighbors.txt', 'w') as f:
        for ns in neighbors:
            for n in ns:
                f.write(str(n))
                f.write(' ')
            f.write('\n')
    with open('id2edge.txt', 'w') as f:
        for (s, d) in id2edge:
            f.write(str(s))
            f.write(' ')
            f.write(str(d))
            f.write('\n')
    os.execv("./target/release/regiorust", "-v vertices.txt -n neighbors.txt -i id2edge.txt -k {k} -w {w} -t {timeout}".format(k=k, w=w, timeout=timeout))
    #return None