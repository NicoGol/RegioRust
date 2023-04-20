import random
import numpy as np
import pandas as pd

def artificial_regions(k,size,n_centers,delta,mean_range=(1,50)):
    N = size*size
    if n_centers > N or k > N:
        print('too many centers or too many regions')
        raise Exception
    cont_m = contiguity_matrix(size)
    clusters = allocate_object_2_centers(size,n_centers)
    regions = merge_clusters(k,clusters,cont_m)
    values = assignate_values(k,size,regions,delta,cont_m,mean_range)
    vertice2region = np.zeros(N)
    for i,region in enumerate(regions):
        for v in region:
            vertice2region[v] = i
    return values, cont_m, regions, vertice2region

def assignate_values(k,size,regions,delta,cont_m,mean_range):
    N = size * size
    values = np.zeros(N)
    regions_adj = [[] for _ in range(k)]
    for i in range(k-1):
        for j in range(i+1,k):
            if connected(regions[i],regions[j],cont_m):
                regions_adj[i].append(j)
                regions_adj[j].append(i)
    regions_means = [None for _ in range(k)]
    means = set(range(mean_range[0],mean_range[1]))
    for i in range(k):
        neighbors_means = set()
        for neighbor in regions_adj[i]:
            if regions_means[neighbor] is not None:
                neighbors_means = neighbors_means.union(set(range(regions_means[neighbor] - delta + 1, regions_means[neighbor] + delta)))
        possible_means = means.difference(neighbors_means)
        if len(possible_means) == 0:
            print('values unpossible to assignate')
            raise Exception
        regions_means[i] = random.choice(list(possible_means))
        for v in regions[i]:
            values[v] = np.random.normal(regions_means[i],1)
    return values

def contiguity_matrix(size):
    N = size * size
    cont_m = np.zeros((N, N))
    for i in range(N):
        if i % size > 0:
            cont_m[i, i - 1] = 1
        if i >= size:
            cont_m[i, i - size] = 1
        if i % size < size - 1:
            cont_m[i, i + 1] = 1
        if i < N - size:
            cont_m[i, i + size] = 1
    return cont_m

def allocate_object_2_centers(size,n_centers):
    N = size * size
    centers = [divmod(ele, size) for ele in random.sample(range(N), n_centers)]
    clusters = [[] for _ in range(n_centers)]
    for i in range(size):
        for j in range(size):
            so_id = j*size + i
            min_dist = N
            center = None
            for id, (c_x, c_y) in enumerate(centers):
                dist = abs(i - c_x) + abs(j - c_y)
                if dist < min_dist:
                    min_dist = dist
                    center = id
            clusters[center].append(so_id)
    return clusters

def connected(c1,c2,cont_m):
    for so1 in c1:
        if sum(cont_m[so1,c2]) > 0:
            return True
    return False


def merge_clusters(k,clusters,cont_m):
    while len(clusters) > k:
        [a,b] = random.sample(range(len(clusters)),2)
        c_a, c_b = clusters[a], clusters[b]
        if connected(c_a,c_b,cont_m):
            new_clusters = []
            for i,cluster in enumerate(clusters):
                if i == a:
                    new_clusters.append(c_a+c_b)
                elif i != b:
                    new_clusters.append(clusters[i])
            clusters = new_clusters
    return clusters

def save_artificial_datasets(k,size,n_centers,delta,n):
    N = size*size
    for id in range(n):
        values, cont_m, regions, vertice2region = artificial_regions(k, size, n_centers, delta)
        df = pd.DataFrame(np.array([values,vertice2region]).transpose(),columns=['val','region'])
        dist_m = np.zeros((N,N))
        for i in range(N-1):
            for j in range(i+1,N):
                d = (values[i]-values[j])**2
                dist_m[i,j] = d
                dist_m[j,i] = d
        df.to_pickle('./data/artificial_datasets/size{}_k{}_centers{}_delta{}_{}.pkl'.format(size,k,n_centers,delta,id))
        pd.DataFrame(dist_m).to_pickle('./data/artificial_datasets/size{}_k{}_centers{}_delta{}_{}_dist.pkl'.format(size,k,n_centers,delta,id))
        pd.DataFrame(cont_m).to_pickle('./data/artificial_datasets/size{}_cont.pkl'.format(size))

