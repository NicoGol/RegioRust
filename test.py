# import random
# import numpy as np
# import regiorust as rr
# from redcap import Redcap
# from SCT import SCT
# import pandas as pd
# import time
# import matplotlib.pyplot as plt
#
#
# class Evaluation:
#
#     def __init__(self, dataset, sct_method, K, W, cutoff, h_tot, b_h_tot, R, b_R, proved_exact, h_r, b_h_r, del_e,
#                  b_del_e, times, b_times):
#         self.K = K
#         self.sct_method = sct_method
#         self.dataset = dataset
#         self.W = W
#         self.cutoff = cutoff
#         self.h_tot = h_tot
#         self.baseline_h_tot = b_h_tot
#         self.regions = R
#         self.baseline_regions = b_R
#         self.proved_exact = proved_exact
#         self.h_regions = h_r
#         self.baseline_h_regions = b_h_r
#         self.deleted_edges = del_e
#         self.baseline_deleted_edges = b_del_e
#         self.times = times
#         self.baseline_times = b_times
#
#     def plot_h(self):
#         baseline_h_tot_ratio = np.array(self.baseline_h_tot) / np.array(self.baseline_h_tot)
#         h_tot_ratio = np.array(self.h_tot) / np.array(self.baseline_h_tot)
#         plt.plot(self.K, baseline_h_tot_ratio, label='baseline')
#         plt.plot(self.K, h_tot_ratio, label='mdd approach')
#         plt.title('Performance comparison between mdd method and baseline for ' + self.dataset + ' dataset')
#         plt.xlabel('Number of regions k')
#         plt.ylabel('Overall Heterogeneity')
#         plt.ylim((0.8, 1.01))
#         plt.legend()
#         plt.show()
#
#
# dataset2cols = {'ecodemo_NUTS1': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
#                 'ecodemo_NUTS2': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
#                 'ecodemo_NUTS3': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
#                 'education_BE': ['EDU_LOW_r', 'EDU_MID_r', 'EDU_HIGH_r'],
#                 'USA_ecoregions': ['FAPAR_max_', 'FAPAR_mean', 'FAPAR_min_', 'FAPAR_rang',
#                                    'LAI_max_zo', 'LAI_mean_z', 'LAI_min_zo', 'LAI_range_', 'precip_max',
#                                    'precip_mea', 'precip_min', 'precip_ran', 'temp_max_Z', 'temp_min_Z',
#                                    'temp_std_Z'],
#                 'USA_vote_2004': ['bush_votes_perc']}
#
#
# def create_sct(dataset, method='full_order_CL', talk=False):
#     vertices = pd.read_pickle('./data/' + dataset + '.pkl')[dataset2cols[dataset]]
#     cont_m = pd.read_pickle('./data/' + dataset + '_cont.pkl')
#     dist_m = pd.read_pickle('./data/' + dataset + '_dist.pkl')
#     return SCT(vertices, cont_m, dist_m, method=method, talk=talk)
#
#
# def compare(dataset, K, sct_method='full_order_CL', W=5, cutoff=60, talk=False):
#     sct = create_sct(dataset, method=sct_method)
#     n = len(K)
#     mdd_h_tot, mdd_regions, mdd_regions_h, mdd_proved_exact, mdd_del_e, mdd_times = [], [], [], [], [], []
#     b_h_tot, b_regions, b_regions_h, b_del_e, b_times = [], [], [], [], []
#     for k in K:
#         if talk:
#             print('k = {}'.format(k))
#         h_tot, _, _, proved_exact, edges_removed, partition_time = sct.partition(k, 'mdd', W=W, cutoff=cutoff)
#         mdd_h_tot.append(h_tot)
#         mdd_proved_exact.append(proved_exact)
#         mdd_del_e.append(edges_removed)
#         mdd_times.append(partition_time)
#         regions, regions_h = sct.del_edges_2_regions(edges_removed)
#         mdd_regions.append(regions)
#         mdd_regions_h.append(regions_h)
#         if sct_method == 'full_order_CL':
#             h_tot, regions, regions_h, _, partition_time = pg_redcap(dataset, k)
#         elif sct_method == 'MST':
#             h_tot, regions, regions_h, _, partition_time = pg_skater(dataset, k)
#         b_h_tot.append(h_tot)
#         b_regions.append(regions)
#         b_regions_h.append(regions_h)
#         b_times.append(partition_time)
#     return Evaluation(dataset, sct_method, K, W, cutoff, mdd_h_tot, b_h_tot, mdd_regions, b_regions, mdd_proved_exact,
#                       mdd_regions_h, b_regions_h, mdd_del_e, None, mdd_times, b_times)
#
#
# import pygeoda as pg
#
# def pg_redcap(dataset,k,method="fullorder-completelinkage",talk=False):
#     if talk:
#         print('start redcap')
#     vertices = pd.read_pickle('./data/'+dataset+'.pkl')
#     t1 = time.time()
#     pg_dataset = pg.open(vertices)
#     w = pg.queen_weights(pg_dataset)
#     vertices = vertices[dataset2cols[dataset]]
#     res_redcap = pg.redcap(k,w,vertices,method,scale_method='raw')
#     t2 = time.time()
#     if talk:
#         print('end redcap '+ str(t2-t1))
#     regions = [[] for _ in range(k)]
#     for i,cluster in enumerate(res_redcap['Clusters']):
#         regions[cluster-1].append(i)
#     rc = Redcap(vertices,None,None)
#     regions_h = [rc.compute_h(region) for region in regions]
#     h_tot = sum(regions_h)
#     if talk:
#         print('redcap total heterogeneity : ', h_tot)
#     return h_tot, regions, regions_h, None, t2-t1
#
# def pg_skater(dataset,k,talk=False):
#     if talk:
#         print('start skater')
#     vertices = pd.read_pickle('./data/'+dataset+'.pkl')
#     t1 = time.time()
#     pg_dataset = pg.open(vertices)
#     w = pg.queen_weights(pg_dataset)
#     vertices = vertices[dataset2cols[dataset]]
#     res_skater = pg.skater(k,w,vertices,scale_method='raw')
#     t2 = time.time()
#     if talk:
#         print('end skater '+ str(t2-t1))
#     regions = [[] for _ in range(k)]
#     for i,cluster in enumerate(res_skater['Clusters']):
#         regions[cluster-1].append(i)
#     rc = Redcap(vertices,None,None)
#     regions_h = [rc.compute_h(region) for region in regions]
#     h_tot = sum(regions_h)
#     if talk:
#         print('skater total heterogeneity : ', h_tot)
#     return h_tot, regions, regions_h, None, t2-t1
#
# # dataset = 'USA_vote_2004'
# # print(dataset)
# # K = [5]
# # comp3 = compare(dataset,K,cutoff=60,talk=True)
# # print(comp3.h_tot)
#
#
# dataset = 'ecodemo_NUTS3'
# print(dataset)
# K = [10]#,15,20]
# comp = compare(dataset,K,cutoff=60,talk=True)
# print(comp.h_tot)
# print(comp.baseline_h_tot)



import random
import numpy as np
#from redcap import Redcap
from SCT import SCT
import pandas as pd
import time
import matplotlib.pyplot as plt
from artificial_regions import *


k,size,n_centers,delta,n = 10,10,40,2,10
i = 3
cont_m = pd.read_json('./data/artificial_datasets/size{}_cont.json'.format(size))
df = pd.read_json(
    './data/artificial_datasets/size{}_k{}_centers{}_delta{}_{}.json'.format(size, k, n_centers, delta, i))
dist_m = pd.read_json(
    './data/artificial_datasets/size{}_k{}_centers{}_delta{}_{}_dist.json'.format(size, k, n_centers, delta, i))
vertices = df['val']
sct = SCT(vertices, cont_m, dist_m)

h_tot, _, _, proved_exact, edges_removed, partition_time = sct.partition(k, 'mdd')
#regions, regions_h = sct.del_edges_2_regions(edges_removed)
#h_tot2, regions2, regions_h2, _, del_edges2, partition_time2 = sct.partition(k, 'redcap')
#print(h_tot,h_tot2)
#print([edge for edge in edges_removed if edge not in del_edges2])
#print([edge for edge in del_edges2 if edge not in edges_removed])
