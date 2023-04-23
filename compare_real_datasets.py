from SCT import SCT
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import pygeoda as pg
import argparse


dataset2cols = {'ecodemo_NUTS1': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
                'ecodemo_NUTS2': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
                'ecodemo_NUTS3': ['density', 'gdp_inhabitant', 'median_age', 'rate_migration'],
                'education_BE': ['EDU_LOW_r', 'EDU_MID_r', 'EDU_HIGH_r'],
                'USA_ecoregions': ['FAPAR_max_', 'FAPAR_mean', 'FAPAR_min_', 'FAPAR_rang',
                                   'LAI_max_zo', 'LAI_mean_z', 'LAI_min_zo', 'LAI_range_', 'precip_max',
                                   'precip_mea', 'precip_min', 'precip_ran', 'temp_max_Z', 'temp_min_Z',
                                   'temp_std_Z'],
                'USA_vote_2004': ['bush_votes_perc']}

columns = ['dataset', 'k', 'model', 'sct_method', 'W', 'cutoff', 'proved_exact', 'h_tot', 'regions', 'h_regions','del_e', 'time']

def pg_regio(dataset,k,sct,method="fullorder-completelinkage",talk=False):
    if talk:
        print('start redcap')
    vertices = pd.read_json('./data/'+dataset+'/'+dataset+'.json')
    t1 = time.time()
    pg_dataset = pg.open(vertices)
    w = pg.queen_weights(pg_dataset)
    vertices = vertices[dataset2cols[dataset]]
    res = None
    if method == "fullorder-completelinkage":
        res = pg.redcap(k,w,vertices,method,scale_method='raw')
    else:
        res = pg.skater(k, w, vertices, scale_method='raw')
    t2 = time.time()
    if talk:
        print('end redcap '+ str(t2-t1))
    regions = [[] for _ in range(k)]
    for i,cluster in enumerate(res['Clusters']):
        regions[cluster-1].append(i)
    regions_h = [sct.compute_h(region) for region in regions]
    h_tot = sum(regions_h)
    if talk:
        print('redcap total heterogeneity : ', h_tot)
    return h_tot, regions, regions_h, None, t2-t1



def compare(dataset, K, sct_methods, W, cutoffs, talk=False):
    results = pd.DataFrame(columns=columns)
    for sct_method in sct_methods:
        if talk:
            print('sct method = {}'.format(sct_method))
        sct = SCT(dataset, method=sct_method)
        for k in K:
            if talk:
                print('k = {}'.format(k))
            for w in W:
                if talk:
                    print('w = {}'.format(w))
                for cutoff in cutoffs:
                    if talk:
                        print('cutoff = {}'.format(cutoff))
                    h_tot, regions, regions_h, proved_exact, edges_removed, partition_time = sct.partition(k, 'mdd', W=w, cutoff=cutoff)
                    results.loc[len(results)] = [dataset, k, 'mdd', sct_method, w, cutoff, proved_exact, h_tot, regions, regions_h, edges_removed, partition_time]
            if sct_method == 'full_order_CL':
                h_tot, regions, regions_h, _, partition_time = pg_regio(dataset, k, sct)
                results.loc[len(results)] = [dataset, k, 'redcap', sct_method, None, None, False, h_tot, regions, regions_h,None, partition_time]
            elif sct_method == 'MST':
                h_tot, regions, regions_h, _, partition_time = pg_regio(dataset, k, sct, method='skater')
                results.loc[len(results)] = [dataset, k, 'skater', sct_method, None, None, False, h_tot, regions, regions_h,None, partition_time]
    return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", help="name of the dataset")
  parser.add_argument("-w", help="width for the mdd")
  parser.add_argument("-c", help="cutoff for the mdd")
  parser.add_argument("-f", help="name of the output file")
  parser.add_argument("-t", help='talk')

  args = parser.parse_args()
  K = [5,10,15,20]
  dataset = args.d
  W = [int(w) for w in args.w.split(',')] if args.w != None else [5]
  cutoffs = [int(c) for c in args.c.split(',')] if args.c != None else [60]
  talk = (args.t != None)
  filename = args.f if args.f != None else dataset+'_result'
  df = compare(dataset,K,['full_order_CL','MST'],W,cutoffs,talk)

  df.to_csv('./data/'+dataset+'/'+filename+'.csv')

