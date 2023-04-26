import random
import numpy as np
from SCT import SCT
import pandas as pd
import pygeoda as pg
import time
import matplotlib.pyplot as plt
from artificial_regions import *
import argparse
from pathlib import Path

columns = ['size', 'k', 'n_centers', 'delta', 'n', 'sct_method', 'W', 'cutoff', 'h_tot', 'baseline_h_tot', 'true_h_tot',
           'precision', 'recall', 'F1 score', 'baseline_precision', 'baseline_recall', 'baseline_F1 score',
           'proved_exact',  'times', 'baseline_times']


class Evaluation:

    def __init__(self, vertices_values, size, k, n_centers, delta, n, sct_method, W, cutoff, h_tot, b_h_tot, R, b_R,
                 true_R, proved_exact, h_r, b_h_r, del_e, b_del_e, times, b_times):
        self.vertices = vertices_values
        self.size = size
        self.N = size * size
        self.k = k
        self.n_centers = n_centers
        self.delta = delta
        self.n = n
        self.sct_method = sct_method
        self.W = W
        self.cutoff = cutoff
        self.h_tot = h_tot
        self.baseline_h_tot = b_h_tot
        self.regions = R
        self.baseline_regions = b_R
        self.true_regions = true_R
        self.true_h_tot = self.compute_true_h_tot()
        self.proved_exact = proved_exact
        self.h_regions = h_r
        self.baseline_h_regions = b_h_r
        self.deleted_edges = del_e
        self.baseline_deleted_edges = b_del_e
        self.times = times
        self.baseline_times = b_times

    def mean_h_tot_ratio(self):
        return (np.array(self.h_tot) / np.array(self.baseline_h_tot)).mean()

    def compute_true_h_tot(self):
        true_h_tot = []
        for d in range(self.n):
            h_tot = 0.0
            regions = [[] for _ in range(self.k)]
            for i, r in enumerate(self.true_regions[d]):
                regions[int(r - 1)].append(i)
            for region in regions:
                values = [self.vertices[d][v] for v in region]
                mean = np.array(values).mean()
                for val in values:
                    h_tot += (val - mean) ** 2
            true_h_tot.append(h_tot)
        return true_h_tot

    def visualization(self, datasets_id=None):
        if datasets_id is None:
            datasets_id = range(self.n)
        plt.figure(figsize=(20, 20))
        for d in datasets_id:
            val = np.array(self.vertices[d]).reshape((self.size, self.size))
            true_regions = np.array(self.true_regions).reshape((self.size, self.size))
            v2r = [-1 for _ in range(self.N)]
            b_v2r = [-1 for _ in range(self.N)]
            for i, region in enumerate(self.regions[d]):
                for v in region:
                    v2r[v] = i
            mdd_regions = np.array(v2r).reshape((self.size, self.size))
            for i, region in enumerate(self.baseline_regions[d]):
                for v in region:
                    b_v2r[v] = i
            b_regions = np.array(b_v2r).reshape((self.size, self.size))

            plt.subplot(1, 4, 1)
            plt.imshow(val)
            plt.title('Values for dataset {}'.format(i))
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(true_regions)
            plt.title('True regions for dataset {}'.format(i))
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(mdd_regions)
            plt.title('MDD regions for dataset {}'.format(i))
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(b_regions)
            plt.title('Baseline regions for dataset {}'.format(i))
            plt.axis('off')

            plt.show()

    def supervised_metrics(self):
        rand_index, recall, precision, F1 = [], [], [], []
        b_rand_index, b_recall, b_precision, b_F1 = [], [], [], []
        for d in range(self.n):
            TP, TN, FP, FN = 0, 0, 0, 0
            b_TP, b_TN, b_FP, b_FN = 0, 0, 0, 0
            v2r = [-1 for _ in range(self.N)]
            b_v2r = [-1 for _ in range(self.N)]
            for i, region in enumerate(self.regions[d]):
                for v in region:
                    v2r[v] = i
            for i, region in enumerate(self.baseline_regions[d]):
                for v in region:
                    b_v2r[v] = i
            for i in range(self.N - 1):
                for j in range(i + 1, self.N):
                    if self.true_regions[d][i] == self.true_regions[d][j]:
                        if v2r[i] == v2r[j]:
                            TP += 1
                        else:
                            FN += 1
                        if b_v2r[i] == b_v2r[j]:
                            b_TP += 1
                        else:
                            b_FN += 1
                    else:
                        if v2r[i] == v2r[j]:
                            FP += 1
                        else:
                            TN += 1
                        if b_v2r[i] == b_v2r[j]:
                            b_FP += 1
                        else:
                            b_TN += 1
            rand_index.append((TP + TN) / (TP + TN + FP + FN))
            recall.append((TP) / (TP + FN))
            precision.append((TP) / (TP + FP))
            F1.append(2 * recall[d] * precision[d] / (recall[d] + precision[d]))
            b_rand_index.append((b_TP + TN) / (b_TP + b_TN + b_FP + b_FN))
            b_recall.append((b_TP) / (b_TP + b_FN))
            b_precision.append((b_TP) / (b_TP + b_FP))
            b_F1.append(2 * b_recall[d] * b_precision[d] / (b_recall[d] + b_precision[d]))
        res = {'rand index': np.mean(rand_index), 'precision': np.mean(precision), 'recall': np.mean(recall),
               'F1 score': np.mean(F1)}
        b_res = {'rand index': np.mean(b_rand_index), 'precision': np.mean(b_precision), 'recall': np.mean(b_recall),
                 'F1 score': np.mean(b_F1)}
        return res, b_res

    def to_list(self):
        sup_metrics, baseline_sup_metrics = self.supervised_metrics()
        return [self.size, self.k, self.n_centers, self.delta, self.n, self.sct_method, self.W, self.cutoff, self.h_tot,
                self.baseline_h_tot, self.true_h_tot, sup_metrics['precision'], sup_metrics['recall'],
                sup_metrics['F1 score'], baseline_sup_metrics['precision'], baseline_sup_metrics['recall'],
                baseline_sup_metrics['F1 score'], self.proved_exact,  self.times, self.baseline_times]


def evaluate(size, k, n_centers, delta, n, mean_range=50, sct_method='full_order_CL', W=5, cutoff=60, talk=False):
    cont_m = pd.read_json('./data/artificial_datasets/size{}_cont.json'.format(size))
    vertices_values = []
    true_R, true_h_tot = [], []
    mdd_h_tot, mdd_regions, mdd_regions_h, mdd_proved_exact, mdd_del_e, mdd_times = [], [], [], [], [], []
    b_h_tot, b_regions, b_regions_h, b_del_e, b_times = [], [], [], [], []
    for i in range(n):
        print('map ' + str(i))
        df = pd.read_json(
            './data/artificial_datasets/size{}_k{}_centers{}_delta{}_mr{}_{}.json'.format(size, k, n_centers, delta,
                                                                                         mean_range, i))
        dist_m = pd.read_json(
            './data/artificial_datasets/size{}_k{}_centers{}_delta{}_mr{}_{}_dist.json'.format(size, k, n_centers, delta,
                                                                                              mean_range, i))
        vertices = df['val']
        vertices_values.append(vertices)
        sct = SCT(vertices, cont_m, dist_m, method=sct_method, talk=talk)
        if talk:
            print('dataset number {}'.format(i))
        h_tot, _, _, proved_exact, edges_removed, partition_time = sct.partition(k, 'mdd', W=W, cutoff=cutoff)
        mdd_h_tot.append(h_tot)
        mdd_proved_exact.append(proved_exact)
        mdd_del_e.append(edges_removed)
        mdd_times.append(partition_time)
        regions, regions_h = sct.del_edges_2_regions(edges_removed)
        mdd_regions.append(regions)
        mdd_regions_h.append(regions_h)
        h_tot, regions, regions_h, _, del_edges, partition_time = sct.partition(k, 'redcap')
        #h_tot, regions, regions_h, _, partition_time = pg_regio(sct,k,method=sct_method)
        b_h_tot.append(h_tot)
        b_regions.append(regions)
        b_regions_h.append(regions_h)
        b_del_e.append(del_edges)
        b_times.append(partition_time)
        true_R.append(list(df['region']))
    return Evaluation(vertices_values, size, k, n_centers, delta, n, sct_method, W, cutoff, mdd_h_tot, b_h_tot,
                      mdd_regions, b_regions, true_R, mdd_proved_exact, mdd_regions_h, b_regions_h, mdd_del_e, b_del_e,
                      mdd_times, b_times)

def pg_regio(sct,k,method='full_order_CL',talk=False):
    if talk:
        print('start redcap')
    vertices = sct.vertices
    neighbors = sct.neighbors.values()
    dataset = pd.DataFrame(vertices)
    dataset.insert(1,'neighbors',neighbors)
    print(vertices)
    t1 = time.time()
    pg_dataset = pg.open(dataset)
    w = pg.queen_weights(pg_dataset)
    vertices = dataset[['val']]
    res = None
    if method == 'full_order_CL':
        res = pg.redcap(k,w,vertices,"fullorder-completelinkage",scale_method='raw')
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

def eval_variate_param(k,sizes,n_centers,deltas,n,mean_ranges,W,cutoffs,filename):
    if Path('./data/artificial_datasets/' + filename + '.csv').is_file():
        results = pd.read_csv('./data/artificial_datasets/' + filename + '.csv', index_col=0)
    else:
        results = pd.DataFrame(columns=columns)
    for size in sizes:
        print('size = ' +str(size))
        for n_center in n_centers:
            print('n center = '+str(n_center))
            for delta in deltas:
                print('delta = ' + str(delta))
                for mean_range in mean_ranges:
                    print('mean range = ' + str(mean_range))
                    for w in W:
                        print('W = ' + str(w))
                        for cutoff in cutoffs:
                            print('cutoff = ' + str(cutoff))
                            #save_artificial_datasets(k, size, n_center, delta, n, mean_range)
                            eva = evaluate(size, k, n_center, delta, n, mean_range, W=w, cutoff=cutoff)
                            results.loc[len(results)] = eva.to_list()
                            results.to_csv('./data/artificial_datasets/' + filename + '.csv')
    return results




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-k", help="number of regions")
  parser.add_argument("-n", help="number of instances")
  parser.add_argument("-s", help="sizes of maps")
  parser.add_argument("-nc", help="number of centers")
  parser.add_argument("-d", help="delta")
  parser.add_argument("-mr", help="mean ranges")
  parser.add_argument("-w", help="width for the mdd")
  parser.add_argument("-c", help="cutoff for the mdd")
  parser.add_argument("-f", help="name of the output file")

  args = parser.parse_args()

  k = int(args.k)
  deltas = [int(d) for d in args.d.split(',')]
  sizes = [int(s) for s in args.s.split(',')]
  n_centers = [int(n_c) for n_c in args.nc.split(',')]
  mean_ranges = [int(m_r) for m_r in args.mr.split(',')]
  W = [int(w) for w in args.w.split(',')] if args.w != None else [5]
  cutoffs = [int(c) for c in args.c.split(',')] if args.c != None else [60]
  n = int(args.n) if args.n != None else 20
  filename = args.f if args.f != None else 'synthetic_maps_results'
  df = eval_variate_param(k,sizes,n_centers,deltas,n,mean_ranges,W,cutoffs,filename)
