use ddo::{Problem, Variable, Decision, StateRanking, Relaxation};
use fixedbitset::FixedBitSet;
use fixedbitset_utils::BitSetIter;
use fxhash::FxHashMap;

use crate::regiostate::{RegionalizationState, EdgeSet};

pub static P : f64 = 100000000.0;

#[derive(Debug)]
pub struct Regionalization {
    pub vertex: Vec<Vec<f64>>,
    /// regions a la fin
    pub k: usize,
    /// adj list
    pub neighbors: Vec<Vec<usize>>,
    /// edge_id -> source, des
    pub id2edge: Vec<(usize, usize)>,
    /// src, dst -> id
    pub edge2id: FxHashMap<(usize, usize), usize>,
    /// initial heterogeneity
    pub h: f64,
}
impl Regionalization {
    pub fn new(
        vertex: Vec<Vec<f64>>,
        neighbors: Vec<Vec<usize>>,
        id2edge: Vec<(usize, usize)>,
        k: usize,
    ) -> Self {
        let h = Self::_compute_h(&vertex, &(0..vertex.len()).collect::<Vec<_>>());
        let mut edge2id = FxHashMap::default();
        for (i, e) in id2edge.iter().copied().enumerate() {
            edge2id.insert(e, i);
        }

        Self { vertex, k, neighbors, edge2id, id2edge, h }
    }
}
impl Problem for Regionalization {
    type State = RegionalizationState;

    fn nb_variables(&self) -> usize {
        self.k - 1 
    }

    fn initial_state(&self) -> Self::State {
        let regions = vec![
            (0..self.vertex.len()).collect::<Vec<_>>()
        ];
        let h = vec![self.h];
        let mut edges = EdgeSet::with_capacity(self.id2edge.len());
        edges.insert_range(0..self.id2edge.len());

        Self::State::new(regions, edges, h)
    }

    fn initial_value(&self) -> isize {
        - (P * self.h).round() as isize
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let deleted_edge_id = decision.value as usize;
        let (src, dst) = self.id2edge[deleted_edge_id];

        let mut edges = state.edges().clone();
        edges.set(deleted_edge_id, false);

        let mut regions = vec![];
        let mut hetero = vec![];

        for (region, h) in state.regions().iter().zip(state.h().iter().copied()) {
            if region.contains(&src) {
                let partition_src = self.dfs_connected_vertex(src, &edges);
                let hetero_src = self.compute_h(&partition_src);
                
                regions.push(partition_src);
                hetero.push(hetero_src);

                let partition_dst = self.dfs_connected_vertex(dst, &edges);
                let hetero_dst = self.compute_h(&partition_dst);
                
                regions.push(partition_dst);
                hetero.push(hetero_dst);
            } else {
                regions.push(region.clone());
                hetero.push(h)
            }
        }

        let ret = Self::State::new(regions, edges, hetero);
        state.store_cost(deleted_edge_id, ret.h_tot());
        ret
    }

    fn transition_cost(&self, state: &Self::State, decision: ddo::Decision) -> isize {
        let deleted_edge_id = decision.value as usize;

        let child_cost = state.get_cost(deleted_edge_id);

        ((state.h_tot() - child_cost) * P ).round() as isize
    }

    fn for_each_in_domain(&self, variable: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let existing = state.edges();
        for edge in BitSetIter::new(existing) {
            let value = edge as isize;
            f.apply(Decision{variable, value})
        }
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>) -> Option<ddo::Variable> {
        if depth < self.nb_variables() {
            Some(Variable(depth))
        } else {
            None
        }
    }
}

/// Utility functions
impl Regionalization {
    pub fn _compute_h(vertex: &[Vec<f64>], regions: &[usize]) -> f64 {
        let n_attr = vertex[0].len();

        let mut h_values = vec![0.0; n_attr];

        for region in regions {
            for (i, att) in vertex[*region].iter().enumerate() {
                h_values[i] += *att;
            }
        }
        
        for v in h_values.iter_mut() {
            *v /= regions.len() as f64;
        }

        regions.iter().copied().map(|region| {
            let row = &vertex[region];
            row.iter().copied().zip(h_values.iter())
                .map(|(att, meanatt)| (att-meanatt) * (att-meanatt))
                .sum::<f64>()
        }).sum::<f64>()
    }

    pub fn compute_h(&self, regions: &[usize]) -> f64 {
        Self::_compute_h(&self.vertex, regions)
    }

    fn dfs_connected_vertex(&self, root: usize, edges: &EdgeSet) -> Vec<usize> {
        let mut visited = FixedBitSet::with_capacity(self.vertex.len());
        visited.insert(root);

        let mut connected = vec![];
        let mut queue = vec![root];


        while let Some(item) = queue.pop() {
            connected.push(item);

            for neigbbor in self.neighbors[item].iter().copied() {
                let src = item.min(neigbbor);
                let dst = item.max(neigbbor);
                let edge_id = self.edge2id[&(src, dst)];

                if edges.contains(edge_id) && !visited.contains(neigbbor) {
                    queue.push(neigbbor);
                    visited.insert(neigbbor);
                }
            }
        }

        connected
    }
}


#[derive(Debug, Clone, Copy)]
pub struct RegionalizationRanking;

impl StateRanking for RegionalizationRanking {
    type State  = RegionalizationState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        a.h_tot().partial_cmp(&b.h_tot()).unwrap().reverse()
    }
}

#[derive(Debug, Clone)]
pub struct RegionalizationRelax<'a> {
    pub pb: &'a Regionalization
}

impl Relaxation for RegionalizationRelax<'_> {
    type State = RegionalizationState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut edges = EdgeSet::with_capacity(self.pb.id2edge.len());

        for state in states {
            edges |= state.edges();
        }

        let mut regions = vec![];
        let mut hetero = vec![];
        let mut visited = FixedBitSet::with_capacity(self.pb.vertex.len());
        for vertex in 0..self.pb.vertex.len() {
            if !visited.contains(vertex) {
                let partition = self.pb.dfs_connected_vertex(vertex, &edges);
                for node in &partition {
                    visited.set(*node, true);
                }
                hetero.push(self.pb.compute_h(&partition));
                regions.push(partition);
            }
        }

        Self::State::new(regions, edges, hetero)
    }

    fn relax(
        &self,
        _source: &Self::State,
        _dest: &Self::State,
        _new: &Self::State,
        _decision: ddo::Decision,
        cost: isize,
    ) -> isize {
        cost
    }

    // if this is too slow, then just ditch it.
    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let n = self.pb.k - state.n_regions();
        let mut h = state.h().clone();
        h.sort_unstable_by(|a, b| a.total_cmp(b).reverse());
        let result = h.iter().take(n).sum::<f64>();
        //let result = state.h().iter().sum::<f64>();
        (result * P).round() as isize
    }
}