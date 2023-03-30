use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use std::{hash::{Hasher, Hash}, sync::{Mutex, Arc}};

pub type Regions       = Vec<Vec<usize>>;
pub type EdgeSet       = FixedBitSet;
pub type Heterogeneity = Vec<f64>;

#[derive(Debug, Clone)]
pub struct RegionalizationState {
    regions: Regions,
    edges:   EdgeSet,
    h:       Heterogeneity,
    h_tot:   f64,
    costs:   Arc<Mutex<FxHashMap<usize, f64>>>
}
impl Hash for RegionalizationState {
    fn hash<H: Hasher>(&self, hash: &mut H) {
        self.edges.hash(hash)
    }
}
impl Eq for RegionalizationState {}
impl PartialEq for RegionalizationState {
    fn eq(&self, other: &Self) -> bool {
        self.edges == other.edges
    }
}
impl RegionalizationState {
    pub fn new(regions: Regions, edges: EdgeSet, h: Vec<f64>) -> Self {
        let h_tot = h.iter().sum::<f64>();
        Self { regions, edges, h, h_tot, costs: Arc::new(Mutex::new(FxHashMap::default())) }
    }
    #[inline]
    pub fn regions(&self) -> &Regions {
        &self.regions
    }
    #[inline]
    pub fn n_regions(&self) -> usize {
        self.regions.len()
    }
    #[inline]
    pub fn edges(&self) -> &EdgeSet {
        &self.edges
    }
    #[inline]
    pub fn h(&self) -> &Heterogeneity {
        &self.h
    }
    #[inline]
    pub fn h_tot(&self) -> f64 {
        self.h_tot
    }
    #[inline]
    pub fn store_cost(&self, deleted_edge: usize, h_tot: f64) {
        self.costs.lock().unwrap().insert(deleted_edge, h_tot);
    }
    #[inline]
    pub fn get_cost(&self, deleted_edge: usize) -> f64 {
        self.costs.lock().unwrap()[&deleted_edge]
    }
}