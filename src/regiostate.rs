use fixedbitset::FixedBitSet;
use std::hash::{Hasher, Hash};

pub type Regions       = Vec<Vec<usize>>;
pub type EdgeSet       = FixedBitSet;
pub type Heterogeneity = Vec<f64>;

#[derive(Debug, Clone)]
pub struct RegionalizationState {
    regions: Regions,
    edges:   EdgeSet,
    h:       Heterogeneity,
    h_tot:   f64,
    costs:   Vec<f64>
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
macro_rules! forgive_me {
    (store $ty:ty, $value: expr, into $dest: expr) => {
        unsafe {
            let dest = &$dest as *const $ty as *mut $ty;
            *dest = $value;
         }
    };
}
impl RegionalizationState {
    pub fn new(regions: Regions, edges: EdgeSet, e_max: usize, h: Vec<f64>) -> Self {
        let h_tot = h.iter().sum::<f64>();
        Self { regions, edges, h, h_tot, costs: vec![0.0; e_max]}
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
        forgive_me!(store f64, h_tot, into self.costs[deleted_edge]);
        //self.costs[deleted_edge].store(h_tot.to_bits(), Ordering::Relaxed);
    }
    #[inline]
    pub fn get_cost(&self, deleted_edge: usize) -> f64 {
        //f64::from_bits(self.costs[deleted_edge].load(Ordering::Relaxed))
        self.costs[deleted_edge]
    }
}