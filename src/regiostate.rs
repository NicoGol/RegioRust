use fxhash::FxHashMap;
use fixedbitset::FixedBitSet;
use std::hash::{Hasher, Hash};

type Regions       = Vec<Vec<u32>>;
type EdgeSet       = FixedBitSet;
type Heterogeneity = Vec<f64>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RegionalizationState {
    regions: Regions,
    edges:   EdgeSet,
    h:       Heterogeneity,
    h_tot:   f64,
}
impl Hash for RegionalizationState {
    fn hash<H: Hasher>(&self, hash: &mut H) {
        self.edges.hash(hash)
    }
}
impl PartialEq for RegionalizationState {
    fn eq(&self, other: &Rhs)
}
impl RegionalizationState {

    pub fn new(regions: Regions, edges: EdgeSet, h: Vec<f64>, h_tot: f64) -> Self {
        Self {
            regions, edges, h, h_tot
        }
    }
    #[inline]
    pub fn regions(&self) -> &Regions {
        &self.regions
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
    pub fn n_regions(&self) -> usize {
        self.regions.len()
    }
}