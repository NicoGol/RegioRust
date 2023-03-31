use std::time::Duration;

use pyo3::prelude::*;

use ddo::*;
use regionalization::{RegionalizationRelax, RegionalizationRanking, P};

mod regiostate;
mod regionalization;


/// This module exposes binding to the ddo (rust) engine to perform
/// fast discrete optimization using decision diagrams.
#[pymodule]
fn regiorust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_regionalization, m)?)?;
    Ok(())
}

#[pyclass]
pub struct RegioSolution {
    pub proved: bool, 
    pub h_tot: f64,
    pub deleted_edges: Vec<(usize, usize)>
}

#[pyfunction]
pub fn solve_regionalization(
    vertex: Vec<Vec<f64>>,
    neighbors: Vec<Vec<usize>>,
    id2edge: Vec<(usize, usize)>,
    k: usize,
    w: usize,
    timeout: u64,
) -> Option<RegioSolution> {
    let problem = regionalization::Regionalization::new(vertex, neighbors, id2edge, k);

    let relax = RegionalizationRelax {pb: &problem};
    let ranking = RegionalizationRanking;
    let width = FixedWidth(w);
    let timeout = TimeBudget::new(Duration::from_secs(timeout));
    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));
    let mut solver = DefaultBarrierSolver::new(
        &problem, 
        &relax, 
        &ranking, 
        &width,
        &timeout, 
        &mut fringe);
    
    let Completion{ is_exact, best_value } = solver.maximize();
    let solution = solver.best_solution();

    let mut deleted_edges = vec![];
    if let Some(solution) = solution {
        for Decision { value, .. } in solution {
            deleted_edges.push(problem.id2edge[value as usize])
        }

        let best_value = best_value
            .map(|h| (-h) as f64 / P)
            .unwrap_or(f64::NAN);

        Some(RegioSolution {
            proved: is_exact,
            h_tot: best_value,
            deleted_edges
        })
    } else {
        None
    }
}


