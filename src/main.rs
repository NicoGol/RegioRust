
mod regiostate;
mod regionalization;

//use clap::Parser;
use ddo::*;
use fxhash::FxHashMap;
use regionalization::{RegionalizationRelax, RegionalizationRanking};


fn main() {
    //let EU_NUTS1  = CsvReader::from_path("./data/ecodemo_NUTS1.csv").unwrap().finish().unwrap();
    //let EU_NUTS1_cont = CsvReader::from_path("./data/ecodemo_NUTS1_dist.csv").unwrap().finish().unwrap();
    //let EU_NUTS1_dist = CsvReader::from_path("./data/ecodemo_NUTS1_cont.csv").unwrap().finish().unwrap();


    let mut e2id = FxHashMap::default();
    e2id.insert((0, 1), 0);
    e2id.insert((1, 2), 1);

    let problem = regionalization::Regionalization{
        vertex: vec![
            vec![ 0.5, 0.9],
            vec![ 0.8, 0.6],
            vec![ 0.8, 0.6],
        ],
        k: 2,
        neighbors: vec![
            vec![1],
            vec![0, 2],
            vec![1]
        ],
        id2edge: vec![
            (0, 1),
            (1, 2),
        ],
        edge2id: e2id,
        h: 0.12,
    };
    let relax = RegionalizationRelax {pb: &problem};
    let ranking = RegionalizationRanking;

    let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));
    let mut solver = DefaultSolver::new(
        &problem, 
        &relax, 
        &ranking, 
        &FixedWidth(100), 
        &NoCutoff, 
        &mut fringe);
    
    let Completion{ is_exact, best_value } = solver.maximize();

    println!("is exact ? {is_exact}, best_value : {best_value:?}");
    
}



