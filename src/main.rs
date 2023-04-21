use std::{time::Duration, fs::read_to_string};
use std::fs;

use clap::Parser;
use ddo::*;
use nom::{
    combinator::map,
    number::complete::double, 
    multi::separated_list0, 
    character::complete::{space1, line_ending, u64}, 
    IResult, 
    sequence::{separated_pair}
};
use regionalization::{RegionalizationRelax, RegionalizationRanking, P};

mod regiostate;
mod regionalization;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to a file containing a matrix of weighted attributes
    #[clap(short, long)]
    vertices: String,
    /// Path to a file containing the adjacency list of the computed tree 
    #[clap(short, long)]
    neighbors: String,
    /// Path to a file containing a list of tuples (source, destination) 
    /// representing the edges of the graph. Each line corresponds to one line.
    /// The line number is the identifier of that edge
    #[clap(short, long)]
    id2edges: String,
    /// The number of desired regions in the end
    #[clap(short, long, default_value="3")]
    k: usize,
    /// The maximum width allowed for any MDD that is compiled.
    #[clap(short, long, default_value="10")]
    w: usize,
    /// The maximum duration to solve the problem instance.
    #[clap(short, long, default_value="60")]
    timeout: u64,
}

#[derive(Debug)]
pub struct RegioSolution {
    pub proved: bool,
    pub h_tot: f64,
    pub deleted_edges: Vec<(usize, usize)>
}

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


fn main() -> Result<(), Error> {
    // parse command line arguments
    let Args{vertices, neighbors, id2edges, k, w, timeout } = Args::parse();
    // parse the content of the various files
    let vertices = parse_vertex_matrix(&read_to_string(vertices)?)
        .map_err(|e| Error::Parsing { fname: "vertices", reason: format!("{e}") })?.1;
    let neighbors = parse_neighbor_adjlist(&read_to_string(neighbors)?)
        .map_err(|e| Error::Parsing { fname: "neighbors", reason: format!("{e}") })?.1;
    let id2edges = parse_edges(&read_to_string(id2edges)?)
        .map_err(|e| Error::Parsing { fname: "id2edges", reason: format!("{e}") })?.1;

    let solution = solve_regionalization(vertices, neighbors, id2edges, k, w, timeout);

    if let Some(RegioSolution{proved, h_tot, deleted_edges}) = solution {
        let status = if proved { "proved" } else { "current-best" };
        fs::write("./result.txt", format!("{} | {} | {:?}",status,h_tot,deleted_edges)).expect("Unable to write file");
    } else {
        fs::write("./result.txt", "no-solution | +inf | []").expect("Unable to write file");
    }
    Ok(())
}


#[derive(Debug, displaythis::Display, thiserror::Error)]
enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("could not parse {fname}: {reason}")]
    Parsing{fname: &'static str, reason: String}
}

fn parse_vertex_matrix(data: &str) -> IResult<&str, Vec<Vec<f64>>> {
    // parse all lines
    separated_list0(line_ending, 
        // parser for one line
        separated_list0(space1, double)
    )(data)
}

fn parse_neighbor_adjlist(data: &str) -> IResult<&str, Vec<Vec<usize>>> {
    // parse all lines
    separated_list0(line_ending, 
        // parser for one line
        separated_list0(space1, usize)
    )(data)
}

fn parse_edges(data: &str) -> IResult<&str, Vec<(usize, usize)>> {
    // parse all lines
    separated_list0(line_ending, 
        // parser for one line
        separated_pair(usize, space1, usize)
    )(data)
}

fn usize(data: &str) -> IResult<&str, usize> {
    map(u64, |x| x as usize)(data)
}