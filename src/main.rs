
mod regiostate;
mod regionalization;

use polars::prelude::*;
use std::{path::Path, fs::File, io::{BufReader, BufRead}, time::{Duration, Instant}, num::ParseIntError};

use clap::Parser;
use ddo::*;


fn main() {
    let EU_NUTS1  = CsvReader::from_path("./data/ecodemo_NUTS1.csv").unwrap().finish().unwrap();
    let EU_NUTS1_cont = CsvReader::from_path("./data/ecodemo_NUTS1_dist.csv").unwrap().finish().unwrap();
    let EU_NUTS1_dist = CsvReader::from_path("./data/ecodemo_NUTS1_cont.csv").unwrap().finish().unwrap();
}



