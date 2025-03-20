#[path = "isapprox.rs"] mod isapprox;
#[path = "sortperm.rs"] mod sortperm;

use std::str::FromStr;
use isapprox::{Tols, isapprox};

pub fn uniquetol(arr: &[f64], args: UniqueTolArgs) -> UniqueTolArray {
    let use_highest = match args.occurrence {
        Occurrence::Highest => true,
        Occurrence::Lowest => false,
    };
    let n = arr.len();
    let mut out = UniqueTolArray {
        arr_unique: Vec::new(),
        indices_unique: Vec::new(),
        inverse_unique: Vec::new(),
        counts_unique: Vec::new(),
    };
    
    return out;
}

pub struct UniqueTolArray {
    pub arr_unique: Vec<f64>,
    pub indices_unique: Vec<usize>,
    pub inverse_unique: Vec<usize>,
    pub counts_unique: Vec<usize>,
}

pub struct UniqueTolArgs {
    pub atol: f64,
    pub rtol: f64,
    pub occurrence: Occurrence,
}

impl Default for UniqueTolArgs {
    fn default() -> UniqueTolArgs {
        UniqueTolArgs {
            atol: 1e-8,
            rtol: (f64::EPSILON).sqrt(),
            occurrence: Occurrence::Highest,
        }
    }
}

pub enum Occurrence {
    Highest,
    Lowest,
}

impl FromStr for Occurrence {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "highest" => Ok(Occurrence::Highest),
            "lowest" => Ok(Occurrence::Lowest),
            _ => Err("Occurrence must be either 'highest' or 'lowest'".to_string()),
        }
    }
}