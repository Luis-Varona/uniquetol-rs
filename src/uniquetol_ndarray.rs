#[path = "isapprox.rs"] mod isapprox;
#[path = "uniquetol.rs"] mod uniquetol;

use ndarray::{Array, IxDyn};
use crate::isapprox::{EqualNan, Tols};
use crate::uniquetol::{Occurrence, UniqueTolArray, uniquetol};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Axis {
    #[default]
    All,
    Dim(usize),
}

impl Axis {
    pub fn new(dim: Option<usize>, d: usize) -> Result<Self, String> {
        match dim {
            None => Ok(Axis::All),
            Some(value) if value < d => Ok(Axis::Dim(value)),
            Some(_) => Err(format!("Dim value must be between 0 and {}", d - 1)),
        }
    }
}

pub fn uniquetol_ndarray(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
    occurrence: Occurrence,
    axis: Axis,
) -> Array<f64, IxDyn> {
    if axis == Axis::All {
        let arr_flat = arr.flatten().to_vec();
        let arr_unique = uniquetol(&arr_flat, tols, equal_nan, occurrence).arr_unique;
        return Array::from_shape_vec(IxDyn(&[arr_unique.len()]), arr_unique).unwrap();
    }
    
    let dim = match axis {
        Axis::Dim(value) => Some(value),
        _ => panic!("Invalid axis"),
    }.unwrap();
    
    let arrs_flat = arr.axis_iter(ndarray::Axis(dim));
    // find largest subset s.t. no pair of (array) elements is element-wise approx. equal ...
    arr.clone() // placeholder return just so rust-analyzer doesn't yell at me :)
}