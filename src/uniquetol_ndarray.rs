#[path = "isapprox.rs"] mod isapprox;
#[path = "uniquetol.rs"] mod uniquetol;

use ndarray::{Array, Axis, IxDyn};
use crate::isapprox::{EqualNan, Tols};
use crate::uniquetol::{Occurrence, UniqueTolArray, uniquetol};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisFlatten {
    Dim(Option<usize>),
}

impl AxisFlatten {
    pub fn new(dim: Option<usize>, d: usize) -> Result<Self, String> {
        match dim {
            None => Ok(AxisFlatten::Dim(None)),
            Some(axis) if axis < d => Ok(AxisFlatten::Dim(Some(axis))),
            Some(_) => Err(format!("Dim axis must be between 0 and {} (inclusive)", d - 1)),
        }
    }
}

impl Default for AxisFlatten {
    fn default() -> Self {
        AxisFlatten::Dim(None)
    }
}

pub fn uniquetol_ndarray(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
    occurrence: Occurrence,
    axis_flatten: AxisFlatten,
) -> Array<f64, IxDyn> {
    let axis = match axis_flatten {
        AxisFlatten::Dim(None) => {
            let arr_flat = arr.flatten().to_vec();
            let arr_unique = uniquetol(&arr_flat, tols, equal_nan, occurrence).arr_unique;
            return Array::from_shape_vec(IxDyn(&[arr_unique.len()]), arr_unique).unwrap();
        }
        AxisFlatten::Dim(Some(axis)) => axis,
    };
    
    let arrs_flat = arr.axis_iter(Axis(axis));
    // find largest subset s.t. no pair of (array) elements is element-wise approx. equal ...
    arr.clone() // placeholder return just so rust-analyzer doesn't yell at me :)
}