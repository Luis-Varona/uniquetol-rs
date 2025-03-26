#[path = "isapprox.rs"] mod isapprox;
#[path = "uniquetol.rs"] mod uniquetol;

use itertools::Itertools;
use ndarray::{Array, Axis, IxDyn};
use crate::isapprox::{EqualNan, Tols, isapprox};
use crate::uniquetol::{Occurrence, uniquetol};

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
    
    let arrs_flat = arr.axis_iter(Axis(axis)).collect::<Vec<_>>();
    
    // find largest subset s.t. no pair of (array) elements is element-wise approx. equal ...
    // the following is a very naive placeholder implementation while I think of a better one
    let mut k = arrs_flat.len();
    
    loop {
        let mut independent = true;
        
        for subset in Itertools::combinations(arrs_flat.iter(), k) {
            let mut i = 0;
            
            while independent && i < k - 1 {
                let mut j = i + 1;
                
                while independent && j < k {
                    let arr_i = subset[i];
                    let arr_j = subset[j];
                    independent = !arr_i.iter().zip(arr_j.iter()).all(|(&x, &y)| isapprox(x, y, tols, equal_nan));
                    j += 1;
                }
                
                i += 1;
            }
            
            if independent {
                let arr_flat = subset.iter().flat_map(|arr| arr.iter()).copied().collect::<Vec<_>>();
                let arr_unique = uniquetol(&arr_flat, tols, equal_nan, occurrence).arr_unique;
                return Array::from_shape_vec(IxDyn(&[arr_unique.len()]), arr_unique).unwrap();
            }
        }
        
        k -= 1;
    }
}