#[path = "isapprox.rs"] mod isapprox;
#[path = "uniquetol.rs"] mod uniquetol;

use ndarray::{Array, Axis, IxDyn};
use crate::isapprox::{EqualNan, Tols, isapprox};
use crate::uniquetol::{Occurrence, uniquetol};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisFlatten {
    #[default]
    None,
    Dim(usize),
}

#[inline]
fn sortperm(arr: &[f64]) -> Vec<usize> {
    let mut perm_sorted: Vec<usize> = (0..arr.len()).collect();
    perm_sorted.sort_by(|&i, &j| arr[i].total_cmp(&arr[j]));
    perm_sorted
}

fn uniquetol_groups(arr: &[f64], tols: Tols, equal_nan: EqualNan) -> Vec<Vec<usize>> {
    let perm_sorted = sortperm(arr);
    let mut groups = vec![vec![perm_sorted[0]]];
    let mut curr = arr[perm_sorted[0]];
    
    for &idx in perm_sorted.iter().skip(1) {
        let next = arr[idx];
        
        match isapprox(curr, next, tols, equal_nan) {
            true => groups.last_mut().unwrap().push(idx),
            false => {
                groups.push(vec![idx]);
                curr = next;
            }
        }
    }
    
    groups
}

pub fn uniquetol_ndarray(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
    axis_flatten: AxisFlatten,
) -> Array<f64, IxDyn> {
    let axis = match axis_flatten {
        AxisFlatten::None => {
            let arr_flat = arr.flatten().to_vec();
            let arr_unique = uniquetol(
                &arr_flat, tols, equal_nan, Occurrence::default()
            ).arr_unique;
            return Array::from_shape_vec(IxDyn(&[arr_unique.len()]), arr_unique).unwrap();
        }
        AxisFlatten::Dim(axis) if axis < arr.ndim() => axis,
        AxisFlatten::Dim(axis) => {
            panic!("Axis {} out of bounds for array of dimension {}", axis, arr.ndim());
        }
    };
    
    let arrs_flat = arr.axis_iter(Axis(axis)).collect::<Vec<_>>();
    let n = arrs_flat[0].len();
    let mut groups: Vec<Vec<usize>>  = vec![(0..n).collect()];
    
    for idx in 0..n {
        let mut groups_new = Vec::new();
        
        for group in groups.iter() {
            let arr: Vec<f64> = group.iter().map(|&i| arrs_flat[i][idx]).collect();
            groups_new.extend(uniquetol_groups(&arr, tols, equal_nan));
        }
        
        groups = groups_new;
    }
    
    let arr_unique: Vec<f64> = groups.iter().map(|group| arr[group[0]]).collect();
    Array::from_shape_vec(IxDyn(&[arr_unique.len()]), arr_unique).unwrap()
}