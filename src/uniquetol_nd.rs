use ndarray::{Array, Axis, IxDyn};

use crate::isapprox::{EqualNan, Tols, isapprox};
use crate::uniquetol_1d::{Occurrence, sortperm, uniquetol_1d};

const SHAPE_ERR: &str = "Failed to convert unique values to ndarray";

#[derive(Debug)]
pub struct BoundsError {
    pub axis: usize,
    pub ndim: usize,
}

impl std::fmt::Display for BoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Axis {} is out of bounds for array with {} dimensions",
            self.axis, self.ndim
        )
    }
}

impl std::error::Error for BoundsError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlattenAxis {
    #[default]
    None,
    Dim(usize),
}

fn uniquetol_groups(arr: &[f64], tols: Tols, equal_nan: EqualNan) -> Vec<Vec<usize>> {
    let perm_sorted = sortperm(arr, false);
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

#[inline]
fn uniquetol_1d_flatten_none(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
) -> Array<f64, IxDyn> {
    let arr_flat = arr.flatten().to_vec();
    let arr_unique = uniquetol_1d(&arr_flat, tols, equal_nan, Occurrence::default()).arr_unique;
    let shape = IxDyn(&[arr_unique.len()]);
    Array::from_shape_vec(shape, arr_unique).expect(SHAPE_ERR)
}

fn uniquetol_1d_flatten_axis(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
    axis: usize,
) -> Array<f64, IxDyn> {
    let arrs_flat = arr.axis_iter(Axis(axis)).collect::<Vec<_>>();
    let n = arrs_flat[0].len();
    let mut groups: Vec<Vec<usize>> = vec![(0..n).collect()];

    for idx in 0..n {
        let mut groups_new = Vec::new();

        for group in groups.iter() {
            let arr: Vec<f64> = group.iter().map(|&i| arrs_flat[i][idx]).collect();
            groups_new.extend(uniquetol_groups(&arr, tols, equal_nan));
        }

        groups = groups_new;
    }

    let arr_unique: Vec<f64> = groups.iter().map(|group| arr[group[0]]).collect();
    let shape = IxDyn(&[arr_unique.len()]);
    Array::from_shape_vec(shape, arr_unique).expect(SHAPE_ERR)
}

pub fn uniquetol_nd(
    arr: &Array<f64, IxDyn>,
    tols: Tols,
    equal_nan: EqualNan,
    flatten_axis: FlattenAxis,
) -> Result<Array<f64, IxDyn>, BoundsError> {
    match flatten_axis {
        FlattenAxis::None => Ok(uniquetol_1d_flatten_none(arr, tols, equal_nan)),
        FlattenAxis::Dim(axis) if axis < arr.ndim() => {
            Ok(uniquetol_1d_flatten_axis(arr, tols, equal_nan, axis))
        }
        FlattenAxis::Dim(axis) => Err(BoundsError {
            axis,
            ndim: arr.ndim(),
        }),
    }
}
