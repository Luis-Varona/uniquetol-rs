use ndarray::{Array, Axis, IxDyn};
use num_traits::Float;
use std::fmt::{Debug, Display};

use crate::isapprox::{NanComparison, Tols, isapprox};
use crate::uniquetol_1d::{Occurrence, sortperm, uniquetol_1d};

const SHAPE_ERR_MSG: &str = "Failed to reshape vector to ndarray";
const CONTIG_ERR_MSG: &str = "Array is not contiguous";

#[derive(Debug)]
pub struct AxisBoundsError {
    pub axis: usize,
    pub ndim: usize,
}

impl Display for AxisBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Axis {} is out of bounds for array with {} dimensions",
            self.axis, self.ndim
        )
    }
}

impl std::error::Error for AxisBoundsError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlattenAxis {
    #[default]
    None,
    Dim(usize),
}

fn uniquetol_groups<F>(
    group: &[usize],
    arr: &[F],
    tols: Tols<F>,
    nan_cmp: NanComparison,
) -> Vec<Vec<usize>>
where
    F: Float + Display + Debug,
{
    let perm_sorted = sortperm(arr, false);
    let mut groups = vec![vec![group[perm_sorted[0]]]];
    let mut curr = arr[perm_sorted[0]];

    for &idx in perm_sorted.iter().skip(1) {
        let next = arr[idx];

        match isapprox(curr, next, tols, nan_cmp) {
            // Safe to unwrap: groups is always initialized with one element
            true => groups.last_mut().unwrap().push(group[idx]),
            false => {
                groups.push(vec![group[idx]]);
                curr = next;
            }
        }
    }

    groups
}

#[inline]
fn uniquetol_nd_flatten_none<F>(
    arr: &Array<F, IxDyn>,
    tols: Tols<F>,
    nan_cmp: NanComparison,
    occurrence: Occurrence,
) -> Array<F, IxDyn>
where
    F: Float + Display + Debug,
{
    let arr_flat = arr.as_slice().expect(CONTIG_ERR_MSG);
    let arr_unique = uniquetol_1d(arr_flat, tols, nan_cmp, occurrence).arr_unique;
    let shape = IxDyn(&[arr_unique.len()]);
    Array::from_shape_vec(shape, arr_unique).expect(SHAPE_ERR_MSG)
}

fn uniquetol_nd_flatten_axis<F>(
    arr: &Array<F, IxDyn>,
    tols: Tols<F>,
    nan_cmp: NanComparison,
    occurrence: Occurrence,
    axis: usize,
) -> Array<F, IxDyn>
where
    F: Float + Display + Debug,
{
    let arr_flat: Vec<F> = arr
        .axis_iter(Axis(axis))
        .flat_map(|slice| slice.to_owned())
        .collect();

    let k = arr.len_of(Axis(axis));
    let n = arr.len() / k;
    let mut groups: Vec<Vec<usize>> = vec![(0..k).collect()];
    let mut groups_new = Vec::with_capacity(k);
    let mut sub_arr = Vec::with_capacity(n);

    for idx in 0..n {
        groups_new.clear();

        for group in groups.iter() {
            sub_arr.clear();
            sub_arr.extend(group.iter().map(|&i| arr_flat[i * n + idx]));
            groups_new.extend(uniquetol_groups(group, &sub_arr, tols, nan_cmp));
        }

        std::mem::swap(&mut groups, &mut groups_new);
    }

    let indices_unique: Vec<usize> = groups
        .iter()
        .map(|group| match occurrence {
            Occurrence::Lowest => group[0],
            Occurrence::Highest => group[group.len() - 1],
        })
        .collect();
    arr.select(Axis(axis), &indices_unique)
}

pub fn uniquetol_nd<F>(
    arr: &Array<F, IxDyn>,
    tols: Tols<F>,
    nan_cmp: NanComparison,
    occurrence: Occurrence,
    flatten_axis: FlattenAxis,
) -> Result<Array<F, IxDyn>, AxisBoundsError>
where
    F: Float + Display + Debug,
{
    match flatten_axis {
        FlattenAxis::None => Ok(uniquetol_nd_flatten_none(arr, tols, nan_cmp, occurrence)),
        FlattenAxis::Dim(axis) if axis < arr.ndim() => Ok(uniquetol_nd_flatten_axis(
            arr, tols, nan_cmp, occurrence, axis,
        )),
        FlattenAxis::Dim(axis) => Err(AxisBoundsError {
            axis,
            ndim: arr.ndim(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    const ARR_2D: [[f64; 3]; 4] = [
        [1.000000, 2.000000, -3.000001],
        [1.000001, 2.000001, -2.999997],
        [-4.300000, 1.999996, -0.000000],
        [1.000002, 2.000002, -2.999998],
    ];
    const SHAPE_2D: (usize, usize) = (4, 3);

    const ARR_3D: [[[f64; 5]; 4]; 2] = [
        [
            [1.000000, 2.000000, -3.000001, 2.000002, 1.999998],
            [1.000001, 2.000001, -2.999997, 2.000001, 1.999999],
            [-4.300000, 1.999996, 0.000000, 1.999994, 2.000002],
            [1.000002, -3.777777, -2.999998, -3.777774, -3.777771],
        ],
        [
            [-4.300000, 1.999996, 0.000000, 1.999994, 2.000002],
            [-4.299994, 2.000002, 0.000007, 1.999994, 2.000002],
            [1.000002, 2.000002, -3.000001, 2.000001, 1.999999],
            [1.000001, 2.000001, -2.999997, 2.000001, 1.999999],
        ],
    ];
    const SHAPE_3D: (usize, usize, usize) = (2, 4, 5);

    fn arr_2d() -> Array2<f64> {
        Array2::from_shape_vec(
            SHAPE_2D,
            ARR_2D
                .iter()
                .flat_map(|slice| slice.iter().copied())
                .collect(),
        )
        .expect(SHAPE_ERR_MSG)
    }

    fn arr_3d() -> Array3<f64> {
        Array3::from_shape_vec(
            SHAPE_3D,
            ARR_3D
                .iter()
                .flat_map(|slice| slice.iter().flat_map(|sub_slice| sub_slice.iter().copied()))
                .collect(),
        )
        .expect(SHAPE_ERR_MSG)
    }

    #[test]
    fn test_uniquetol_2d_none() {
        let arr = arr_2d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::default(),
            FlattenAxis::None,
        )
        .unwrap();
        let expected = array![-4.300000, -3.000001, 0.000000, 1.000000, 1.999996];
        assert_eq!(result, &expected.into_dyn());
    }

    #[test]
    fn test_uniquetol_2d_0() {
        let arr = arr_2d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::default(),
            FlattenAxis::Dim(0),
        )
        .unwrap();
        let expected = array![
            [-4.300000, 1.999996, -0.000000],
            [1.000000, 2.000000, -3.000001],
        ];
        assert_eq!(result, &expected.into_dyn());
    }

    #[test]
    fn test_uniquetol_2d_1() {
        let arr = arr_2d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::default(),
            FlattenAxis::Dim(1),
        )
        .unwrap();
        let expected = array![
            [-3.000001, 1.000000, 2.000000],
            [-2.999997, 1.000001, 2.000001],
            [0.000000, -4.300000, 1.999996],
            [-2.999998, 1.000002, 2.000002],
        ];
        assert_eq!(result, &expected.into_dyn());
    }

    #[test]
    fn test_uniquetol_3d_none() {
        let arr = arr_3d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::Highest,
            FlattenAxis::None,
        )
        .unwrap();
        let expected = array![
            2.000002, 1.000002, 0.000007, -2.999997, -3.777771, -4.299994
        ];
        assert_eq!(result, &expected.into_dyn());
    }

    #[test]
    fn test_uniquetol_3d_0() {
        let arr = arr_3d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::Highest,
            FlattenAxis::Dim(0),
        )
        .unwrap();
        let shape_expected: [usize; 3] = SHAPE_3D.into();
        assert_eq!(result.shape(), shape_expected);
        println!("result: {:?}", result);
    }

    #[test]
    fn test_uniquetol_3d_1() {
        let arr = arr_3d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::Highest,
            FlattenAxis::Dim(1),
        )
        .unwrap();
        let shape_expected = [2, 3, 5];
        assert_eq!(result.shape(), shape_expected);
        println!("result: {:?}", result);
    }

    #[test]
    fn test_uniquetol_3d_2() {
        let arr = arr_3d();
        let result = uniquetol_nd(
            &arr.into_dyn(),
            Tols {
                atol: 1e-5,
                rtol: 1e-2,
            },
            NanComparison::default(),
            Occurrence::Highest,
            FlattenAxis::Dim(2),
        )
        .unwrap();
        let shape_expected = [2, 4, 3];
        assert_eq!(result.shape(), shape_expected);
        println!("result: {:?}", result);
    }
}
