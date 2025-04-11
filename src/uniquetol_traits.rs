use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn};

use crate::isapprox::{NanComparison, Tols};
use crate::uniquetol_1d::{Occurrence, UniqueTolArray, uniquetol_1d};
use crate::uniquetol_nd::{FlattenAxis, uniquetol_nd};

pub trait UniqueTol1D {
    fn uniquetol(
        &self,
        tols: Tols,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
    ) -> UniqueTolArray;
}

impl<T> UniqueTol1D for T
where
    T: AsRef<[f64]>,
{
    #[inline]
    fn uniquetol(
        &self,
        tols: Tols,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
    ) -> UniqueTolArray {
        uniquetol_1d(self.as_ref(), tols, nan_cmp, occurrence)
    }
}

pub trait UniqueTolND {
    fn uniquetol(
        &self,
        tols: Tols,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
        flatten_axis: FlattenAxis,
    ) -> Array<f64, IxDyn>;
}

impl<S, D> UniqueTolND for &ArrayBase<S, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    #[inline]
    fn uniquetol(
        &self,
        tols: Tols,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
        flatten_axis: FlattenAxis,
    ) -> Array<f64, IxDyn> {
        uniquetol_nd(
            &self.mapv(|x| x).into_dyn(),
            tols,
            nan_cmp,
            occurrence,
            flatten_axis,
        )
        .expect("Failed to compute unique values") // TODO: Make error message a const
    }
}
