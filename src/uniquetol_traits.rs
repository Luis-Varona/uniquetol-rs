use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn};
use num_traits::Float;
use std::fmt::{Debug, Display};

use crate::isapprox::{NanComparison, Tols};
use crate::uniquetol_1d::{Occurrence, UniqueTolResult, uniquetol_1d};
use crate::uniquetol_nd::{FlattenAxis, uniquetol_nd};

const UNIQUETOL_ERR_MSG: &str = "Failed to compute unique values";

pub trait UniqueTol1D<F>
where
    F: Float + Display + Debug,
{
    fn uniquetol(
        &self,
        tols: Tols<F>,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
    ) -> UniqueTolResult<F>;
}

impl<A, F> UniqueTol1D<F> for A
where
    A: AsRef<[F]>,
    F: Float + Display + Debug,
{
    #[inline]
    fn uniquetol(
        &self,
        tols: Tols<F>,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
    ) -> UniqueTolResult<F> {
        uniquetol_1d(self, tols, nan_cmp, occurrence)
    }
}

pub trait UniqueTolND<F>
where
    F: Float + Display + Debug,
{
    fn uniquetol(
        &self,
        tols: Tols<F>,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
        flatten_axis: FlattenAxis,
    ) -> Array<F, IxDyn>;
}

impl<T, D, F> UniqueTolND<F> for &ArrayBase<T, D>
where
    T: Data<Elem = F>,
    F: Float + Display + Debug,
    D: Dimension,
{
    #[inline]
    fn uniquetol(
        &self,
        tols: Tols<F>,
        nan_cmp: NanComparison,
        occurrence: Occurrence,
        flatten_axis: FlattenAxis,
    ) -> Array<F, IxDyn> {
        uniquetol_nd(
            &self.mapv(|x| x).into_dyn(),
            tols,
            nan_cmp,
            occurrence,
            flatten_axis,
        )
        .expect(UNIQUETOL_ERR_MSG)
    }
}
