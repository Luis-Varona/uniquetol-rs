mod isapprox;
mod uniquetol_1d;
mod uniquetol_nd;
mod uniquetol_traits;

pub use isapprox::{NanComparison, Tols};
pub use uniquetol_1d::{Occurrence, UniqueTolArray};
pub use uniquetol_nd::FlattenAxis;
pub use uniquetol_traits::{UniqueTol1D, UniqueTolND};
