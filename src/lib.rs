mod isapprox;
mod uniquetol_1d;
mod uniquetol_nd;
mod uniquetol;

pub use isapprox::{Tols, EqualNan};
pub use uniquetol_1d::{Occurrence, UniqueTolArray};
pub use uniquetol_nd::FlattenAxis;
pub use uniquetol::{UniqueTol1D, UniqueTolND};