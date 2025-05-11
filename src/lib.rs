// Copyright 2025 Luis M. B. Varona
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod isapprox;
mod uniquetol_1d;
mod uniquetol_nd;
mod uniquetol_traits;

pub use isapprox::{NanComparison, Tols};
pub use uniquetol_1d::{Occurrence, UniqueTolResult};
pub use uniquetol_nd::FlattenAxis;
pub use uniquetol_traits::{UniqueTol1D, UniqueTolND};
