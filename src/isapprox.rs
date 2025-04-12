use num_traits::Float;
use std::fmt::{Debug, Display};

const ATOL_DEFAULT: f64 = 1e-8;
const ATOL_DEFAULT_ERR_MSG: &str = "Failed to create atol from default value";
const RTOL_DEFAULT_ERR_MSG: &str = "Failed to create rtol from epsilon";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NanComparison {
    #[default]
    Equal,
    NotEqual,
}

impl From<bool> for NanComparison {
    fn from(value: bool) -> Self {
        match value {
            true => NanComparison::Equal,
            false => NanComparison::NotEqual,
        }
    }
}

impl From<NanComparison> for bool {
    fn from(value: NanComparison) -> Self {
        matches!(value, NanComparison::Equal)
    }
}

#[derive(Debug)]
pub enum TolsError<F>
where
    F: Float + Display + Debug,
{
    NegativeAtol(F),
    NegativeRtol(F),
}

impl<F> Display for TolsError<F>
where
    F: Float + Display + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TolsError::NegativeAtol(value) => {
                write!(f, "atol must be non-negative, got {}", value)
            }
            TolsError::NegativeRtol(value) => {
                write!(f, "rtol must be non-negative, got {}", value)
            }
        }
    }
}

impl<F> std::error::Error for TolsError<F> where F: Float + Display + Debug {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tols<F>
where
    F: Float + Display + Debug,
{
    pub atol: F,
    pub rtol: F,
}

impl<F> Tols<F>
where
    F: Float + Display + Debug,
{
    pub fn new(atol: F, rtol: F) -> Result<Self, TolsError<F>> {
        if atol.is_sign_negative() {
            Err(TolsError::NegativeAtol(atol))
        } else if rtol.is_sign_negative() {
            Err(TolsError::NegativeRtol(rtol))
        } else {
            Ok(Tols { atol, rtol })
        }
    }
}

impl<F> Default for Tols<F>
where
    F: Float + Display + Debug,
{
    fn default() -> Self {
        Self {
            atol: F::from(ATOL_DEFAULT).expect(ATOL_DEFAULT_ERR_MSG),
            rtol: F::from(F::epsilon()).expect(RTOL_DEFAULT_ERR_MSG).sqrt(),
        }
    }
}

#[inline]
pub fn isapprox<F>(x: F, y: F, tols: Tols<F>, nan_cmp: NanComparison) -> bool
where
    F: Float + Display + Debug,
{
    if x.is_nan() && y.is_nan() {
        return nan_cmp.into();
    }

    if x.is_nan() || y.is_nan() {
        return false;
    }

    if x == y {
        return true;
    }

    let Tols { atol, rtol } = tols;
    let max_val = x.abs().max(y.abs());
    let tol = atol.max(rtol * max_val);
    (x - y).abs() <= tol
}
