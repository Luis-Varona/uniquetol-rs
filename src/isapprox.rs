#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NanComparison {
    #[default]
    Equal,
    NotEqual,
}

impl From<bool> for NanComparison {
    fn from(value: bool) -> Self {
        if value {
            NanComparison::Equal
        } else {
            NanComparison::NotEqual
        }
    }
}

impl From<NanComparison> for bool {
    fn from(value: NanComparison) -> Self {
        matches!(value, NanComparison::Equal)
    }
}

#[derive(Debug)]
pub enum TolsError {
    NegativeAtol,
    NegativeRtol,
}

impl std::fmt::Display for TolsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TolsError::NegativeAtol => write!(f, "atol must be non-negative"),
            TolsError::NegativeRtol => write!(f, "rtol must be non-negative"),
        }
    }
}

impl std::error::Error for TolsError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tols {
    pub atol: f64,
    pub rtol: f64,
}

impl Tols {
    pub fn new(atol: f64, rtol: f64) -> Result<Self, TolsError> {
        if atol < 0.0 {
            Err(TolsError::NegativeAtol)
        } else if rtol < 0.0 {
            Err(TolsError::NegativeRtol)
        } else {
            Ok(Tols { atol, rtol })
        }
    }
}

impl Default for Tols {
    fn default() -> Self {
        Self {
            atol: 1e-8,
            rtol: (f64::EPSILON).sqrt(),
        }
    }
}

#[inline]
pub fn isapprox(x: f64, y: f64, tols: Tols, nan_cmp: NanComparison) -> bool {
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
