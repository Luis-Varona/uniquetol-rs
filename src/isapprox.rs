#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EqualNan {
    #[default]
    Yes,
    No,
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
    pub fn new (atol: f64, rtol: f64) -> Result<Self, TolsError> {
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
        Self { atol: 1e-8, rtol: (f64::EPSILON).sqrt() }
    }
}

#[inline]
pub fn isapprox(x: f64, y: f64, tols: Tols, equal_nan: EqualNan) -> bool {
    if x.is_nan() && y.is_nan() {
        return match equal_nan {
            EqualNan::Yes => true,
            EqualNan::No => false,
        };
    }
    
    if x.is_nan() || y.is_nan() {
        return false;
    }
    
    let max_val = f64::max(f64::abs(x), f64::abs(y));
    let max_tol = f64::max(tols.atol, tols.rtol * max_val).min(f64::MAX);
    f64::abs(x - y) <= max_tol
}