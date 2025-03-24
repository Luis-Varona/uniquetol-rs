pub struct Tols {
    pub atol: f64,
    pub rtol: f64,
}

impl Tols {
    pub fn new (atol: f64, rtol: f64) -> Result<Self, &'static str> {
        if atol < 0.0 {
            Err("atol must be non-negative")
        } else if rtol < 0.0 {
            Err("rtol must be non-negative")
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
pub fn isapprox(x: f64, y: f64, tols: &Tols) -> bool {
    if x.is_nan() || y.is_nan() {
        return false;
    }
    
    let max_val = f64::max(f64::abs(x), f64::abs(y));
    let max_tol = f64::max(tols.atol, tols.rtol * max_val);
    f64::abs(x - y) <= max_tol
}