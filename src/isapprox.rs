pub fn isapprox(x: f64, y: f64, tols: &Tols) -> bool {
    return f64::abs(x - y) <= f64::max(
        tols.atol, tols.rtol * f64::max(f64::abs(x), f64::abs(y))
    );
}

pub struct Tols {
    pub atol: f64,
    pub rtol: f64,
}

impl Default for Tols {
    fn default() -> Tols {
        Tols {
            atol: 1e-8,
            rtol: (f64::EPSILON).sqrt(),
        }
    }
}