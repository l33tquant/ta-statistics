use num_traits::Float;

use crate::{Kbn, RingBuffer};

/// Rolling moments for online statistics
#[derive(Debug, Clone)]
pub struct RollingMoments<T> {
    period: usize,
    /// Ring buffer to maintain the window
    buf: RingBuffer<T>,
    /// Most recent value pushed into the rolling window.
    value: Option<T>,
    /// Most recent value popped out of the rolling window (if full).
    popped: Option<T>,
    ddof: bool,
    sum: Kbn<T>,
    sum_sq: Kbn<T>,
    sum_cube: Kbn<T>,
    sum_quad: Kbn<T>,
    mean: T,
    m2: T,
    m3: T,
    m4: T,
}

impl<T: Float + Default> RollingMoments<T> {
    /// Returns new instance for rolling moments
    pub fn new(period: usize) -> Self {
        Self {
            period,
            buf: RingBuffer::new(period),
            value: None,
            popped: None,
            ddof: false,
            sum: Kbn::default(),
            sum_sq: Kbn::default(),
            sum_cube: Kbn::default(),
            sum_quad: Kbn::default(),
            mean: T::zero(),
            m2: T::zero(),
            m3: T::zero(),
            m4: T::zero(),
        }
    }

    fn reset_sums(&mut self) {
        self.sum = Kbn::default();
        self.sum_sq = Kbn::default();
        self.sum_cube = Kbn::default();
        self.sum_quad = Kbn::default();
    }

    fn reset_moments(&mut self) {
        self.mean = T::zero();
        self.m2 = T::zero();
        self.m3 = T::zero();
        self.m4 = T::zero();
    }

    fn update_central_moments(&mut self) -> Option<()> {
        let n = T::from(self.buf.len()).unwrap_or(T::zero());
        if n == T::zero() {
            self.reset_moments();
            return None;
        }

        self.mean = self.sum.total() / n;

        let m1 = self.mean;
        let m2_raw = self.sum_sq.total() / n;
        let m3_raw = self.sum_cube.total() / n;
        let m4_raw = self.sum_quad.total() / n;

        let m1_sq = m1 * m1;
        let m1_cb = m1_sq * m1;
        let m1_4 = m1_cb * m1;

        let _2 = T::from(2.0)?;
        let _3 = T::from(3.0)?;
        let _4 = T::from(4.0)?;
        let _6 = T::from(6.0)?;
        self.m2 = m2_raw - m1_sq;
        self.m3 = m3_raw - _3 * m1 * m2_raw + _2 * m1_cb;
        self.m4 = m4_raw - _4 * m1 * m3_raw + _6 * m1_sq * m2_raw - _3 * m1_4;
        Some(())
    }

    /// Returns the Delta Degrees of Freedom
    ///
    /// # Returns
    ///
    /// * `bool` - The Delta Degrees of Freedom
    pub const fn ddof(&self) -> bool {
        self.ddof
    }

    /// Sets the Delta Degrees of Freedom
    ///
    /// # Arguments
    ///
    /// * `ddof` - The Delta Degrees of Freedom
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The statistics object
    pub const fn set_ddof(&mut self, ddof: bool) -> &mut Self {
        self.ddof = ddof;
        self
    }

    /// Resets the rolling moments
    pub fn reset(&mut self) -> &mut Self {
        self.buf.reset();
        self.value = None;
        self.popped = None;
        self.reset_sums();
        self.reset_moments();
        self
    }

    /// Updates the value to the rolling moments
    pub fn next(&mut self, value: T) -> &mut Self {
        self.value = Some(value);
        self.popped = self.buf.push(value);
        if let Some(popped) = self.popped {
            self.sum -= popped;
            self.sum_sq -= popped * popped;
            self.sum_cube -= popped * popped * popped;
            self.sum_quad -= popped * popped * popped * popped;
        }

        self.sum += value;
        self.sum_sq += value * value;
        self.sum_cube += value * value * value;
        self.sum_quad += value * value * value * value;

        self.update_central_moments();

        self
    }

    /// Recomputes the rolling statistics, could be called to avoid prolonged compounding of floating rounding errors
    pub fn recompute(&mut self) {
        self.reset_sums();

        for &v in self.buf.iter() {
            self.sum += v;
            self.sum_sq += v * v;
            self.sum_cube += v * v * v;
            self.sum_quad += v * v * v * v;
        }

        self.update_central_moments();
    }

    /// Returns the window period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Returns true of the calculation was ready
    pub fn is_ready(&self) -> bool {
        self.buf.is_full()
    }

    /// Returns the number of elements in the buffer
    pub fn count(&self) -> usize {
        self.buf.len()
    }

    /// Returns the sum of all values in the rolling window
    pub fn sum(&self) -> Option<T> {
        self.is_ready().then_some(self.sum.total())
    }

    /// Returns the sum of squares of all values in the rolling window
    pub fn sum_sq(&self) -> Option<T> {
        self.is_ready().then_some(self.sum_sq.total())
    }

    /// Returns the mean of all values in the rolling window
    pub fn mean(&self) -> Option<T> {
        self.is_ready().then_some(self.mean)
    }

    /// Returns the mean of squared values in the rolling window
    pub fn mean_sq(&self) -> Option<T> {
        self.sum_sq()
            .zip(T::from(self.count()))
            .map(|(ss, n)| ss / n)
    }

    /// Returns the rolling variance
    pub fn variance(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        let n = T::from(self.count())?;
        let denom = if self.ddof { n - T::one() } else { n };
        if denom > T::zero() {
            Some(self.m2 * n / denom)
        } else {
            None
        }
    }

    /// Returns the rolling standard deviation
    pub fn std_dev(&self) -> Option<T> {
        self.variance().and_then(|var| {
            if var >= T::zero() {
                Some(var.sqrt())
            } else {
                None
            }
        })
    }

    /// Returns the Zscore
    pub fn zscore(&self) -> Option<T> {
        let value = self.value?;
        let mean = self.mean()?;
        let std_dev = self.std_dev()?;

        if std_dev > T::zero() {
            Some((value - mean) / std_dev)
        } else {
            None
        }
    }

    /// Returns the skewness
    pub fn skew(&self) -> Option<T> {
        if !self.is_ready() || self.m2 <= T::zero() {
            return None;
        }

        let n = T::from(self.count())?;
        let m3 = self.m3;
        let m2 = self.m2;

        let denominator = m2 * m2.sqrt();
        if denominator <= T::zero() {
            return None;
        }
        let g1 = m3 / denominator;

        if self.ddof {
            if n <= T::from(2)? {
                return None;
            }
            let correction = (n * (n - T::one())).sqrt() / (n - T::from(2)?);
            Some(correction * g1)
        } else {
            Some(g1)
        }
    }

    /// Returns the kurtosis
    pub fn kurt(&self) -> Option<T>
    where
        T: core::fmt::Debug,
    {
        if !self.is_ready() || self.m2 <= T::zero() {
            return None;
        }

        let n = T::from(self.count())?;
        if n < T::from(4.0)? {
            return None;
        }

        let _1 = T::from(1.0)?;
        let _2 = T::from(2.0)?;
        let _3 = T::from(3.0)?;

        if !self.ddof {
            let kurt_pop = self.m4 / (self.m2 * self.m2);
            Some(kurt_pop - _3)
        } else {
            let sample_var = self.m2 * n / (n - _1);
            let numerator = n * n * (n + _1);
            let denominator = (n - _1) * (n - _2) * (n - _3);
            let correction = (_3 * (n - _1) * (n - _1)) / ((n - _2) * (n - _3));

            let g2 = (numerator / denominator) * (self.m4 / (sample_var * sample_var)) - correction;

            Some(g2)
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn sum_works() {
        let mut stats = RollingMoments::new(3);
        let inputs = [
            1_000_000.1,
            1_000_000.2,
            1_000_000.3,
            1_000_000.4,
            1_000_000.5,
            1_000_000.6,
            1_000_000.7,
        ];
        let mut results = vec![];

        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).sum() {
                results.push(v)
            }
        });

        let expected = [3000000.6, 3000000.9, 3000001.2, 3000001.5, 3000001.8];
        assert_eq!(&results, &expected);
    }

    #[test]
    fn sum_sq_works() {
        let mut stats = RollingMoments::new(3);
        let inputs = [
            100_000.1, 100_000.2, 100_000.3, 100_000.4, 100_000.5, 100_000.6, 100_000.7,
        ];
        let mut results = vec![];

        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).sum_sq() {
                results.push(v)
            }
        });
        let expected = [
            30000120000.14,
            30000180000.289997,
            30000240000.5,
            30000300000.769997,
            30000360001.1,
        ];
        assert_eq!(&results, &expected);
    }

    #[test]
    fn mean_works() {
        let mut stats = RollingMoments::new(3);
        let inputs = [
            1_000_000.1,
            1_000_000.2,
            1_000_000.3,
            1_000_000.4,
            1_000_000.5,
            1_000_000.6,
            1_000_000.7,
        ];
        let mut results = vec![];

        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).mean() {
                results.push(v)
            }
        });

        let expected: [f64; 5] = [1000000.2, 1000000.3, 1000000.4, 1000000.5, 1000000.6];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }

    #[test]
    fn mean_sq_works() {
        let mut stats = RollingMoments::new(3);
        let inputs = [
            1_000_000.1,
            1_000_000.2,
            1_000_000.3,
            1_000_000.4,
            1_000_000.5,
            1_000_000.6,
            1_000_000.7,
        ];
        let mut results = vec![];

        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).mean_sq() {
                results.push(v)
            }
        });

        let expected: [f64; 5] = [
            1000000400000.05,
            1000000600000.1,
            1000000800000.17,
            1000001000000.26,
            1000001200000.37,
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.01);
        }
    }

    #[test]
    fn variance_works() {
        let mut stats = RollingMoments::new(3);
        let mut results = vec![];
        let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).variance() {
                results.push(v)
            }
        });

        let expected: [f64; 7] = [0.1156, 0.0067, 0.0156, 0.0156, 0.0467, 0.0289, 0.0156];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }

        stats.reset().set_ddof(true);
        results = vec![];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).variance() {
                results.push(v)
            }
        });

        let expected: [f64; 7] = [0.1733, 0.01, 0.0233, 0.0233, 0.07, 0.0433, 0.0233];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }

    #[test]
    fn std_dev_works() {
        let mut stats = RollingMoments::new(3);
        let mut results = vec![];
        let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).std_dev() {
                results.push(v)
            }
        });

        let expected: [f64; 7] = [0.3399, 0.0816, 0.1247, 0.1247, 0.216, 0.17, 0.1247];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }

        stats.reset().set_ddof(true);
        results = vec![];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).std_dev() {
                results.push(v)
            }
        });

        let expected: [f64; 7] = [0.4163, 0.1, 0.1528, 0.1528, 0.2646, 0.2082, 0.1528];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }

    #[test]
    fn zscore_works() {
        let mut stats = RollingMoments::new(3);
        let mut results = vec![];
        let inputs = [1.2, -0.7, 3.4, 2.1, -1.5, 0.0, 2.2, -0.3, 1.5, -2.0];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).zscore() {
                results.push(v)
            }
        });

        let expected: [f64; 8] = [
            1.2535, 0.2923, -1.3671, -0.1355, 1.2943, -0.8374, 0.3482, -1.2129,
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }

        stats.reset().set_ddof(true);
        results = vec![];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).zscore() {
                results.push(v)
            }
        });

        let expected: [f64; 8] = [
            1.0235, 0.2386, -1.1162, -0.1106, 1.0568, -0.6837, 0.2843, -0.9903,
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }

    #[test]
    fn skew_works() {
        let mut stats = RollingMoments::new(4);
        let mut results = vec![];
        let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).skew() {
                results.push(v)
            }
        });

        let expected: [f64; 6] = [-0.9794, -0.4347, 0.0000, 0.2780, 0.0000, -0.3233];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }

        stats.reset().set_ddof(true);
        results = vec![];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).skew() {
                results.push(v)
            }
        });

        let expected: [f64; 6] = [-1.6964, -0.7528, 0.0000, 0.4816, 0.0000, -0.5600];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }

    #[test]
    fn kurt_works() {
        let mut stats = RollingMoments::new(4);
        let mut results = vec![];
        let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).kurt() {
                results.push(v)
            }
        });

        let expected: [f64; 6] = [-0.7981, -1.1543, -1.3600, -1.4266, -1.7785, -1.0763];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }

        stats.reset().set_ddof(true);
        results = vec![];
        inputs.iter().for_each(|i| {
            if let Some(v) = stats.next(*i).kurt() {
                results.push(v);
            }
        });

        let expected: [f64; 6] = [3.0144, 0.3429, -1.2, -1.6995, -4.3391, 0.928];
        for (i, e) in expected.iter().enumerate() {
            assert_approx_eq!(e, results[i], 0.0001);
        }
    }
}
