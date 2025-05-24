use num_traits::Float;

use core::iter::Sum;

use crate::{
    PairedStatistics, SingleStatistics, Window,
    helper::{median_from_sorted_slice, quantile_from_sorted_slice},
};

/// A structure that computes various statistics over a fixed-size window of values.
///
/// `Statistics<T>` maintains a circular buffer of values and computes statistical measures
/// such as mean, variance, standard deviation, median, etc. It can handle both single value
/// statistics and paired statistics (when `T` is a tuple).
///
/// The structure automatically updates statistics as new values are added and old values
/// are removed from the window, making it efficient for streaming data analysis.
///
/// # Type Parameters
///
/// * `T` - The type of values to compute statistics over. Can be a single numeric type
///   or a tuple for paired statistics.
#[derive(Debug, Clone)]
pub struct Statistics<T> {
    /// Fixed circular buffer
    buf: Window<T>,
    /// Fixed sorted buffer
    sorted_buf: Window<T>,
    /// Statistics period
    period: usize,
    /// Delta Degrees of Freedom
    ddof: bool,
    /// Latest updated value to statistics
    current_value: Option<T>,
    /// Previous value popped out the window, only available after full window
    popped: Option<T>,
    /// Sum of inputs
    sum: Option<T>,
    /// Sum of squares (used for variance calculation)
    sum_sq: Option<T>,
    /// Sum of product of x and y, used in paired statistics
    sum_prod: Option<T>,
    /// Current minimum value
    min: Option<T>,
    /// Current maximum value
    max: Option<T>,
    /// Maximum drawdown
    max_drawdown: Option<T>,
}

impl<T> Statistics<T> {
    /// Creates a new statistics object with the specified period
    ///
    /// # Arguments
    ///
    /// * `period` - The period of the statistics
    ///
    /// # Returns
    ///
    /// * `Statistics<T>` - The new statistics object
    pub fn new(period: usize) -> Self
    where
        T: Copy + Default,
    {
        Self {
            buf: Window::new(period),
            sorted_buf: Window::new(period),
            period,
            ddof: false,
            current_value: None,
            popped: None,
            sum: None,
            sum_sq: None,
            sum_prod: None,
            min: None,
            max: None,
            max_drawdown: None,
        }
    }

    /// Resets the statistics
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The statistics object
    pub fn reset(&mut self) -> &mut Self
    where
        T: Default + Copy,
    {
        self.buf.reset();
        self.sorted_buf.reset();
        self.ddof = false;
        self.current_value = None;
        self.popped = None;
        self.sum = None;
        self.sum_sq = None;
        self.sum_prod = None;
        self.min = None;
        self.max = None;
        self.max_drawdown = None;
        self
    }

    /// Returns the current number of elements in the buffer.
    ///
    /// # Returns
    ///
    /// * `usize` - The current number of elements in the buffer
    pub const fn len(&self) -> usize {
        self.buf.len()
    }

    /// Check if the statistics buffer is full
    ///
    /// # Returns
    ///
    /// * `bool` - True if the statistics buffer is full
    pub const fn is_full(&self) -> bool {
        self.buf.is_full()
    }

    /// Returns the period of the statistics
    ///
    /// # Returns
    ///
    /// * `usize` - The period of the statistics
    pub const fn period(&self) -> usize {
        self.period
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

    // Copies and sorts the buf
    fn sorted_buf(&mut self) -> &[T]
    where
        T: Copy + Default + PartialOrd,
    {
        self.sorted_buf.copy_from_slice(self.buf.as_slice());
        self.sorted_buf.sort()
    }

    fn period_t(&self) -> Option<T>
    where
        T: Float,
    {
        T::from(self.period)
    }
}

impl<T> Statistics<T> {}

impl<T: Float + Default + Sum> SingleStatistics<T> for Statistics<T> {
    fn next(&mut self, value: T) -> &mut Self {
        let popped = self.buf.next(value);
        self.current_value = Some(value);

        if self.is_full() {
            self.popped = match self.popped {
                None => (self.buf.index() > 0).then_some(popped),
                _ => Some(popped),
            };
        }

        self.sum = match self.sum {
            None => Some(value - popped),
            Some(s) => Some(s + value - popped),
        };

        self.sum_sq = match self.sum_sq {
            None => Some(value * value - popped * popped),
            Some(sq) => Some(sq + value * value - popped * popped),
        };

        self
    }
    fn sum(&self) -> Option<T> {
        if self.is_full() { self.sum } else { None }
    }

    fn sum_sq(&self) -> Option<T> {
        if self.is_full() { self.sum_sq } else { None }
    }

    fn mean(&self) -> Option<T> {
        self.sum().zip(self.period_t()).map(|(sum, n)| sum / n)
    }

    fn mean_sq(&self) -> Option<T> {
        self.sum_sq()
            .zip(self.period_t())
            .map(|(sum_sq, n)| sum_sq / n)
    }

    fn mode(&mut self) -> Option<T> {
        if !self.is_full() {
            return None;
        }

        let slice = self.sorted_buf();

        let mut mode = slice[0];
        let mut mode_count = 1;

        let mut current = slice[0];
        let mut current_count = 1;

        for &value in &slice[1..] {
            if value == current {
                current_count += 1;
            } else {
                if current_count > mode_count || (current_count == mode_count && current < mode) {
                    mode = current;
                    mode_count = current_count;
                }
                current = value;
                current_count = 1;
            }
        }

        if current_count > mode_count || (current_count == mode_count && current < mode) {
            mode = current;
        }

        Some(mode)
    }

    fn median(&mut self) -> Option<T> {
        self.is_full()
            .then_some(median_from_sorted_slice(self.sorted_buf()))
    }

    fn min(&mut self) -> Option<T> {
        if !self.is_full() {
            return None;
        }

        self.min = match self.min {
            None => self.buf.min(),
            Some(min) => {
                if self.popped == Some(min) {
                    self.buf.min()
                } else if self.current_value < Some(min) {
                    self.current_value
                } else {
                    Some(min)
                }
            }
        };
        self.min
    }

    fn max(&mut self) -> Option<T> {
        if !self.is_full() {
            return None;
        }

        self.max = match self.max {
            None => self.buf.max(),
            Some(max) => {
                if self.popped == Some(max) {
                    self.buf.max()
                } else if self.current_value > Some(max) {
                    self.current_value
                } else {
                    Some(max)
                }
            }
        };

        self.max
    }

    fn mean_absolute_deviation(&self) -> Option<T> {
        let mean = self.mean()?;
        let abs_sum = self.buf.iter().map(|&x| (x - mean).abs()).sum::<T>();
        self.period_t().map(|n| abs_sum / n)
    }

    fn median_absolute_deviation(&mut self) -> Option<T> {
        let median = self.median()?;

        self.sorted_buf
            .iter_mut()
            .zip(self.buf.as_slice())
            .for_each(|(dev, &x)| *dev = (x - median).abs());

        Some(median_from_sorted_slice(self.sorted_buf.sort()))
    }

    fn variance(&self) -> Option<T> {
        let variance = self
            .mean()
            .zip(self.mean_sq())
            .map(|(mean, mean_sq)| (mean_sq - (mean * mean)));

        if self.ddof() {
            variance
                .zip(self.period_t())
                .map(|(var, n)| var * (n / (n - T::one())))
        } else {
            variance
        }
    }

    fn stddev(&self) -> Option<T> {
        self.variance().map(T::sqrt)
    }

    fn zscore(&self) -> Option<T> {
        self.mean()
            .zip(self.stddev())
            .zip(self.current_value)
            .map(|((mean, stddev), x)| match stddev.abs() < T::epsilon() {
                true => T::zero(),
                _ => (x - mean) / stddev,
            })
    }

    fn skew(&self) -> Option<T> {
        let len = self.len();
        if self.ddof() && len < 3 {
            return None;
        }

        let (mean, stddev) = self.mean().zip(self.stddev())?;
        if stddev.abs() < T::epsilon() {
            return Some(T::zero());
        }

        let sum_cubed = self
            .buf
            .iter()
            .map(|&x| {
                let z = (x - mean) / stddev;
                z * z * z
            })
            .sum();

        let n = T::from(len)?;
        let _1 = T::one();
        let _2 = T::from(2.0)?;
        if self.ddof() {
            Some((n / ((n - _1) * (n - _2))) * sum_cubed)
        } else {
            Some(sum_cubed / n)
        }
    }

    fn kurt(&self) -> Option<T> {
        let len = self.len();
        if len < 4 {
            return None;
        }

        let (mean, stddev) = self.mean().zip(self.stddev())?;
        if stddev.abs() < T::epsilon() {
            return Some(T::zero());
        }

        let sum_fourth = self
            .buf
            .iter()
            .map(|&x| {
                let z = (x - mean) / stddev;
                z * z * z * z
            })
            .sum();

        let n = T::from(len)?;
        let _1 = T::one();
        let _2 = T::from(2)?;
        let _3 = T::from(3)?;

        let kurt = if self.ddof() {
            let numerator = n * (n + _1) * sum_fourth;
            let denominator = (n - _1) * (n - _2) * (n - _3);
            let correction = _3 * ((n - _1) * (n - _1)) / ((n - _2) * (n - _3));
            (numerator / denominator) - correction
        } else {
            sum_fourth / n - _3
        };

        Some(kurt)
    }

    fn linreg_slope(&self) -> Option<T> {
        if !self.is_full() {
            return None;
        }

        let mut s = Statistics::new(self.period);
        for (i, &x) in self.buf.iter().enumerate() {
            <Statistics<(T, T)> as PairedStatistics<T>>::next(&mut s, (x, T::from(i)?));
        }

        s.beta()
    }

    fn linreg_slope_intercept(&self) -> Option<(T, T)> {
        let (mean, slope) = self.mean().zip(self.linreg_slope())?;
        let _1 = T::one();
        self.period_t()
            .zip(T::from(2))
            .map(|(p, _2)| (p - _1) / _2)
            .map(|mt| (slope, mean - slope * mt))
    }

    fn linreg_intercept(&self) -> Option<T> {
        self.linreg_slope_intercept()
            .map(|(_, intercept)| intercept)
    }

    fn linreg_angle(&self) -> Option<T> {
        self.linreg_slope().map(|slope| slope.atan())
    }

    fn linreg(&self) -> Option<T> {
        let _1 = T::one();
        self.linreg_slope_intercept()
            .zip(self.period_t())
            .map(|((slope, intercept), period)| slope * (period - _1) + intercept)
    }

    fn drawdown(&mut self) -> Option<T> {
        self.max().zip(self.current_value).map(|(max, input)| {
            if max <= T::zero() || input <= T::zero() {
                T::zero()
            } else {
                ((max - input) / max).max(T::zero())
            }
        })
    }

    fn max_drawdown(&mut self) -> Option<T> {
        let drawdown = self.drawdown()?;
        self.max_drawdown = match self.max_drawdown {
            Some(md) => Some(md.max(drawdown)),
            None => Some(drawdown),
        };
        self.max_drawdown
    }

    fn diff(&self) -> Option<T> {
        self.current_value
            .zip(self.popped)
            .map(|(input, popped)| input - popped)
    }

    fn pct_change(&self) -> Option<T> {
        self.diff().zip(self.popped).and_then(|(diff, popped)| {
            if popped.is_zero() {
                None
            } else {
                Some(diff / popped)
            }
        })
    }

    fn log_return(&self) -> Option<T> {
        self.current_value
            .zip(self.popped)
            .and_then(|(current, popped)| {
                if popped <= T::zero() || current <= T::zero() {
                    None
                } else {
                    Some(current.ln() - popped.ln())
                }
            })
    }

    fn quantile(&mut self, q: f64) -> Option<T> {
        if !self.is_full() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        let period = self.period();
        let sorted = self.sorted_buf();
        quantile_from_sorted_slice(sorted, q, period)
    }

    fn iqr(&mut self) -> Option<T> {
        if !self.is_full() {
            return None;
        }

        let period = self.period();
        let sorted = self.sorted_buf();

        let q1 = quantile_from_sorted_slice(sorted, 0.25, period);
        let q3 = quantile_from_sorted_slice(sorted, 0.75, period);

        q1.zip(q3).map(|(q1, q3)| q3 - q1)
    }
}

impl<T: Float> Statistics<(T, T)> {
    fn mean(&self) -> Option<(T, T)> {
        if self.is_full() {
            let n = T::from(self.period)?;
            return self.sum.map(|(sum_x, sum_y)| (sum_x / n, sum_y / n));
        }
        None
    }

    fn mean_prod(&self) -> Option<(T, T)> {
        if self.is_full() {
            let n = T::from(self.period)?;
            return self.sum_prod.map(|(sum_xy, _)| (sum_xy / n, sum_xy / n));
        }
        None
    }

    fn mean_sq(&self) -> Option<(T, T)> {
        if self.is_full() {
            let n = T::from(self.period)?;
            return self
                .sum_sq
                .map(|(sum_sq_x, sum_sq_y)| (sum_sq_x / n, sum_sq_y / n));
        }
        None
    }

    fn variance(&self) -> Option<(T, T)> {
        let variance = self
            .mean()
            .zip(self.mean_sq())
            .map(|(mean, mean_sq)| (mean_sq.0 - (mean.0 * mean.0), mean_sq.1 - (mean.1 * mean.1)));

        if self.ddof() {
            variance
                .zip(T::from(self.period))
                .map(|(var, n)| (var.0 * (n / (n - T::one())), var.1 * (n / (n - T::one()))))
        } else {
            variance
        }
    }

    fn stddev(&self) -> Option<(T, T)> {
        self.variance().map(|var| (var.0.sqrt(), var.1.sqrt()))
    }
}

impl<T: Float + Default> PairedStatistics<T> for Statistics<(T, T)> {
    fn next(&mut self, (x, y): (T, T)) -> &mut Self {
        let popped = self.buf.next((x, y));
        self.current_value = Some((x, y));

        let x_diff = x - popped.0;
        let y_diff = y - popped.1;
        self.sum = match self.sum {
            None => Some((x_diff, y_diff)),
            Some((sx, sy)) => Some((sx + x_diff, sy + y_diff)),
        };

        let prod_diff = (x * y) - (popped.0 * popped.1);
        self.sum_prod = match self.sum_prod {
            None => Some((prod_diff, prod_diff)),
            Some((spx, spy)) => Some((spx + prod_diff, spy + prod_diff)),
        };

        let sq_x_diff = (x * x) - (popped.0 * popped.0);
        let sq_y_diff = (y * y) - (popped.1 * popped.1);
        self.sum_sq = match self.sum_sq {
            None => Some((sq_x_diff, sq_y_diff)),
            Some((ssx, ssy)) => Some((ssx + sq_x_diff, ssy + sq_y_diff)),
        };
        self
    }

    fn cov(&self) -> Option<T> {
        let (mean_x, mean_y) = self.mean()?;
        let (mean_xy, _) = self.mean_prod()?;

        let cov = mean_xy - mean_x * mean_y;

        let n = T::from(self.period)?;
        if self.ddof() {
            Some(cov * (n / (n - T::one())))
        } else {
            Some(cov)
        }
    }

    fn corr(&self) -> Option<T> {
        self.cov()
            .zip(self.stddev())
            .and_then(|(cov, (stddev_x, stddev_y))| {
                if stddev_x.is_zero() || stddev_y.is_zero() {
                    None
                } else {
                    Some(cov / (stddev_x * stddev_y))
                }
            })
    }

    fn beta(&self) -> Option<T> {
        self.cov().zip(self.variance()).and_then(
            |(cov, (_, var))| {
                if var.is_zero() { None } else { Some(cov / var) }
            },
        )
    }
}
