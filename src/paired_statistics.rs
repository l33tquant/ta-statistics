use num_traits::Float;

use crate::{Kbn, RollingMoments};

/// A structure that computes various statistics over a fixed-size window of paired values.
///
/// `PairedStatistics<T>` maintains a circular buffer of paired values and computes statistical measures
/// such as covariance, correlation, beta, etc.
///
/// The structure automatically updates statistics as new values are added and old values
/// are removed from the window, making it efficient for rolling statistics analysis.
#[derive(Debug, Clone)]
pub struct PairedStatistics<T> {
    moments_x: RollingMoments<T>,
    moments_y: RollingMoments<T>,
    sum_xy: Kbn<T>,
    ddof: bool,
}

impl<T> PairedStatistics<T>
where
    T: Default + Clone + Float,
{
    /// Creates a new `PairedStatistics` instance with the specified period.
    ///
    /// # Arguments
    ///
    /// * `period` - The period of the statistics
    ///
    /// # Returns
    ///
    /// * `Self` - The `PairedStatistics` instance
    pub fn new(period: usize) -> Self {
        Self {
            moments_x: RollingMoments::new(period),
            moments_y: RollingMoments::new(period),
            sum_xy: Kbn::default(),
            ddof: false,
        }
    }

    /// Returns the period of the statistics
    ///
    /// # Returns
    ///
    /// * `usize` - The period of the statistics
    pub fn period(&self) -> usize {
        self.moments_x.period()
    }

    /// Resets the statistics
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The statistics object
    pub fn reset(&mut self) -> &mut Self {
        self.moments_x.reset();
        self.moments_y.reset();
        self.sum_xy = Default::default();
        self
    }

    /// Updates the paired statistical calculations with a new value pair in the time series
    ///
    /// Incorporates a new data point pair into the rolling window, maintaining the specified
    /// window size by removing the oldest pair when necessary. This core method provides
    /// the foundation for all paired statistical measures that examine relationships
    /// between two variables.
    ///
    /// # Arguments
    ///
    /// * `value` - A tuple containing the paired values (x, y) to incorporate into calculations
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The updated statistics object for method chaining
    pub fn next(&mut self, (x, y): (T, T)) -> &mut Self {
        self.moments_x.next(x);
        self.moments_y.next(y);

        if self.moments_x.is_ready() {
            if let Some((px, py)) = self.moments_x.popped().zip(self.moments_y.popped()) {
                self.sum_xy -= px * py;
            }
        }

        if let Some((vx, vy)) = self.moments_x.value().zip(self.moments_y.value()) {
            self.sum_xy += vx * vy;
        }

        self
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
    /// Returns the mean of the values in the rolling window
    ///
    /// # Returns
    ///
    /// * `Option<(T, T)>` - The mean of the values in the window, or `None` if the window is not full
    fn mean(&self) -> Option<(T, T)> {
        self.moments_x.mean().zip(self.moments_y.mean())
    }

    /// Returns the mean of the product of the values in the rolling window
    ///
    /// # Returns
    ///
    /// * `Option<(T, T)>` - The mean of the product of the values in the window, or `None` if the window is not full
    fn mean_prod(&self) -> Option<(T, T)> {
        if !self.moments_x.is_ready() {
            return None;
        }

        let n = T::from(self.period())?;
        let mp = self.sum_xy.total() / n;
        Some((mp, mp))
    }

    /// Returns the variance of the values in the rolling window
    ///
    /// # Returns
    ///
    /// * `Option<(T, T)>` - The variance of the values in the window, or `None` if the window is not full
    fn variance(&self) -> Option<(T, T)> {
        self.moments_x.variance().zip(self.moments_y.variance())
    }

    /// Returns the standard deviation of the values in the rolling window
    ///
    /// # Returns
    ///
    /// * `Option<(T, T)>` - The standard deviation of the values in the window, or `None` if the window is not full
    fn stddev(&self) -> Option<(T, T)> {
        self.moments_x.stddev().zip(self.moments_y.stddev())
    }

    /// Returns the covariance of the paired values in the rolling window
    ///
    /// Covariance measures how two variables change together, indicating the direction
    /// of their linear relationship. This fundamental measure of association provides:
    ///
    /// - Directional relationship analysis between paired time series
    /// - Foundation for correlation, beta, and regression calculations
    /// - Raw measurement of how variables move in tandem
    /// - Basis for portfolio diversification and risk assessments
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The covariance of the values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// use ta_statistics::PairedStatistics;
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = PairedStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [(2.0, 1.0), (4.0, 3.0), (6.0, 2.0), (8.0, 5.0), (10.0, 7.0)];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).cov().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [0.6667, 1.3333, 3.3333];
    ///  for (i, e) in expected.iter().enumerate() {
    ///  assert_approx_eq!(e, results[i], 0.001);
    ///  }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).cov().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [1.0, 2.0, 5.0];
    /// for (i, e) in expected.iter().enumerate() {
    ///    assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn cov(&self) -> Option<T> {
        let (mean_x, mean_y) = self.mean()?;
        let (mean_xy, _) = self.mean_prod()?;

        let cov = mean_xy - mean_x * mean_y;

        let n = T::from(self.period())?;
        if self.ddof() {
            Some(cov * (n / (n - T::one())))
        } else {
            Some(cov)
        }
    }

    /// Returns the correlation coefficient (Pearson's r) of paired values in the rolling window
    ///
    /// Correlation normalizes covariance by the product of standard deviations, producing
    /// a standardized measure of linear relationship strength between -1 and 1:
    ///
    /// - Quantifies the strength and direction of relationships between variables
    /// - Enables cross-pair comparison on a standardized scale
    /// - Provides the foundation for statistical arbitrage models
    /// - Identifies regime changes in intermarket relationships
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The correlation coefficient in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// use ta_statistics::PairedStatistics;
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = PairedStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [
    ///     (0.496714, 0.115991),
    ///     (-0.138264, -0.329650),
    ///     (0.647689, 0.574363),
    ///     (1.523030, 0.109481),
    ///     (-0.234153, -1.026366),
    ///     (-0.234137, -0.445040),
    ///     (1.579213, 0.599033),
    ///     (0.767435, 0.694328),
    ///     (-0.469474, -0.782644),
    ///     (0.542560, -0.326360)
    /// ];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).corr().map(|v| results.push(v));
    /// });
    /// let expected: [f64; 8] = [0.939464, 0.458316, 0.691218, 0.859137, 0.935658, 0.858379, 0.895148, 0.842302,];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// ```
    pub fn corr(&self) -> Option<T> {
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

    /// Returns the beta coefficient of the paired values in the rolling window
    ///
    /// Beta measures the relative volatility between two time series, indicating
    /// the sensitivity of one variable to changes in another:
    ///
    /// - Quantifies systematic risk exposure between related instruments
    /// - Determines optimal hedge ratios for risk management
    /// - Provides relative sensitivity analysis for pair relationships
    /// - Serves as a key input for factor modeling and attribution analysis
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The beta coefficient in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// use ta_statistics::PairedStatistics;
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = PairedStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [
    ///      (0.015, 0.010),
    ///      (0.025, 0.015),
    ///      (-0.010, -0.005),
    ///      (0.030, 0.020),
    ///      (0.005, 0.010),
    ///      (-0.015, -0.010),
    ///      (0.020, 0.015),
    /// ];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).beta().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 5] = [1.731, 1.643, 1.553, 1.429, 1.286];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.001);
    /// }
    /// ```
    pub fn beta(&self) -> Option<T> {
        self.cov().zip(self.variance()).and_then(
            |(cov, (_, var))| {
                if var.is_zero() { None } else { Some(cov / var) }
            },
        )
    }
}
