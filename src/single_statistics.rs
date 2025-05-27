use num_traits::Float;

use core::iter::Sum;

use crate::{
    PairedStatistics, RingBuffer, RollingMoments,
    helper::{median_from_sorted_slice, quantile_from_sorted_slice},
};

/// A structure that computes various statistics over a fixed-size window of values.
/// A specialized statistics implementation for single time-series data analysis.
///
/// This structure provides comprehensive statistical calculations for financial
/// time-series data, optimized for algorithmic trading applications. It maintains
/// a fixed-size window of values and efficiently updates statistics as new data
/// points arrive in a streaming fashion.
///
/// The structure is particularly useful for technical analysis, risk management,
/// and alpha generation in quantitative trading strategies.
#[derive(Debug, Clone)]
pub struct SingleStatistics<T> {
    /// Rolling moments
    moments: RollingMoments<T>,
    /// Fixed buffer for sorting on demand
    sorted_buf: RingBuffer<T>,
    /// Current minimum value
    min: Option<T>,
    /// Current maximum value
    max: Option<T>,
    /// Maximum drawdown
    max_drawdown: Option<T>,
}

impl<T> SingleStatistics<T>
where
    T: Default + Clone + Float,
{
    /// Creates a new `SingleStatistics` instance with the specified period.
    ///
    /// # Arguments
    ///
    /// * `period` - The period of the statistics
    ///
    /// # Returns
    ///
    /// * `Self` - The statistics object
    pub fn new(period: usize) -> Self {
        Self {
            moments: RollingMoments::new(period),
            sorted_buf: RingBuffer::new(period),
            min: None,
            max: None,
            max_drawdown: None,
        }
    }

    /// Returns the period of the statistics
    ///
    /// # Returns
    ///
    /// * `usize` - The period of the statistics
    pub fn period(&self) -> usize {
        self.moments.period()
    }

    /// Resets the statistics
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The statistics object
    pub fn reset(&mut self) -> &mut Self {
        self.moments.reset();
        self.sorted_buf.reset();
        self.min = None;
        self.max = None;
        self.max_drawdown = None;
        self
    }

    /// Recomputes the single statistics, could be called to avoid
    /// prolonged compounding of floating rounding errors
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The rolling moments object
    pub fn recompute(&mut self) -> &mut Self {
        self.moments.recompute();
        self
    }

    fn period_t(&self) -> Option<T>
    where
        T: Float,
    {
        T::from(self.period())
    }

    // Copies and sorts the buf
    fn sorted_buf(&mut self) -> &[T]
    where
        T: Copy + Default + PartialOrd,
    {
        self.sorted_buf.copy_from_slice(self.moments.as_slice());
        self.sorted_buf.sort()
    }

    /// Returns the Delta Degrees of Freedom
    ///
    /// # Returns
    ///
    /// * `bool` - The Delta Degrees of Freedom
    pub const fn ddof(&self) -> bool {
        self.moments.ddof()
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
        self.moments.set_ddof(ddof);
        self
    }

    /// Updates the statistical calculations with a new value in the time series
    ///
    /// Incorporates a new data point into the rolling window, maintaining the specified
    /// window size by removing the oldest value when necessary. This is the core method
    /// that should be called whenever new data is available for processing.
    ///
    /// The statistics are calculated using the Kahan-Babuška-Neumaier (Kbn) algorithm
    /// for numerically stable summation. This compensated summation technique minimizes
    /// floating-point errors that would otherwise accumulate in long-running calculations,
    /// particularly important for financial time-series analysis where precision is critical.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to be added to the time series
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The statistics object
    pub fn next(&mut self, value: T) -> &mut Self {
        self.moments.next(value);
        self
    }

    /// Returns the sum of all values in the rolling window
    ///
    /// This fundamental calculation serves as the basis for numerous higher-order statistics
    /// and provides key insights for:
    ///
    /// - Aggregating transaction volumes to identify participant interest levels
    /// - Constructing accumulation/distribution profiles across price action
    /// - Measuring net directional pressure in time series data
    /// - Quantifying capital flows between market segments
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The sum of all values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [1_000_000.1, 1_000_000.2, 1_000_000.3, 1_000_000.4, 1_000_000.5, 1_000_000.6, 1_000_000.7];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).sum().map(|v| results.push(v));
    /// });
    ///
    /// let expected = [3000000.6, 3000000.9, 3000001.2, 3000001.5, 3000001.8];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn sum(&self) -> Option<T> {
        self.moments.sum()
    }

    /// Returns the sum of squares of all values in the rolling window
    ///
    /// This calculation provides the sum of squared values in the series, offering insights into
    /// the magnitude of values regardless of sign:
    ///
    /// - Serves as a component for calculating variance and other dispersion measures
    /// - Emphasizes larger values due to the squaring operation
    /// - Provides power estimation in signal processing applications
    /// - Helps detect significant deviations from expected behavior
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The sum of squares in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [1_000_000.1, 1_000_000.2, 1_000_000.3, 1_000_000.4, 1_000_000.5, 1_000_000.6, 1_000_000.7];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).sum_sq().map(|v| results.push(v));
    /// });
    /// let expected = [3000001200000.14, 3000001800000.29, 3000002400000.5, 3000003000000.77, 3000003600001.0996];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn sum_sq(&self) -> Option<T> {
        self.moments.sum_sq()
    }

    /// Returns the arithmetic mean of all values in the rolling window
    ///
    /// This central tendency measure forms the foundation for numerous statistical calculations
    /// and serves as a reference point for analyzing data distributions:
    ///
    /// - Establishes equilibrium levels for reversion-based analytical models
    /// - Provides baseline reference for filtering noisy sequential data
    /// - Creates statistical foundation for pattern detection in time series
    /// - Enables feature normalization in advanced predictive modeling
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The arithmetic mean of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [1_000_000.1, 1_000_000.2, 1_000_000.3, 1_000_000.4, 1_000_000.5, 1_000_000.6, 1_000_000.7];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mean().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 5] = [1000000.2, 1000000.3, 1000000.4, 1000000.5, 1000000.6];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn mean(&self) -> Option<T> {
        self.moments.mean()
    }

    /// Returns the mean of squares of all values in the rolling window
    ///
    /// This calculation provides the average of squared values in the series, offering
    /// insights into the magnitude of values regardless of sign:
    ///
    /// - Serves as a component for calculating variance and other dispersion measures
    /// - Emphasizes larger values due to the squaring operation
    /// - Provides power estimation in signal processing applications
    /// - Helps detect significant deviations from expected behavior
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The mean of squared values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [1_000_000.1, 1_000_000.2, 1_000_000.3, 1_000_000.4, 1_000_000.5, 1_000_000.6, 1_000_000.7];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mean_sq().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 5] = [1000000400000.05,1000000600000.1,1000000800000.17,1000001000000.26,1000001200000.37];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.01);
    /// }
    /// ```
    pub fn mean_sq(&self) -> Option<T> {
        self.moments.mean_sq()
    }

    /// Returns the mode (most frequently occurring value) in the rolling window
    ///
    /// The mode identifies the most common value within a distribution, providing
    /// insight into clustering behavior and prevalent conditions:
    ///
    /// - Identifies common price points that may act as magnets or barriers
    /// - Detects clustering in volume or activity patterns
    /// - Provides non-parametric central tendency alternative to mean/median
    /// - Highlights dominant price levels where transactions concentrate
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The mode of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mode().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 8] = [1.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn mode(&mut self) -> Option<T> {
        if !self.moments.is_ready() {
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

    /// Returns the median (middle value) of the rolling window
    ///
    /// The median represents the central value when data is sorted, providing a robust
    /// measure of central tendency that's less affected by outliers than the mean:
    ///
    /// - Offers resilient central price estimation in volatile conditions
    /// - Establishes more stable reference points during extreme events
    /// - Provides core input for non-parametric statistical models
    /// - Creates baseline for interquartile range and other robust dispersion measures
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The median of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).median().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 5] = [5.0, 2.0, 7.0, 3.0, 7.0];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn median(&mut self) -> Option<T> {
        self.moments
            .is_ready()
            .then_some(median_from_sorted_slice(self.sorted_buf()))
    }

    /// Returns the minimum value in the rolling window
    ///
    /// The minimum value represents the lower bound of a data series over the observation
    /// period, providing key reference points for analysis and decision-making:
    ///
    /// - Establishes potential support levels in price-based analysis
    /// - Identifies optimal entry thresholds for mean-reverting sequences
    /// - Sets critical risk boundaries for position management systems
    /// - Provides baseline scenarios for stress-testing and risk modeling
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The minimum value in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).min().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [25.4, 26.0, 25.8, 25.8, 25.8, 25.9, 26.2];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn min(&mut self) -> Option<T> {
        if !self.moments.is_ready() {
            return None;
        }

        self.min = match self.min {
            None => self.moments.min(),
            Some(min) => {
                if self.moments.popped() == Some(min) {
                    self.moments.min()
                } else if self.moments.value() < Some(min) {
                    self.moments.value()
                } else {
                    Some(min)
                }
            }
        };
        self.min
    }

    /// Returns the maximum value in the rolling window
    ///
    /// The maximum value identifies the upper bound of a data series over the observation
    /// period, establishing crucial reference points for analytical frameworks:
    ///
    /// - Identifies potential resistance zones in technical analysis
    /// - Optimizes profit-taking thresholds based on historical precedent
    /// - Confirms genuine breakouts from established trading ranges
    /// - Defines upper boundaries for range-bound trading approaches
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The maximum value in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).max().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [26.2, 26.2, 26.1, 26.1, 26.3, 26.3, 26.5];
    /// assert_eq!(&results, &expected);
    /// ```
    pub fn max(&mut self) -> Option<T> {
        if !self.moments.is_ready() {
            return None;
        }

        self.max = match self.max {
            None => self.moments.max(),
            Some(max) => {
                if self.moments.popped() == Some(max) {
                    self.moments.max()
                } else if self.moments.value() > Some(max) {
                    self.moments.value()
                } else {
                    Some(max)
                }
            }
        };

        self.max
    }

    /// Returns the mean absolute deviation of values in the rolling window
    ///
    /// This robust dispersion measure calculates the average absolute difference from the mean,
    /// offering advantages over variance in certain analytical contexts:
    ///
    /// - Provides volatility measurement that's less sensitive to extreme outliers
    /// - Quantifies data noise levels to enhance signal processing accuracy
    /// - Complements standard risk models with an alternative dispersion metric
    /// - Establishes more stable thresholds for adaptive signal generation
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The mean absolute deviation of values, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [2.0, 4.0, 6.0, 8.0, 10.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mean_absolute_deviation().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [1.3333, 1.3333, 1.3333];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn mean_absolute_deviation(&self) -> Option<T>
    where
        T: Sum,
    {
        let mean = self.mean()?;
        let abs_sum = self.moments.iter().map(|&x| (x - mean).abs()).sum::<T>();
        self.period_t().map(|n| abs_sum / n)
    }

    /// Returns the median absolute deviation of values in the rolling window
    ///
    /// This exceptionally robust dispersion measure calculates the median of absolute differences
    /// from the median, offering superior resistance to outliers:
    ///
    /// - Provides reliable volatility assessment in erratic or noisy environments
    /// - Serves as a foundation for robust anomaly detection systems
    /// - Enables stable threshold calibration for adaptive decision systems
    /// - Forms basis for robust statistical estimators in non-normal distributions
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The median absolute deviation of values, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 9.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).median_absolute_deviation().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 5] = [3.0, 1.0, 1.0, 2.0, 2.0];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn median_absolute_deviation(&mut self) -> Option<T> {
        let median = self.median()?;
        self.sorted_buf
            .iter_mut()
            .zip(self.moments.as_slice())
            .for_each(|(dev, &x)| *dev = (x - median).abs());

        Some(median_from_sorted_slice(self.sorted_buf.sort()))
    }

    /// Returns the variance of values in the rolling window
    ///
    /// This second-moment statistical measure quantifies dispersion around the mean
    /// and serves multiple analytical purposes:
    ///
    /// - Providing core risk assessment metrics for position sizing decisions
    /// - Enabling volatility regime detection to adapt methodologies appropriately
    /// - Filtering signal noise to improve discriminatory power
    /// - Identifying dispersion-based opportunities in related instrument groups
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The variance of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).variance().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.1156, 0.0067, 0.0156, 0.0156, 0.0467, 0.0289, 0.0156];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).variance().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.1733, 0.01, 0.0233, 0.0233, 0.07, 0.0433, 0.0233];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn variance(&self) -> Option<T> {
        self.moments.variance()
    }

    /// Returns the standard deviation of values in the rolling window
    ///
    /// As the square root of variance, this statistic provides an intuitive measure
    /// of data dispersion in the original units and enables:
    ///
    /// - Setting dynamic volatility thresholds for risk boundaries
    /// - Detecting potential mean-reversion opportunities when values deviate significantly
    /// - Normalizing position sizing across different volatility environments
    /// - Identifying market regime changes to adapt strategic approaches
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The standard deviation of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).stddev().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.3399, 0.0816, 0.1247, 0.1247, 0.216, 0.17, 0.1247];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).stddev().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.4163, 0.1, 0.1528, 0.1528, 0.2646, 0.2082, 0.1528];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn stddev(&self) -> Option<T> {
        self.moments.stddev()
    }

    /// Returns the z-score of the most recent value relative to the rolling window
    ///
    /// Z-scores express how many standard deviations a value deviates from the mean,
    /// providing a normalized measure that facilitates:
    ///
    /// - Statistical arbitrage through relative valuation in correlated series
    /// - Robust outlier detection across varying market conditions
    /// - Cross-instrument comparisons on a standardized scale
    /// - Setting consistent thresholds that remain valid across changing volatility regimes
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The z-score of the most recent value, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [1.2, -0.7, 3.4, 2.1, -1.5, 0.0, 2.2, -0.3, 1.5, -2.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).zscore().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 8] = [1.2535, 0.2923, -1.3671, -0.1355, 1.2943, -0.8374, 0.3482, -1.2129];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).zscore().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 8] = [1.0235, 0.2386, -1.1162, -0.1106, 1.0568, -0.6837, 0.2843, -0.9903];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn zscore(&self) -> Option<T> {
        self.moments.zscore()
    }

    /// Returns the skewness of values in the rolling window
    ///
    /// This third-moment statistic measures distribution asymmetry, revealing whether
    /// extreme values tend toward one direction. A comprehensive analysis of skewness:
    ///
    /// - Detects asymmetric return distributions critical for accurate risk modeling
    /// - Reveals directional biases in market microstructure that may predict future movements
    /// - Provides early signals of potential regime transitions in volatile environments
    /// - Enables refined models for derivative pricing beyond simple volatility assumptions
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The skewness of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(4);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).skew().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] =  [-0.97941, -0.43465, 0.0, 0.27803, 0.0, -0.32332];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).skew().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [-1.69639, -0.75284, 0.0, 0.48156, 0.0, -0.56];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn skew(&self) -> Option<T> {
        self.moments.skew()
    }

    /// Returns the kurtosis of values in the rolling window
    ///
    /// This fourth-moment statistic measures the 'tailedness' of a distribution, describing
    /// the frequency of extreme values compared to a normal distribution:
    ///
    /// - Quantifies fat-tail risk exposure essential for anticipating extreme market movements
    /// - Signals potentially exploitable market inefficiencies through distribution analysis
    /// - Provides critical parameters for selecting appropriate derivatives strategies
    /// - Enhances Value-at-Risk models by incorporating more realistic tail behavior
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The kurtosis of values in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(4);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).kurt().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [-0.7981, -1.1543, -1.3600, -1.4266, -1.7785, -1.0763];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.0001);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).kurt().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [3.0144, 0.3429, -1.2, -1.6995, -4.3391, 0.928];
    /// for (i, e) in expected.iter().enumerate() {
    ///   assert_approx_eq!(e, results[i], 0.0001);
    /// }
    /// ```
    pub fn kurt(&self) -> Option<T> {
        self.moments.kurt()
    }

    /// Returns the slope of the linear regression line
    ///
    /// The regression slope represents the rate of change in the best-fit linear model,
    /// quantifying the directional movement and its magnitude within the data:
    ///
    /// - Provides precise measurement of trend strength and conviction
    /// - Quantifies velocity of change for optimal timing decisions
    /// - Signals potential reversion points when diverging from historical patterns
    /// - Measures the relative imbalance between supply and demand forces
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The slope of the linear regression line, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(5);
    /// let mut results = vec![];
    /// let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).linreg_slope().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [0.34, 0.31, 0.32, 0.32, 0.08, 0.07];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    pub fn linreg_slope(&self) -> Option<T> {
        if !self.moments.is_ready() {
            return None;
        }

        let mut s = PairedStatistics::new(self.period());
        for (i, &x) in self.moments.iter().enumerate() {
            s.next((x, T::from(i)?));
        }

        s.beta()
    }

    /// Returns both slope and intercept of the linear regression line
    ///
    /// This comprehensive regression analysis provides the complete linear model,
    /// enabling more sophisticated trend-based calculations:
    ///
    /// - Constructs complete linear models of price or indicator evolution
    /// - Determines both direction and reference level in a single calculation
    /// - Enables advanced divergence analysis against actual values
    /// - Provides foundation for channel-based analytical frameworks
    ///
    /// # Returns
    ///
    /// * `Option<(T, T)>` - A tuple containing (slope, intercept), or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(5);
    /// let mut ddof = false;
    /// let mut results = vec![];
    /// let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).linreg_slope_intercept().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [(f64, f64); 6] = [
    ///     (0.34, 10.14),
    ///     (0.31, 10.58),
    ///     (0.32, 10.92),
    ///     (0.32, 11.1),
    ///     (0.08, 11.76),
    ///     (0.07, 11.98),
    /// ];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e.0, results[i].0, 0.1);
    ///     assert_approx_eq!(e.1, results[i].1, 0.1);
    /// }
    /// ```
    pub fn linreg_slope_intercept(&self) -> Option<(T, T)> {
        let (mean, slope) = self.mean().zip(self.linreg_slope())?;
        let _1 = T::one();
        self.period_t()
            .zip(T::from(2))
            .map(|(p, _2)| (p - _1) / _2)
            .map(|mt| (slope, mean - slope * mt))
    }

    /// Returns the y-intercept of the linear regression line
    ///
    /// The regression intercept represents the base level or starting point of the
    /// best-fit linear model, providing key reference information:
    ///
    /// - Establishes the theoretical zero-point reference level
    /// - Complements slope calculations to complete linear projections
    /// - Assists in fair value determination for mean-reversion models
    /// - Provides a fixed component for decomposing price into trend and oscillation
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The y-intercept of the regression line, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(5);
    /// let mut results = vec![];
    /// let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).linreg_intercept().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [10.14, 10.58, 10.92, 11.1, 11.76, 11.98];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn linreg_intercept(&self) -> Option<T> {
        self.linreg_slope_intercept()
            .map(|(_, intercept)| intercept)
    }

    /// Returns the angle (in degrees) of the linear regression line
    ///
    /// The regression angle converts the slope into degrees, providing a more intuitive
    /// measure of trend inclination that's bounded between -90 and 90 degrees:
    ///
    /// - Offers an easily interpretable measure of trend strength
    /// - Provides normalized measurement across different scaling contexts
    /// - Enables clear categorization of trend intensity
    /// - Simplifies visual representation of directional movement
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The angle of the regression line in degrees, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(5);
    /// let mut ddof = false;
    /// let mut results = vec![];
    /// let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).linreg_angle().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [0.3396, 0.3100, 0.3199, 0.3199, 0.0799, 0.0699];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn linreg_angle(&self) -> Option<T> {
        self.linreg_slope().map(|slope| slope.atan())
    }

    /// Returns the linear regression value (predicted y) for the last position
    ///
    /// This calculation provides the expected value at the current position according to
    /// the best-fit linear model across the window period:
    ///
    /// - Establishes theoretical fair value targets for mean-reversion analysis
    /// - Projects trend trajectory for momentum-based methodologies
    /// - Filters cyclical noise to extract underlying directional bias
    /// - Provides basis for divergence analysis between actual and expected values
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The predicted value at the current position, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(5);
    /// let mut ddof = false;
    /// let mut results = vec![];
    /// let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).linreg().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [11.5, 11.82, 12.2, 12.38, 12.08, 12.26];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn linreg(&self) -> Option<T> {
        let _1 = T::one();
        self.linreg_slope_intercept()
            .zip(self.period_t())
            .map(|((slope, intercept), period)| slope * (period - _1) + intercept)
    }

    /// Returns the current drawdown from peak
    ///
    /// Measures the percentage decline from the highest observed value to the current value,
    /// providing crucial insights for risk management and performance evaluation:
    ///
    /// - Enables dynamic adjustment of risk exposure during challenging conditions
    /// - Facilitates strategy rotation based on relative performance metrics
    /// - Forms the foundation of capital preservation systems during market stress
    /// - Identifies potential opportunities for strategic positioning during dislocations
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The current drawdown from peak, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 110.0, 105.0, 115.0, 100.0, 95.0, 105.0, 110.0, 100.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).drawdown().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.045, 0.0, 0.13, 0.174, 0.0, 0.0, 0.091];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    pub fn drawdown(&mut self) -> Option<T> {
        self.max().zip(self.moments.value()).map(|(max, input)| {
            if max <= T::zero() || input <= T::zero() {
                T::zero()
            } else {
                ((max - input) / max).max(T::zero())
            }
        })
    }

    /// Returns the maximum drawdown in the window
    ///
    /// Maximum drawdown measures the largest peak-to-trough decline within a time series,
    /// serving as a foundational metric for risk assessment and strategy evaluation:
    ///
    /// - Establishes critical constraints for comprehensive risk management frameworks
    /// - Provides an objective metric for evaluating strategy viability under stress
    /// - Informs position sizing parameters to maintain proportional risk exposure
    /// - Contributes valuable input to market regime classification models
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The maximum drawdown in the window, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 110.0, 105.0, 115.0, 100.0, 95.0, 105.0, 110.0, 100.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).max_drawdown().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.045, 0.045, 0.13, 0.174, 0.174, 0.174, 0.174];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn max_drawdown(&mut self) -> Option<T> {
        let drawdown = self.drawdown()?;
        self.max_drawdown = match self.max_drawdown {
            Some(md) => Some(md.max(drawdown)),
            None => Some(drawdown),
        };
        self.max_drawdown
    }

    /// Returns the difference between the last and first values
    ///
    /// This fundamental calculation of absolute change between two points provides
    /// essential directional and magnitude information for time series analysis:
    ///
    /// - Enables momentum measurement for trend strength evaluation
    /// - Quantifies rate-of-change to optimize timing decisions
    /// - Serves as a building block for pattern recognition in sequential data
    /// - Provides critical inputs for calculating hedge ratios and exposure management
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The difference between values, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 102.0, 105.0, 101.0, 98.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).diff().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 2] = [1.0, -4.0];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn diff(&self) -> Option<T> {
        self.moments
            .value()
            .zip(self.moments.popped())
            .map(|(input, popped)| input - popped)
    }

    /// Returns the percentage change between the first and last values
    ///
    /// Percentage change normalizes absolute changes by the starting value, enabling
    /// meaningful comparisons across different scales and measurement contexts:
    ///
    /// - Facilitates cross-asset performance comparison for relative strength analysis
    /// - Provides risk-normalized return metrics that account for initial exposure
    /// - Enables position sizing that properly adjusts for varying volatility environments
    /// - Serves as a key input for comparative performance evaluation across related groups
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The percentage change, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 105.0, 103.0, 106.0, 110.0, 108.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).pct_change().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [0.06, 0.04761905, 0.04854369];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn pct_change(&self) -> Option<T> {
        self.diff()
            .zip(self.moments.popped())
            .and_then(|(diff, popped)| {
                if popped.is_zero() {
                    None
                } else {
                    Some(diff / popped)
                }
            })
    }

    /// Returns the logarithmic return between the first and last values
    ///
    /// Logarithmic returns (continuous returns) offer mathematical advantages over simple
    /// returns, particularly for time series analysis:
    ///
    /// - Provides time-additive metrics that can be properly aggregated across periods
    /// - Normalizes signals in relative-value analysis of related securities
    /// - Creates a more consistent volatility scale regardless of price levels
    /// - Improves accuracy in long-horizon analyses through proper handling of compounding
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The logarithmic return, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 105.0, 103.0, 106.0, 110.0, 108.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).log_return().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [0.05827, 0.04652, 0.04727];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn log_return(&self) -> Option<T> {
        self.moments
            .value()
            .zip(self.moments.popped())
            .and_then(|(current, popped)| {
                if popped <= T::zero() || current <= T::zero() {
                    None
                } else {
                    Some(current.ln() - popped.ln())
                }
            })
    }

    /// Returns the quantile of the values in the window
    ///
    /// # Arguments
    ///
    /// * `q` - The quantile to calculate
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The quantile, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [10.0, 20.0, 30.0, 40.0, 50.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).quantile(0.5).map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [20.0, 30.0, 40.0];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    pub fn quantile(&mut self, q: f64) -> Option<T> {
        if !self.moments.is_ready() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        let period = self.period();
        let sorted = self.sorted_buf();
        quantile_from_sorted_slice(sorted, q, period)
    }

    /// Returns the interquartile range of the values in the window
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The interquartile range, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::SingleStatistics;
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = SingleStatistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [10.0, 20.0, 30.0, 40.0, 50.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).iqr().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [10.0, 10.0, 10.0];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    pub fn iqr(&mut self) -> Option<T> {
        if !self.moments.is_ready() {
            return None;
        }

        let period = self.period();
        let sorted = self.sorted_buf();

        let q1 = quantile_from_sorted_slice(sorted, 0.25, period);
        let q3 = quantile_from_sorted_slice(sorted, 0.75, period);

        q1.zip(q3).map(|(q1, q3)| q3 - q1)
    }
}
