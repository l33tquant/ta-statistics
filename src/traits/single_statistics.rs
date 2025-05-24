/// Single statistics trait for calculating statistics on a single value
///
/// This trait provides a foundation for calculating various statistics on a single value,
/// such as mean, variance, standard deviation, and other measures of central tendency and
/// dispersion. It is used as a base trait for more specific statistical calculations that
/// examine relationships between variables.
pub trait SingleStatistics<T> {
    /// Updates the statistical calculations with a new value in the time series
    ///
    /// Incorporates a new data point into the rolling window, maintaining the specified
    /// window size by removing the oldest value when necessary. This is the core method
    /// that should be called whenever new data is available for processing.
    fn next(&mut self, value: T) -> &mut Self
    where
        Self: Sized;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// let mut stats = Statistics::new(3);
    /// let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).sum().map(|v| results.push(v));
    /// });
    ///
    /// let expected = [6.0, 9.0, 12.0, 15.0];
    /// assert_eq!(&results, &expected);
    /// ```
    fn sum(&self) -> Option<T>;

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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// let mut stats = Statistics::new(3);
    /// let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).sum_sq().map(|v| results.push(v));
    /// });
    ///
    /// let expected = [14.0, 29.0, 50.0, 77.0];
    /// assert_eq!(&results, &expected);
    /// ```
    fn sum_sq(&self) -> Option<T>;

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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// let mut stats = Statistics::new(3);
    /// let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mean().map(|v| results.push(v));
    /// });
    ///
    /// let expected = [2.0, 3.0, 4.0, 5.0];
    /// assert_eq!(&results, &expected);
    /// ```
    fn mean(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
    /// let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut results = vec![];
    ///
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).mean_sq().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 4] = [4.66, 9.66, 16.66, 25.66];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn mean_sq(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn mode(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn median(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn min(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn max(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn mean_absolute_deviation(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn median_absolute_deviation(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).variance().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.1156, 0.0067, 0.0156, 0.0156, 0.0422, 0.0289, 0.0156];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).variance().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.1733, 0.0100, 0.0233, 0.0233, 0.0633, 0.0433, 0.0233];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn variance(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).stddev().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.3400, 0.0816, 0.1249, 0.1249, 0.2054, 0.1700, 0.1249];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).stddev().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 7] = [0.4162, 0.1000, 0.1527, 0.1527, 0.2516, 0.2081, 0.1527];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn stddev(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [1.2, -0.7, 3.4, 2.1, -1.5, 0.0, 2.2, -0.3, 1.5, -2.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).zscore().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 8] = [1.253, 0.292, -1.366, -0.136, 1.294, -0.835, 0.349, -1.210];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).zscore().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 8] = [1.024, 0.239, -1.116, -0.103, 1.000, -0.686, 0.285, -0.990];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn zscore(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(4);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).skew().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] =  [-0.979, -0.435, -0.0, 0.278, -0.0, -0.323];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).skew().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [-1.696, -0.753, 0.000, 0.482, 0.000, -0.560];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn skew(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(4);
    /// let mut results = vec![];
    /// let inputs = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).kurt().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [-0.7981, -1.1543, -1.3600, -1.4266, -1.7785, -1.0763];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// stats.reset().set_ddof(true);
    /// results = vec![];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).kurt().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 6] = [3.014, 0.343, -1.200, -1.700, -4.339, 0.928];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn kurt(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(5);
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
    fn linreg_slope(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(5);
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
    fn linreg_slope_intercept(&self) -> Option<(T, T)>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(5);
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
    fn linreg_intercept(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(5);
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
    fn linreg_angle(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(5);
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
    fn linreg(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn drawdown(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn max_drawdown(&mut self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn diff(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn pct_change(&self) -> Option<T>;
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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [100.0, 105.0, 103.0, 106.0, 110.0, 108.0];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).pct_change().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [0.05827, 0.04652, 0.04727];
    /// for (i, e) in expected.iter().enumerate() {
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    fn log_return(&self) -> Option<T>;

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
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn quantile(&mut self, q: f64) -> Option<T>;

    /// Returns the interquartile range of the values in the window
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The interquartile range, or `None` if the window is not full
    ///
    /// # Examples
    ///
    /// ```
    /// # use ta_statistics::{Statistics, SingleStatistics};
    /// # use assert_approx_eq::assert_approx_eq;
    /// let mut stats = Statistics::new(3);
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
    fn iqr(&mut self) -> Option<T>;
}
