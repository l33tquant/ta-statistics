/// Paired statistics trait for calculating statistics on pairs of values
///
/// This trait provides a foundation for calculating various statistics on pairs of values,
/// such as covariance, correlation, and beta. It is used as a base trait for more specific
/// statistical calculations that examine relationships between two variables.
pub trait PairedStatistics<T> {
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
    fn next(&mut self, value: (T, T)) -> &mut Self
    where
        Self: Sized;

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
    /// use ta_statistics::{Statistics, PairedStatistics};
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = Statistics::new(3);
    /// let mut results = vec![];
    /// let inputs = [(2.0, 1.0), (4.0, 3.0), (6.0, 2.0), (8.0, 5.0), (10.0, 7.0)];
    /// inputs.iter().for_each(|i| {
    ///     stats.next(*i).cov().map(|v| results.push(v));
    /// });
    ///
    /// let expected: [f64; 3] = [0.6667, 1.3333, 3.3333];
    ///  for (i, e) in expected.iter().enumerate() {
    ///  assert_approx_eq!(e, results[i], 0.1);
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
    ///    assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn cov(&self) -> Option<T>;

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
    /// use ta_statistics::{Statistics, PairedStatistics};
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = Statistics::new(3);
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
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    ///
    /// ```
    fn corr(&self) -> Option<T>;

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
    /// use ta_statistics::{Statistics, PairedStatistics};
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let mut stats = Statistics::new(3);
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
    ///     assert_approx_eq!(e, results[i], 0.1);
    /// }
    /// ```
    fn beta(&self) -> Option<T>;
}
