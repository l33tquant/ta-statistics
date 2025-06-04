use num_traits::Float;

use crate::utils::{Min, MonotonicQueue};

/// # Minimum Value Calculation for Rolling Windows
///
/// A specialized module for tracking minimum values in rolling windows of financial time-series data.
///
/// Uses an efficient data structure to maintain the minimum value at all times, providing
/// constant-time lookups and amortized constant-time updates. This approach is particularly
/// valuable for technical analysis indicators that require identifying price extremes,
/// such as Donchian channels, volatility measures, and support/resistance level detection.
///
/// The implementation is optimized for financial time-series analysis where
/// identifying minimum values within specific lookback periods is essential
/// for decision-making processes.
#[derive(Debug)]
pub struct Minimum<T>(MonotonicQueue<T, Min>);

impl<T: Default + Clone + Float> Minimum<T> {
    /// Creates a new Minimum instance with the specified period
    ///
    /// # Arguments
    ///
    /// * `period` - The size of the rolling window
    ///
    /// # Returns
    ///
    /// A new Minimum instance
    pub fn new(period: usize) -> Self {
        Self(MonotonicQueue::new(period))
    }

    /// Pushes a new value into the rolling window
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to be added to the rolling window
    ///
    /// # Returns
    ///
    /// None if the window is not yet full, otherwise returns the minimum value
    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    /// Returns the minimum value in the rolling window
    ///
    /// # Returns
    ///
    /// None if the window is not yet full, otherwise returns the minimum value
    pub fn get(&self) -> Option<T> {
        self.0.front()
    }

    /// Resets the rolling window
    pub fn reset(&mut self) {
        self.0.reset();
    }
}
