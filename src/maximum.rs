use num_traits::Float;

use crate::utils::{Max, MonotonicQueue};

/// # Maximum Value Calculation for Rolling Windows
///
/// A specialized module for tracking maximum values in rolling windows of financial time-series data.
///
/// Uses an efficient data structure to maintain the maximum value at all times, providing
/// constant-time lookups and amortized constant-time updates. This approach is particularly
/// valuable for technical analysis indicators that require identifying price extremes,
/// such as Donchian channels, volatility measures, and support/resistance level detection.
///
/// The implementation is optimized for financial time-series analysis where
/// identifying maximum values within specific lookback periods is essential
/// for decision-making processes.
#[derive(Debug)]
pub struct Maximum<T>(MonotonicQueue<T, Max>);

impl<T: Default + Clone + Float> Maximum<T> {
    /// Creates a new Maximum instance with the specified period
    ///
    /// # Arguments
    ///
    /// * `period` - The size of the rolling window
    ///
    /// # Returns
    ///
    /// A new Maximum instance
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
    /// None if the window is not yet full, otherwise returns the maximum value
    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    /// Returns the maximum value in the rolling window
    ///
    /// # Returns
    ///
    /// None if the window is not yet full, otherwise returns the maximum value
    pub fn get(&self) -> Option<T> {
        self.0.front()
    }

    /// Resets the rolling window
    pub fn reset(&mut self) {
        self.0.reset();
    }
}
