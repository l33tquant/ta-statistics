use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use num_traits::Float;
use ordered_float::{OrderedFloat, PrimitiveFloat};

use alloc::vec::Vec;

/// A structure that tracks the mode (most frequent value) with push/pop calls from a rolling window.
///
/// This structure efficiently maintains frequency counts for values in a rolling window
/// and provides O(1) lookup time for the current mode. It uses hash maps to track:
/// - The frequency of each value
/// - The set of values for each frequency
///
/// - Insertion time complexity: O(1) amortized
/// - Removal time complexity: O(1) amortized
/// - Mode lookup time complexity: O(1)
///
/// This implementation is optimized for financial time-series analysis where identifying
/// the most common values can reveal important price levels and market tendencies.

#[derive(Debug, Clone)]
pub struct RollingMode<T> {
    /// Maps each value to its frequency count
    freq: HashMap<OrderedFloat<T>, usize, RandomState>,
    /// Maps each frequency to the set of values that occur that many times
    freq_bucket: HashMap<usize, HashSet<OrderedFloat<T>, RandomState>, RandomState>,
    /// The current mode (most frequent value)
    mode: Option<OrderedFloat<T>>,
    /// The frequency of the current mode
    mode_freq: usize,
}

impl<T> RollingMode<T>
where
    T: Float + PrimitiveFloat,
{
    /// Creates a new instance of the `RollingMode` structure.
    ///
    /// Returns an empty `RollingMode` with no values tracked yet.
    /// The structure is initialized with a default random state for hash maps.
    ///
    /// # Returns
    ///
    /// * `Self` - The rolling mode object
    pub fn new() -> Self {
        let hasher = RandomState::default();
        let freq = HashMap::with_hasher(hasher.clone());
        let freq_bucket = HashMap::with_hasher(hasher);
        Self {
            freq,
            freq_bucket,
            mode: None,
            mode_freq: 0,
        }
    }

    /// Adds a new value to the frequency tracking system
    ///
    /// This method updates the internal frequency counters when a new value enters the rolling window.
    /// It maintains the frequency maps and updates the current mode if necessary.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to add to the window
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The rolling mode object
    pub fn push(&mut self, value: T) {
        let value = OrderedFloat(value);
        let old_freq = self.freq.get(&value).copied().unwrap_or(0);
        let new_freq = old_freq + 1;

        self.freq.insert(value, new_freq);

        self.update_freq_buckets(value, old_freq, new_freq);

        if new_freq > self.mode_freq {
            self.mode = Some(value);
            self.mode_freq = new_freq;
        } else if new_freq == self.mode_freq && (self.mode.is_none() || Some(value) < self.mode) {
            self.mode = Some(value);
        }
    }

    /// This method updates the internal frequency counters when a value leaves the rolling window.
    /// It maintains the frequency maps and updates the current mode if necessary.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to remove from the window
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The rolling mode object
    pub fn pop(&mut self, value: T) {
        let value = OrderedFloat(value);
        if let Some(&old_freq) = self.freq.get(&value) {
            let new_freq = old_freq - 1;

            if new_freq == 0 {
                self.freq.remove(&value);
            } else {
                self.freq.insert(value, new_freq);
            }

            self.update_freq_buckets(value, old_freq, new_freq);

            let should_recalculate = self.mode == Some(value) || old_freq == self.mode_freq;
            if should_recalculate {
                self.recalculate_mode();
            }
        }
    }

    /// Returns the mode (most frequent value) if any
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The mode of values in the window, or `None` if the window is not full
    pub fn mode(&self) -> Option<T> {
        self.mode.map(|x| x.0)
    }

    /// Returns all values that have the maximum frequency
    ///
    /// # Returns
    ///
    /// * `Vec<T>` - A vector of values that have the maximum frequency in the window
    #[allow(dead_code)]
    pub fn modes(&self) -> Vec<T> {
        if self.mode_freq == 0 {
            return Vec::new();
        }

        self.freq_bucket
            .get(&self.mode_freq)
            .map_or_else(Vec::new, |values| values.iter().map(|&v| v.0).collect())
    }

    /// Clears all data
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The rolling mode object
    pub fn reset(&mut self) {
        self.freq.clear();
        self.freq_bucket.clear();
        self.mode = None;
        self.mode_freq = 0;
    }

    /// Helper method to update frequency buckets when a value's frequency changes
    fn update_freq_buckets(&mut self, value: OrderedFloat<T>, old_freq: usize, new_freq: usize) {
        if old_freq > 0 {
            if let Some(bucket) = self.freq_bucket.get_mut(&old_freq) {
                bucket.remove(&value);

                if bucket.is_empty() {
                    self.freq_bucket.remove(&old_freq);
                }
            }
        }

        if new_freq > 0 {
            self.freq_bucket.entry(new_freq).or_default().insert(value);
        }
    }

    /// Recalculates the mode when the current mode's frequency changes
    fn recalculate_mode(&mut self) {
        if self.freq.is_empty() {
            self.mode = None;
            self.mode_freq = 0;
            return;
        }

        self.mode_freq = *self.freq_bucket.keys().max().unwrap_or(&0);

        if self.mode_freq > 0 {
            if let Some(values) = self.freq_bucket.get(&self.mode_freq) {
                self.mode = values.iter().min().copied();
            } else {
                self.mode = None;
            }
        } else {
            self.mode = None;
        }
    }

    /// Returns the current maximum frequency
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum frequency of values in the window
    #[allow(dead_code)]
    pub fn max_frequency(&self) -> usize {
        self.mode_freq
    }

    /// Returns the frequency of a specific value
    ///
    /// # Arguments
    ///
    /// * `value` - The value to get the frequency of
    ///
    /// # Returns
    ///
    /// * `usize` - The frequency of the value in the window
    #[allow(dead_code)]
    pub fn frequency_of(&self, value: T) -> usize {
        self.freq.get(&OrderedFloat(value)).copied().unwrap_or(0)
    }

    /// Returns the number of unique values being tracked
    ///
    /// # Returns
    ///
    /// * `usize` - The number of unique values being tracked
    #[allow(dead_code)]
    pub fn unique_count(&self) -> usize {
        self.freq.len()
    }

    /// Returns whether the tracker is empty
    ///
    /// # Returns
    ///
    /// * `bool` - Whether the tracker is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.freq.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_empty() {
        let tracker: RollingMode<f64> = RollingMode::new();
        assert_eq!(tracker.mode(), None);
        assert_eq!(tracker.modes(), Vec::<f64>::new());
        assert_eq!(tracker.max_frequency(), 0);
        assert_eq!(tracker.unique_count(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_single_value() {
        let mut tracker = RollingMode::new();
        tracker.push(42.0);

        assert_eq!(tracker.mode(), Some(42.0));
        assert_eq!(tracker.modes(), vec![42.0]);
        assert_eq!(tracker.max_frequency(), 1);
        assert_eq!(tracker.unique_count(), 1);
        assert!(!tracker.is_empty());
    }

    #[test]
    fn test_multiple_values() {
        let mut tracker = RollingMode::new();
        tracker.push(1.5);
        tracker.push(2.5);
        tracker.push(1.5);
        tracker.push(3.5);

        assert_eq!(tracker.mode(), Some(1.5));
        assert_eq!(tracker.modes(), vec![1.5]);
        assert_eq!(tracker.max_frequency(), 2);
        assert_eq!(tracker.unique_count(), 3);
        assert_eq!(tracker.frequency_of(1.5), 2);
        assert_eq!(tracker.frequency_of(2.5), 1);
        assert_eq!(tracker.frequency_of(3.5), 1);
        assert_eq!(tracker.frequency_of(4.5), 0);
    }

    #[test]
    fn test_numerical_order_tie_breaking() {
        let mut tracker = RollingMode::new();

        tracker.push(3.0);
        tracker.push(1.0);
        tracker.push(2.0);

        assert_eq!(tracker.mode(), Some(1.0));

        tracker.push(2.0);
        tracker.push(1.0);
        tracker.push(3.0);

        assert_eq!(tracker.mode(), Some(1.0));

        tracker.pop(1.0);

        assert_eq!(tracker.mode(), Some(2.0));
    }

    #[test]
    fn test_reset() {
        let mut tracker = RollingMode::new();
        tracker.push(1.5);
        tracker.push(2.5);
        tracker.push(1.5);

        assert_eq!(tracker.mode(), Some(1.5));
        assert_eq!(tracker.max_frequency(), 2);

        tracker.reset();

        assert_eq!(tracker.mode(), None);
        assert_eq!(tracker.modes(), Vec::<f64>::new());
        assert_eq!(tracker.max_frequency(), 0);
        assert_eq!(tracker.unique_count(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_nan_handling() {
        let mut tracker = RollingMode::new();
        tracker.push(1.0);
        tracker.push(f64::NAN);
        tracker.push(f64::NAN);

        assert!(tracker.mode().unwrap().is_nan());
        assert_eq!(tracker.max_frequency(), 2);

        tracker.pop(f64::NAN);

        assert_eq!(tracker.mode(), Some(1.0));
        assert_eq!(tracker.max_frequency(), 1);
    }

    #[test]
    fn test_rolling_window_scenario() {
        let mut tracker = RollingMode::new();
        let inputs = [1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0];

        let mut window = Vec::new();

        for &value in inputs.iter() {
            window.push(value);
            tracker.push(value);

            if window.len() > 3 {
                let oldest = window.remove(0);
                tracker.pop(oldest);
            }
        }

        assert_eq!(tracker.mode(), Some(2.0));
    }

    #[test]
    fn test_multiple_ties() {
        let mut tracker = RollingMode::new();

        tracker.push(3.0);
        tracker.push(1.0);
        tracker.push(2.0);

        assert_eq!(tracker.mode(), Some(1.0));
        assert_eq!(tracker.modes().len(), 3);

        tracker.pop(1.0);

        assert_eq!(tracker.mode(), Some(2.0));
        assert_eq!(tracker.modes().len(), 2);
    }

    #[test]
    fn test_special_floating_point_values() {
        let mut tracker = RollingMode::new();

        tracker.push(10.0);
        tracker.push(20.0);
        assert_eq!(tracker.mode(), Some(10.0));

        tracker.push(20.0);
        assert_eq!(tracker.mode(), Some(20.0));

        tracker.reset();
        tracker.push(0.0);
        tracker.push(-0.0);

        assert_eq!(tracker.unique_count(), 1);
        assert_eq!(tracker.frequency_of(0.0), 2);
        assert_eq!(tracker.frequency_of(-0.0), 2);

        tracker.reset();
        for i in 0..5 {
            tracker.push(i as f64);
        }
        assert_eq!(tracker.mode(), Some(0.0));

        tracker.push(3.0);
        assert_eq!(tracker.mode(), Some(3.0));
    }

    #[test]
    fn test_removing_nonexistent_value() {
        let mut tracker = RollingMode::new();

        tracker.push(1.0);
        tracker.push(2.0);

        tracker.pop(3.0);

        assert_eq!(tracker.unique_count(), 2);
        assert_eq!(tracker.mode(), Some(1.0));
    }

    #[test]
    fn test_repeated_values() {
        let mut tracker = RollingMode::new();

        for _ in 0..100 {
            tracker.push(42.0);
        }

        assert_eq!(tracker.mode(), Some(42.0));
        assert_eq!(tracker.max_frequency(), 100);

        for i in 0..100 {
            tracker.pop(42.0);
            assert_eq!(tracker.max_frequency(), 100 - i - 1);
        }

        assert_eq!(tracker.mode(), None);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_reset_behavior() {
        let mut tracker = RollingMode::new();

        tracker.push(1.0);
        tracker.push(2.0);
        tracker.push(1.0);

        assert_eq!(tracker.mode(), Some(1.0));

        tracker.reset();

        assert!(tracker.is_empty());
        assert_eq!(tracker.mode(), None);
        assert_eq!(tracker.max_frequency(), 0);
        assert_eq!(tracker.unique_count(), 0);
        assert_eq!(tracker.frequency_of(1.0), 0);

        tracker.push(3.0);
        assert_eq!(tracker.mode(), Some(3.0));
    }

    #[test]
    fn test_pushing_many_unique_values() {
        let mut tracker = RollingMode::new();

        for i in 0..1000 {
            tracker.push(i as f64);
        }

        assert_eq!(tracker.max_frequency(), 1);
        assert_eq!(tracker.unique_count(), 1000);

        assert_eq!(tracker.mode(), Some(0.0));

        tracker.push(42.0);
        assert_eq!(tracker.mode(), Some(42.0));
        assert_eq!(tracker.max_frequency(), 2);
    }
}
