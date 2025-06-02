use core::marker::PhantomData;

use super::Deque;

/// Trait for defining order policies for monotonic queue
///
/// # Type Parameters
///
/// * `T` - The type of the elements in the queue
///
/// # Methods
///
/// * `should_remove(existing: &T, new: &T) -> bool` - Returns true if the existing element should be removed
///
pub trait OrderPolicy<T> {
    fn should_remove(existing: &T, new: &T) -> bool;
}

/// Order policy for minimum
#[derive(Debug, Clone)]
pub struct Min;

/// Order policy for maximum
#[derive(Debug, Clone)]
pub struct Max;

impl<T: PartialOrd> OrderPolicy<T> for Min {
    #[inline]
    fn should_remove(existing: &T, new: &T) -> bool {
        existing > new
    }
}

impl<T: PartialOrd> OrderPolicy<T> for Max {
    #[inline]
    fn should_remove(existing: &T, new: &T) -> bool {
        existing < new
    }
}

// Pair of (value, position)
type Entry<T> = (T, usize);

/// Monotonic queue implementation
///
/// # Type Parameters
///
/// * `T` - The type of the elements in the queue
/// * `O` - The order policy for the queue
#[derive(Debug, Clone)]
pub struct MonotonicQueue<T, O> {
    deque: Deque<Entry<T>>,
    element_count: usize,
    _order: PhantomData<O>,
}

impl<T, O> MonotonicQueue<T, O>
where
    T: PartialOrd + Copy + Default,
    O: OrderPolicy<T>,
{
    /// Creates a new `MonotonicQueue` instance with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The capacity of the queue
    ///
    /// # Returns
    ///
    /// * `Self` - The `MonotonicQueue` instance
    #[inline]
    pub fn new(window_size: usize) -> Self {
        Self {
            deque: Deque::new(window_size),
            element_count: 0,
            _order: PhantomData,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.deque.capacity()
    }

    /// Returns true if the queue has processed enough elements to fill the window
    ///
    /// # Returns
    ///
    /// * `bool` - True if the window is filled with elements
    #[inline]
    pub fn has_complete_window(&self) -> bool {
        self.element_count >= self.capacity()
    }

    /// Determines if a position is outside the current sliding window
    ///
    /// # Arguments
    ///
    /// * `pos` - The position to check
    ///
    /// # Returns
    ///
    /// * `bool` - True if the position is outside the current sliding window
    #[inline]
    fn is_position_outside_window(&self, pos: usize) -> bool {
        if !self.has_complete_window() {
            return false;
        }

        let window_start = self.element_count.saturating_sub(self.capacity());
        pos <= window_start
    }

    /// Removes expired elements from the front of the deque
    #[inline]
    fn remove_expired_elements(&mut self) {
        while let Some(&(_, pos)) = self.deque.front() {
            if self.is_position_outside_window(pos) {
                self.deque.pop_front();
            } else {
                break;
            }
        }
    }

    /// Maintains monotonic property by removing dominated elements
    #[inline]
    fn maintain_monotonic_property(&mut self, value: T) {
        while let Some(&(existing, _)) = self.deque.back() {
            if O::should_remove(&existing, &value) {
                self.deque.pop_back();
            } else {
                break;
            }
        }
    }

    /// Pushes a new value into the queue
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push into the queue
    #[inline]
    pub fn push(&mut self, value: T) {
        self.remove_expired_elements();
        self.maintain_monotonic_property(value);
        self.deque.push_back((value, self.element_count));
        self.element_count += 1;
    }

    /// Returns the front element of the queue
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The front element of the queue, or `None` if the queue is empty
    #[inline]
    pub fn front(&self) -> Option<T> {
        self.deque.front().map(|&(value, _)| value)
    }

    /// Resets the queue to its initial state
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The queue object
    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.deque.reset();
        self.element_count = 0;
        self
    }

    /// Returns true if the queue is empty
    ///
    /// # Returns
    ///
    /// * `bool` - True if the queue is empty
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::all)]
mod tests {
    use super::{Max, Min, MonotonicQueue};

    #[test]
    fn test_monotonic_queue_equal_values_handling() {
        // Test that equal values are handled consistently
        let mut mq = MonotonicQueue::<_, Min>::new(3);
        mq.push(5);
        mq.push(5);
        mq.push(5);
        assert_eq!(mq.front(), Some(5));

        mq.push(4);
        assert_eq!(mq.front(), Some(4)); // Should pick the newer equal value
    }

    #[test]
    fn test_monotonic_queue_min_sliding_window() {
        let input = [25.4, 26.2, 26.0, 26.1, 25.8, 25.9, 26.3, 26.2, 26.5];
        let window = 3;
        let mut mq = MonotonicQueue::<_, Min>::new(window);
        let mut result = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq.push(val);
            if i >= window - 1 {
                result.push(mq.front().unwrap());
            }
        }
        assert_eq!(result, vec![25.4, 26.0, 25.8, 25.8, 25.8, 25.9, 26.2]);
    }

    #[test]
    fn test_monotonic_queue_max_sliding_window() {
        let input = [
            31, 4, 52, 60, 61, 15, 28, 2, 36, 1, 4, 39, 12, 96, 1, 21, 95, 20, 35, 83,
        ];
        let window = 3;
        let mut mq = MonotonicQueue::<_, Max>::new(window);
        let mut result = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq.push(val);
            if i >= window - 1 {
                result.push(mq.front().unwrap());
            }
        }

        assert_eq!(
            result,
            vec![
                52, 60, 61, 61, 61, 28, 36, 36, 36, 39, 39, 96, 96, 96, 95, 95, 95, 83,
            ]
        );
    }

    #[test]
    fn test_monotonic_queue_min_single_element_window() {
        let input = [9, 7, 8];
        let mut mq = MonotonicQueue::<_, Min>::new(1);
        let mut result = vec![];

        for &val in &input {
            mq.push(val);
            result.push(mq.front().unwrap());
        }

        assert_eq!(result, input);
    }

    #[test]
    fn test_monotonic_reset() {
        let mut mq = MonotonicQueue::<_, Min>::new(3);
        mq.push(14);
        mq.push(13);
        mq.push(12);
        assert_eq!(mq.front(), Some(12));
        mq.push(11);
        assert_eq!(mq.front(), Some(11));
        mq.reset();
        assert!(mq.front().is_none());
        mq.push(10);
        assert_eq!(mq.front(), Some(10));
    }

    #[test]
    fn test_monotonic_queue_max_full_length_window() {
        let input = [2, 4, 1, 3];
        let mut mq = MonotonicQueue::<_, Max>::new(input.len());
        let mut result = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq.push(val);
            if i == input.len() - 1 {
                result.push(mq.front().unwrap());
            }
        }

        assert_eq!(result, vec![4]);
    }

    #[test]
    fn test_monotonic_queue_random_window_sizes() {
        let input = [5, 1, 3, 8, 6, 2, 9];
        let windows = [1, 2, 3, 4, 5];

        for &w in &windows {
            let mut mq_min = MonotonicQueue::<_, Min>::new(w);
            let mut min_result = vec![];

            for (i, &val) in input.iter().enumerate() {
                mq_min.push(val);
                if i >= w - 1 {
                    min_result.push(mq_min.front().unwrap());
                }
            }

            for (i, &val) in min_result.iter().enumerate() {
                let start = i;
                let end = (i + w).min(input.len());
                let window_slice = &input[start..end];
                assert!(window_slice.iter().any(|&x| x == val));
            }
        }
    }

    #[test]
    fn test_edge_case_window_size_one() {
        let input = [5, 2, 9, 1, 7, 3];
        let mut mq_min = MonotonicQueue::<_, Min>::new(1);
        let mut mq_max = MonotonicQueue::<_, Max>::new(1);

        for &val in &input {
            mq_min.push(val);
            mq_max.push(val);
            assert_eq!(mq_min.front(), Some(val));
            assert_eq!(mq_max.front(), Some(val));
        }
    }

    #[test]
    fn test_duplicated_values() {
        let input = [3, 3, 3, 3, 2, 2, 2, 4, 4];
        let window = 3;

        let mut mq_min = MonotonicQueue::<_, Min>::new(window);
        let mut min_results = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq_min.push(val);
            if i >= window - 1 {
                min_results.push(mq_min.front().unwrap());
            }
        }

        assert_eq!(min_results, vec![3, 3, 2, 2, 2, 2, 2]);

        let mut mq_max = MonotonicQueue::<_, Max>::new(window);
        let mut max_results = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq_max.push(val);
            if i >= window - 1 {
                max_results.push(mq_max.front().unwrap());
            }
        }

        assert_eq!(max_results, vec![3, 3, 3, 3, 2, 4, 4]);
    }

    #[test]
    fn test_ascending_descending_sequences() {
        let ascending = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let window = 3;

        let mut mq_min = MonotonicQueue::<_, Min>::new(window);
        let mut min_results = vec![];

        for (i, &val) in ascending.iter().enumerate() {
            mq_min.push(val);
            if i >= window - 1 {
                min_results.push(mq_min.front().unwrap());
            }
        }

        assert_eq!(min_results, vec![1, 2, 3, 4, 5, 6, 7]);

        let descending = [9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut mq_max = MonotonicQueue::<_, Max>::new(window);
        let mut max_results = vec![];

        for (i, &val) in descending.iter().enumerate() {
            mq_max.push(val);
            if i >= window - 1 {
                max_results.push(mq_max.front().unwrap());
            }
        }

        assert_eq!(max_results, vec![9, 8, 7, 6, 5, 4, 3]);
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let input = [0, -10, 1000, -5, 5, 0, -1000, 42];
        let window = 3;

        let mut mq_min = MonotonicQueue::<_, Min>::new(window);
        let mut min_results = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq_min.push(val);
            if i >= window - 1 {
                min_results.push(mq_min.front().unwrap());
            }
        }

        assert_eq!(min_results, vec![-10, -10, -5, -5, -1000, -1000]);

        let mut mq_max = MonotonicQueue::<_, Max>::new(window);
        let mut max_results = vec![];

        for (i, &val) in input.iter().enumerate() {
            mq_max.push(val);
            if i >= window - 1 {
                max_results.push(mq_max.front().unwrap());
            }
        }

        assert_eq!(max_results, vec![1000, 1000, 1000, 5, 5, 42]);
    }

    #[test]
    fn test_empty_queue() {
        let mut mq = MonotonicQueue::<i32, Min>::new(3);
        assert!(mq.is_empty());
        assert_eq!(mq.front(), None);

        mq.push(5);
        assert!(!mq.is_empty());
        assert_eq!(mq.front(), Some(5));

        mq.reset();
        assert!(mq.is_empty());
        assert_eq!(mq.front(), None);
    }

    #[test]
    fn test_oscillating_values() {
        let oscillating = [10, 2, 8, 1, 9, 3, 7, 0];
        let window = 4;

        let mut mq_min = MonotonicQueue::<_, Min>::new(window);
        let mut min_results = vec![];

        for (i, &val) in oscillating.iter().enumerate() {
            mq_min.push(val);
            if i >= window - 1 {
                min_results.push(mq_min.front().unwrap());
            }
        }

        assert_eq!(min_results, vec![1, 1, 1, 1, 0]);

        let mut mq_max = MonotonicQueue::<_, Max>::new(window);
        let mut max_results = vec![];

        for (i, &val) in oscillating.iter().enumerate() {
            mq_max.push(val);
            if i >= window - 1 {
                max_results.push(mq_max.front().unwrap());
            }
        }

        assert_eq!(max_results, vec![10, 9, 9, 9, 9]);
    }
}
