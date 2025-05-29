use ahash::RandomState;
use hashbrown::HashMap;
use num_traits::Float;
use ordered_float::{OrderedFloat, PrimitiveFloat};

use alloc::collections::BinaryHeap;
use core::cmp::Reverse;
/// A median calculator that efficiently computes the median using a two-heap approach.
///
/// This implementation is designed to work with external window management.
/// The caller is responsible for managing the sliding window and calling push/pop methods.
/// Using two balanced heaps to ensure O(log n) time complexity for insertions and deletions,
/// with O(1) median access.
///
/// # Type Parameters
///
/// * `T` - A floating point type that implements the `PrimitiveFloat` trait
#[derive(Debug, Clone)]
pub struct RollingMedian<T> {
    /// Max heap for the lower half of values (elements ≤ median)
    lower_heap: BinaryHeap<OrderedFloat<T>>,
    /// Min heap for the upper half of values (elements > median)
    upper_heap: BinaryHeap<Reverse<OrderedFloat<T>>>,
    /// Tracks elements scheduled for removal from heaps Maps values to their removal count for lazy deletion
    removal_tracker: HashMap<OrderedFloat<T>, usize, RandomState>,
    /// Count of elements currently in the heaps (not counting those marked for removal)
    /// This is used internally to determine if we have an odd or even number of elements
    element_count: usize,
}

impl<T: Float> RollingMedian<T>
where
    T: PrimitiveFloat,
{
    /// Creates a new `RollingMedian` instance with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The size of the sliding window
    ///
    /// # Returns
    ///
    /// A new `RollingMedian` instance with pre-allocated capacity
    #[inline]
    pub fn new(window_size: usize) -> Self {
        RollingMedian {
            lower_heap: BinaryHeap::with_capacity(window_size),
            upper_heap: BinaryHeap::with_capacity(window_size),
            removal_tracker: HashMap::with_capacity_and_hasher(window_size, RandomState::default()),
            element_count: 0,
        }
    }

    /// Pushes a new value to the median calculator.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push
    pub fn push(&mut self, value: T) {
        let value = OrderedFloat(value);
        self.element_count += 1;
        self.add_to_appropriate_heap(value);
        self.rebalance_heaps();
    }

    /// Adds a value to the appropriate heap based on its value
    ///
    /// # Arguments
    ///
    /// * `value` - The value to add
    #[inline]
    fn add_to_appropriate_heap(&mut self, value: OrderedFloat<T>) {
        if self.should_add_to_lower_heap(value) {
            self.lower_heap.push(value);
        } else {
            self.upper_heap.push(Reverse(value));
        }
    }

    /// Determines if a value should be added to the lower heap
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value should be added to the lower heap, `false` otherwise
    #[inline]
    fn should_add_to_lower_heap(&self, value: OrderedFloat<T>) -> bool {
        self.lower_heap.is_empty()
            || value
                <= *self
                    .lower_heap
                    .peek()
                    .unwrap_or(&OrderedFloat(Float::max_value()))
    }

    /// Pops a value from the median calculator.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to pop
    ///
    /// # Returns
    ///
    /// `true` if the value was found and removed, `false` otherwise
    pub fn pop(&mut self, value: T) -> bool {
        if self.mark_for_removal(OrderedFloat(value)) {
            self.decrement_element_count();
            self.rebalance_heaps();
            true
        } else {
            false
        }
    }

    /// Decrements the element count safely
    #[inline]
    fn decrement_element_count(&mut self) {
        if self.element_count > 0 {
            self.element_count -= 1;
        }
    }

    /// Marks a value for removal from the heaps.
    ///
    /// The value isn't immediately removed from the heaps for efficiency.
    /// Instead, it's tracked in the removal_tracker and will be removed
    /// during heap operations when necessary.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to mark for removal
    ///
    /// # Returns
    ///
    /// `true` if the value was found and marked for removal, `false` otherwise
    #[inline]
    fn mark_for_removal(&mut self, value: OrderedFloat<T>) -> bool {
        if self.value_exists_in_heaps(value) {
            self.increment_removal_count(value);
            true
        } else {
            false
        }
    }

    /// Checks if a value exists in either heap
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value exists in either heap, `false` otherwise
    #[inline]
    fn value_exists_in_heaps(&self, value: OrderedFloat<T>) -> bool {
        self.value_exists_in_lower_heap(value) || self.value_exists_in_upper_heap(value)
    }

    /// Checks if a value exists in the lower heap
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value exists in the lower heap, `false` otherwise
    #[inline]
    fn value_exists_in_lower_heap(&self, value: OrderedFloat<T>) -> bool {
        self.lower_heap.iter().any(|v| *v == value)
    }

    /// Checks if a value exists in the upper heap
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value exists in the upper heap, `false` otherwise
    #[inline]
    fn value_exists_in_upper_heap(&self, value: OrderedFloat<T>) -> bool {
        self.upper_heap.iter().any(|v| v.0 == value)
    }

    /// Increments the removal count for a value
    ///
    /// # Arguments
    ///
    /// * `value` - The value to increment the removal count for
    #[inline]
    fn increment_removal_count(&mut self, value: OrderedFloat<T>) {
        let entry = self.removal_tracker.entry(value).or_insert(0);
        *entry += 1;
    }

    /// Ensures that both heaps are balanced according to the median invariant:
    /// - The lower heap contains elements ≤ median
    /// - The upper heap contains elements > median
    /// - For odd element_count: lower_heap.size = upper_heap.size + 1
    /// - For even element_count: lower_heap.size = upper_heap.size
    ///
    /// This method also performs lazy deletion of elements marked for removal.
    fn rebalance_heaps(&mut self) {
        self.clean_heap_tops();
        self.optimize_mem();
        let (target_lower_size, target_upper_size) = self.calculate_target_heap_sizes();
        let lower_size = self.get_effective_heap_size(&self.lower_heap);
        let upper_size = self.get_effective_heap_size_upper();
        self.balance_heaps(lower_size, upper_size, target_lower_size, target_upper_size);
    }

    /// Calculates the target sizes for each heap based on element count
    ///
    /// # Returns
    ///
    /// A tuple containing the target sizes for the lower and upper heaps
    #[inline]
    fn calculate_target_heap_sizes(&self) -> (usize, usize) {
        let target_lower_size = self.element_count.div_ceil(2);
        let target_upper_size = self.element_count / 2;
        (target_lower_size, target_upper_size)
    }

    /// Balances both heaps to achieve their target sizes
    ///
    /// This method handles all heap balancing in a single pass, ensuring that
    /// the heaps are properly balanced according to the median invariant.
    ///
    /// # Arguments
    ///
    /// * `lower_size` - Current effective size of the lower heap
    /// * `upper_size` - Current effective size of the upper heap
    /// * `target_lower_size` - Target size for the lower heap
    /// * `target_upper_size` - Target size for the upper heap
    fn balance_heaps(
        &mut self,
        lower_size: usize,
        upper_size: usize,
        target_lower_size: usize,
        target_upper_size: usize,
    ) {
        if lower_size > target_lower_size {
            let elements_to_move = lower_size - target_lower_size;
            self.move_elements_from_lower_to_upper(elements_to_move);
        } else if upper_size > target_upper_size {
            let elements_to_move = upper_size - target_upper_size;
            self.move_elements_from_upper_to_lower(elements_to_move);
        } else if lower_size < target_lower_size && upper_size > 0 {
            let elements_to_move = core::cmp::min(target_lower_size - lower_size, upper_size);
            self.move_elements_from_upper_to_lower(elements_to_move);
        } else if upper_size < target_upper_size && lower_size > 0 {
            let elements_to_move = core::cmp::min(target_upper_size - upper_size, lower_size);
            self.move_elements_from_lower_to_upper(elements_to_move);
        }
    }

    /// Moves a specific number of elements from the lower heap to the upper heap
    ///
    /// This method will move at most the specified number of elements, but may
    /// move fewer if there aren't enough valid elements to move.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of elements to move
    fn move_elements_from_lower_to_upper(&mut self, count: usize) {
        let mut moved = 0;

        while moved < count {
            if self.lower_heap.is_empty() {
                break;
            }

            if let Some(value) = self.lower_heap.pop() {
                if self.is_marked_for_removal(value) {
                    self.process_removed_element(value);
                    continue;
                }

                self.upper_heap.push(Reverse(value));
                moved += 1;
            } else {
                break;
            }
        }
    }

    /// Moves a specific number of elements from the upper heap to the lower heap
    ///
    /// This method will move at most the specified number of elements, but may
    /// move fewer if there aren't enough valid elements to move.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of elements to move
    fn move_elements_from_upper_to_lower(&mut self, count: usize) {
        let mut moved = 0;

        while moved < count {
            if self.upper_heap.is_empty() {
                break;
            }

            if let Some(Reverse(value)) = self.upper_heap.pop() {
                if self.is_marked_for_removal(value) {
                    self.process_removed_element(value);
                    continue;
                }

                self.lower_heap.push(value);
                moved += 1;
            } else {
                break;
            }
        }
    }

    /// Checks if a value is marked for removal
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value is marked for removal, `false` otherwise
    #[inline]
    fn is_marked_for_removal(&self, value: OrderedFloat<T>) -> bool {
        self.removal_tracker.contains_key(&value)
    }

    /// Processes a removed element by updating its removal count
    ///
    /// # Arguments
    ///
    /// * `value` - The value to process
    #[inline]
    fn process_removed_element(&mut self, value: OrderedFloat<T>) {
        if let Some(count) = self.removal_tracker.get_mut(&value) {
            *count -= 1;
            if *count == 0 {
                self.removal_tracker.remove(&value);
            }
        }
    }

    /// Gets the effective size of a heap (excluding elements marked for removal)
    ///
    /// # Arguments
    ///
    /// * `heap` - The heap to get the effective size of
    ///
    /// # Returns
    ///
    /// The effective size of the heap
    #[inline]
    fn get_effective_heap_size(&self, heap: &BinaryHeap<OrderedFloat<T>>) -> usize {
        heap.iter()
            .filter(|item| !self.removal_tracker.contains_key(*item))
            .count()
    }

    /// Gets the effective size of the upper heap (excluding elements marked for removal)
    ///
    /// # Returns
    ///
    /// The effective size of the upper heap
    #[inline]
    fn get_effective_heap_size_upper(&self) -> usize {
        self.upper_heap
            .iter()
            .filter(|item| !self.removal_tracker.contains_key(&item.0))
            .count()
    }

    /// Removes elements from the tops of the heaps if they're marked for removal.
    fn clean_heap_tops(&mut self) {
        self.clean_lower_heap_top();
        self.clean_upper_heap_top();
    }

    /// Removes elements from the top of the lower heap if they're marked for removal
    fn clean_lower_heap_top(&mut self) {
        while let Some(top) = self.lower_heap.peek() {
            if self.removal_tracker.contains_key(top) {
                if let Some(value) = self.lower_heap.pop() {
                    self.process_removed_element(value);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Removes elements from the top of the upper heap if they're marked for removal
    fn clean_upper_heap_top(&mut self) {
        while let Some(Reverse(top)) = self.upper_heap.peek() {
            if self.removal_tracker.contains_key(top) {
                if let Some(Reverse(value)) = self.upper_heap.pop() {
                    self.process_removed_element(value);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Calculates the current median of the values.
    ///
    /// # Returns
    ///
    /// * `Some(median)` if there is at least one value
    /// * `None` if there are no values
    pub fn median(&mut self) -> Option<T> {
        self.rebalance_heaps();
        if self.element_count == 0 {
            return None;
        }
        self.calculate_median()
    }

    /// Calculates the median based on the current state of the heaps
    ///
    /// # Returns
    ///
    /// * `Some(median)` if there is at least one value
    /// * `None` if there are no valid elements in the heaps
    fn calculate_median(&self) -> Option<T> {
        let lower_top = self.find_first_valid_in_lower();
        let upper_top = self.find_first_valid_in_upper();

        match (lower_top, upper_top) {
            (Some(lower), Some(upper)) => {
                if self.has_odd_number_of_elements() {
                    Some(lower)
                } else {
                    self.average(lower, upper)
                }
            }
            (Some(lower), None) => Some(lower),
            (None, Some(upper)) => Some(upper),
            (None, None) => None,
        }
    }

    /// Checks if there is an odd number of elements
    ///
    /// # Returns
    ///
    /// `true` if there is an odd number of elements, `false` otherwise
    #[inline]
    fn has_odd_number_of_elements(&self) -> bool {
        self.element_count % 2 == 1
    }

    /// Calculates the average of two values
    ///
    /// # Arguments
    ///
    /// * `a` - The first value
    /// * `b` - The second value
    ///
    /// # Returns
    ///
    /// The average of the two values
    #[inline]
    fn average(&self, a: T, b: T) -> Option<T> {
        T::from(2.0).map(|n| (a + b) / n)
    }

    /// Finds the first valid (not marked for removal) element in the lower heap
    ///
    /// # Returns
    ///
    /// * `Some(value)` if a valid element was found
    /// * `None` if no valid elements were found
    #[inline]
    fn find_first_valid_in_lower(&self) -> Option<T> {
        self.lower_heap
            .iter()
            .find(|item| !self.removal_tracker.contains_key(*item))
            .map(|item| item.0)
    }

    /// Finds the first valid (not marked for removal) element in the upper heap
    ///
    /// # Returns
    ///
    /// * `Some(value)` if a valid element was found
    /// * `None` if no valid elements were found
    #[inline]
    fn find_first_valid_in_upper(&self) -> Option<T> {
        self.upper_heap
            .iter()
            .find(|item| !self.removal_tracker.contains_key(&item.0))
            .map(|item| item.0.0)
    }

    /// Returns the current number of active elements.
    ///
    /// # Returns
    ///
    /// The number of elements currently tracked (not marked for removal)
    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.element_count
    }

    /// Checks if there are no active elements.
    ///
    /// # Returns
    ///
    /// `true` if there are no active elements, `false` otherwise
    #[allow(dead_code)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// Clears all elements.
    pub fn reset(&mut self) {
        self.lower_heap.clear();
        self.upper_heap.clear();
        self.removal_tracker.clear();
        self.element_count = 0;
    }

    #[inline]
    fn optimize_mem(&mut self) {
        if self.removal_tracker.len() > self.element_count * 2 {
            self.removal_tracker.retain(|_, count| *count > 0);

            if self.removal_tracker.capacity() > self.removal_tracker.len() * 2 {
                self.removal_tracker.shrink_to_fit();
            }

            if self.lower_heap.capacity() > self.element_count * 2 {
                self.lower_heap.shrink_to_fit();
            }

            if self.upper_heap.capacity() > self.element_count * 2 {
                self.upper_heap.shrink_to_fit();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_empty() {
        let mut median = RollingMedian::<f64>::new(5);
        assert_eq!(median.median(), None);
    }

    #[test]
    fn test_median_single_value() {
        let mut median = RollingMedian::<f64>::new(5);
        median.push(42.0);
        assert_eq!(median.median(), Some(42.0));
    }

    #[test]
    fn test_median_odd_number_of_values() {
        let mut median = RollingMedian::<f64>::new(5);
        median.push(1.0);
        median.push(3.0);
        median.push(2.0);
        assert_eq!(median.median(), Some(2.0));
    }

    #[test]
    fn test_median_even_number_of_values() {
        let mut median = RollingMedian::<f64>::new(5);
        median.push(1.0);
        median.push(3.0);
        median.push(2.0);
        median.push(4.0);
        assert_eq!(median.median(), Some(2.5));
    }

    #[test]
    fn test_median_window_sliding() {
        let mut median = RollingMedian::<f64>::new(3);
        median.push(1.0);
        median.push(2.0);
        median.push(3.0);
        assert_eq!(median.median(), Some(2.0));

        median.pop(1.0);
        median.push(4.0);
        assert_eq!(median.median(), Some(3.0));

        median.pop(2.0);
        median.push(5.0);
        assert_eq!(median.median(), Some(4.0));
    }

    #[test]
    fn test_median_with_duplicates() {
        let mut median = RollingMedian::<f64>::new(5);
        median.push(1.0);
        median.push(2.0);
        median.push(2.0);
        median.push(3.0);
        median.push(4.0);
        assert_eq!(median.median(), Some(2.0));

        median.pop(1.0);
        median.push(5.0);
        assert_eq!(median.median(), Some(3.0));
    }

    #[test]
    fn test_median_large_window() {
        let mut median = RollingMedian::<f64>::new(100);
        for i in 0..100 {
            median.push(i as f64);
        }
        assert_eq!(median.median(), Some(49.5));

        median.pop(0.0);
        median.push(100.0);
        assert_eq!(median.median(), Some(50.5));
    }

    #[test]
    fn test_median_all_same_values() {
        let mut median = RollingMedian::<f64>::new(5);
        for _ in 0..5 {
            median.push(7.0);
        }
        assert_eq!(median.median(), Some(7.0));

        median.pop(7.0);
        median.push(7.0);
        assert_eq!(median.median(), Some(7.0));
    }

    #[test]
    fn test_median_with_extreme_values() {
        let mut median = RollingMedian::<f64>::new(3);
        median.push(1.0e-10);
        median.push(0.0);
        median.push(0.0);
        assert_eq!(median.median(), Some(0.0));
    }

    #[test]
    fn test_median_with_negative_values() {
        let mut median = RollingMedian::<f64>::new(5);
        median.push(-5.0);
        median.push(-3.0);
        median.push(-1.0);
        median.push(2.0);
        median.push(4.0);
        assert_eq!(median.median(), Some(-1.0));
    }

    #[test]
    fn test_median_alternating_add_remove() {
        let mut median = RollingMedian::<f64>::new(3);

        median.push(1.0);
        median.push(2.0);
        median.push(3.0);
        assert_eq!(median.median(), Some(2.0));

        median.pop(1.0);
        median.push(4.0); // Window: [2, 3, 4]
        assert_eq!(median.median(), Some(3.0));

        median.pop(2.0);
        median.push(5.0); // Window: [3, 4, 5]
        assert_eq!(median.median(), Some(4.0));

        median.pop(3.0);
        median.push(1.0); // Window: [4, 5, 1]
        assert_eq!(median.median(), Some(4.0));

        median.pop(4.0);
        median.push(2.0); // Window: [5, 1, 2]
        assert_eq!(median.median(), Some(2.0));
    }

    #[test]
    fn test_median_with_capacity_one() {
        let mut median = RollingMedian::<f64>::new(1);

        median.push(5.0);
        assert_eq!(median.median(), Some(5.0));

        median.pop(5.0);
        median.push(10.0);
        assert_eq!(median.median(), Some(10.0));

        median.pop(10.0);
        median.push(15.0);
        assert_eq!(median.median(), Some(15.0));
    }

    #[test]
    fn test_median_reset() {
        let mut median = RollingMedian::<f64>::new(5);

        median.push(1.0);
        median.push(2.0);
        median.push(3.0);
        assert_eq!(median.median(), Some(2.0));

        median.reset();
        assert_eq!(median.median(), None);
        assert_eq!(median.len(), 0);
        assert!(median.is_empty());

        median.push(10.0);
        median.push(20.0);
        assert_eq!(median.median(), Some(15.0));
    }

    #[test]
    fn test_median_pop_nonexistent_value() {
        let mut median = RollingMedian::<f64>::new(5);

        median.push(1.0);
        median.push(2.0);

        let result = median.pop(3.0);
        assert!(!result);
        assert_eq!(median.median(), Some(1.5));
        assert_eq!(median.len(), 2);
    }

    #[test]
    fn test_median_multiple_pops() {
        let mut median = RollingMedian::<f64>::new(5);

        median.push(1.0);
        median.push(2.0);
        median.push(3.0);
        median.push(4.0);
        median.push(5.0);

        assert!(median.pop(1.0));
        assert!(median.pop(3.0));
        assert!(median.pop(5.0));

        assert_eq!(median.median(), Some(3.0));
        assert_eq!(median.len(), 2);
    }

    #[test]
    fn test_median_infinite_loop_prevention() {
        let mut median = RollingMedian::<f64>::new(3);

        median.push(1.0);
        median.push(2.0);
        median.push(3.0);

        let _ = median.mark_for_removal(OrderedFloat(1.0));
        let _ = median.mark_for_removal(OrderedFloat(2.0));
        let _ = median.mark_for_removal(OrderedFloat(3.0));

        assert_eq!(median.median(), None);
    }
}
