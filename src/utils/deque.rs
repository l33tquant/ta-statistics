use alloc::boxed::Box;

/// A fixed-size double-ended queue
///
/// Maintains efficient insertions and overwriting behavior in a rolling window context.
#[derive(Debug, Clone)]
pub struct Deque<T> {
    /// The buffer with fixed capacity and allocated on the heap.
    buf: Box<[T]>,
    /// The capacity of the deque
    cap: usize,
    /// The index of the front element in the buffer
    front: usize,
    /// The index of the back element in the buffer
    back: usize,
    /// The current number of elements stored in the deque
    len: usize,
}

impl<T> Deque<T>
where
    T: Default + Clone,
{
    /// Creates a new `Deque` instance with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The capacity of the deque
    ///
    /// # Returns
    ///
    /// * `Self` - The `Deque` instance
    #[inline]
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "capacity must be > 0");
        Self {
            buf: vec![T::default(); cap].into_boxed_slice(),
            cap,
            front: 0,
            back: 0,
            len: 0,
        }
    }

    /// Returns true if the deque is empty
    ///
    /// # Returns
    ///
    /// * `bool` - True if the deque is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the deque is full
    ///
    /// # Returns
    ///
    /// * `bool` - True if the deque is full
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len == self.cap
    }

    /// Returns the current number of elements stored in the deque
    ///
    /// # Returns
    ///
    /// * `usize` - The current number of elements stored in the deque
    #[inline]
    #[allow(dead_code)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns the capacity of the deque
    ///
    /// # Returns
    ///
    /// * `usize` - The capacity of the deque
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Resets the deque, clearing all elements and resetting the indices
    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.buf.fill(T::default());
        self.front = 0;
        self.back = 0;
        self.len = 0;
        self
    }

    /// Pushes a new element to the back of the deque
    ///
    /// If the deque is full, the front element is evicted
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push to the back of the deque
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The evicted element if the deque was full, otherwise None
    #[inline]
    pub fn push_back(&mut self, value: T) -> Option<T> {
        let evicted = if self.is_full() {
            self.front = (self.front + 1) % self.cap;
            Some(core::mem::replace(&mut self.buf[self.back], value))
        } else {
            self.buf[self.back] = value;
            self.len += 1;
            None
        };

        self.back = (self.back + 1) % self.cap;

        evicted
    }

    /// Pops the element from the back of the deque
    ///
    /// If the deque is empty, returns None
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The element at the back of the deque, if it exists
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        self.back = if self.back == 0 {
            self.cap - 1
        } else {
            self.back - 1
        };

        self.len -= 1;
        Some(core::mem::take(&mut self.buf[self.back]))
    }

    /// Pops the element from the front of the deque
    ///
    /// If the deque is empty, returns None
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The element at the front of the deque, if it exists
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let value = core::mem::take(&mut self.buf[self.front]);
        self.front = (self.front + 1) % self.cap;
        self.len -= 1;

        Some(value)
    }

    /// Returns a reference to the front element of the deque
    ///
    /// If the deque is empty, returns None
    ///
    /// # Returns
    ///
    /// * `Option<&T>` - A reference to the front element of the deque, if it exists
    #[inline]
    pub const fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a reference to the back element of the deque
    ///
    /// If the deque is empty, returns None
    ///
    /// # Returns
    ///
    /// * `Option<&T>` - A reference to the back element of the deque, if it exists
    pub const fn back(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            let idx = if self.back == 0 {
                self.cap - 1
            } else {
                self.back - 1
            };
            Some(&self.buf[idx])
        }
    }

    /// Returns a reference to the element at the specified index from the front of the queue
    ///
    /// If the index is out of bounds, returns None
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the element to retrieve
    ///
    /// # Returns
    ///
    /// * `Option<&T>` - A reference to the element at the specified index, if it exists
    #[inline]
    pub const fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len {
            None
        } else {
            let idx = (self.front + i) % self.cap;
            Some(&self.buf[idx])
        }
    }

    /// Returns a mutable reference to the element at the specified index from the front of the queue
    ///
    /// If the index is out of bounds, returns None
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the element to retrieve
    ///
    /// # Returns
    ///
    /// * `Option<&mut T>` - A mutable reference to the element at the specified index, if it exists
    #[inline]
    #[allow(dead_code)]
    pub const fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        if i >= self.len {
            None
        } else {
            let idx = (self.front + i) % self.cap;
            Some(&mut self.buf[idx])
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| &self.buf[(self.front + i) % self.cap])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_front_back() {
        let mut deque = Deque::new(3);

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        assert_eq!(deque.front(), Some(&1));
        assert_eq!(deque.back(), Some(&3));

        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.pop_back(), Some(3));
        assert_eq!(deque.pop_front(), Some(2));
        assert!(deque.is_empty());
    }

    #[test]
    fn test_overwrite_on_full() {
        let mut deque = Deque::new(2);

        deque.push_back(10);
        deque.push_back(20);
        assert_eq!(deque.push_back(30), Some(10));

        assert_eq!(deque.len(), 2);
        assert_eq!(deque.front(), Some(&20));
        assert_eq!(deque.back(), Some(&30));
    }

    #[test]
    fn test_wraparound_behavior() {
        let mut deque = Deque::new(3);

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);
        deque.pop_front();

        deque.push_back(4);

        assert_eq!(deque.front(), Some(&2));
        assert_eq!(deque.back(), Some(&4));
        assert_eq!(deque.get(0), Some(&2));
        assert_eq!(deque.get(1), Some(&3));
        assert_eq!(deque.get(2), Some(&4));
    }

    #[test]
    fn test_get() {
        let mut deque = Deque::new(5);
        deque.push_back(5);
        deque.push_back(6);
        deque.push_back(7);

        assert_eq!(deque.get(0), Some(&5));
        assert_eq!(deque.get(1), Some(&6));
        assert_eq!(deque.get(2), Some(&7));
        assert_eq!(deque.get(3), None);
    }

    #[test]
    fn test_reset() {
        let mut deque = Deque::new(3);

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);
        assert_eq!(deque.len(), 3);

        deque.reset();
        assert!(deque.is_empty());
        assert_eq!(deque.front(), None);
        assert_eq!(deque.back(), None);

        deque.push_back(42);
        assert_eq!(deque.front(), Some(&42));
    }

    #[test]
    fn test_is_empty_is_full() {
        let mut deque = Deque::new(2);
        assert!(deque.is_empty());
        assert!(!deque.is_full());

        deque.push_back(1);
        assert!(!deque.is_empty());
        assert!(!deque.is_full());

        deque.push_back(2);
        assert!(deque.is_full());

        deque.pop_back();
        assert!(!deque.is_full());
    }

    #[test]
    fn test_capacity_one() {
        let mut deque = Deque::new(1);
        assert!(deque.is_empty());
        assert_eq!(deque.capacity(), 1);

        deque.push_back(10);
        assert_eq!(deque.len(), 1);
        assert!(deque.is_full());

        assert_eq!(deque.front(), Some(&10));
        assert_eq!(deque.back(), Some(&10));

        assert_eq!(deque.push_back(20), Some(10));
        assert_eq!(deque.front(), Some(&20));
        assert_eq!(deque.back(), Some(&20));

        assert_eq!(deque.pop_front(), Some(20));
        assert!(deque.is_empty());
    }

    #[test]
    fn test_pop_empty() {
        let mut deque = Deque::<usize>::new(2);
        assert_eq!(deque.pop_front(), None);
        assert_eq!(deque.pop_back(), None);
    }

    #[test]
    fn test_multiple_overwrites() {
        let mut deque = Deque::new(3);
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        assert_eq!(deque.push_back(4), Some(1));
        assert_eq!(deque.push_back(5), Some(2));

        assert_eq!(deque.get(0), Some(&3));
        assert_eq!(deque.get(1), Some(&4));
        assert_eq!(deque.get(2), Some(&5));
        assert!(deque.is_full());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let mut deque = Deque::new(2);
        deque.push_back(10);
        assert_eq!(deque.get(1), None);
        assert_eq!(deque.get_mut(1), None);

        if let Some(val) = deque.get_mut(0) {
            *val = 20;
        }
        assert_eq!(deque.get(0), Some(&20));
    }

    #[test]
    fn test_reset_partial_fill() {
        let mut deque = Deque::new(3);
        deque.push_back(1);
        deque.push_back(2);
        deque.reset();

        assert!(deque.is_empty());
        assert_eq!(deque.len(), 0);
        assert_eq!(deque.front(), None);
        assert_eq!(deque.back(), None);
    }

    #[test]
    fn test_push_pop_interleaved() {
        let mut deque = Deque::new(2);
        assert_eq!(deque.push_back(1), None);
        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.push_back(2), None);
        assert_eq!(deque.pop_back(), Some(2));
        assert!(deque.is_empty());
    }

    #[test]
    fn test_alternating_operations() {
        let mut deque = Deque::new(3);

        deque.push_back(1);
        deque.push_back(2);
        assert_eq!(deque.len(), 2);

        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.len(), 1);
        deque.push_back(3);
        deque.push_back(4);
        assert_eq!(deque.len(), 3);
        assert_eq!(deque.pop_back(), Some(4));
        assert_eq!(deque.len(), 2);
        deque.push_back(5);

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.front(), Some(&2));
        assert_eq!(deque.back(), Some(&5));
    }

    #[test]
    fn test_capacity_one_operations() {
        let mut deque = Deque::new(1);

        assert_eq!(deque.push_back(10), None);
        assert_eq!(deque.push_back(20), Some(10));

        assert_eq!(deque.front(), Some(&20));
        assert_eq!(deque.back(), Some(&20));

        deque.pop_front();
        assert!(deque.is_empty());

        deque.push_back(30);
        assert_eq!(deque.pop_back(), Some(30));
    }

    #[test]
    fn test_multiple_wraps() {
        let mut deque = Deque::new(3);

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        for i in 0..10 {
            deque.pop_front();
            deque.push_back(4 + i);
        }

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.front(), Some(&11));
        assert_eq!(deque.get(1), Some(&12));
        assert_eq!(deque.back(), Some(&13));
    }

    #[test]
    fn test_front_back_empty() {
        let deque: Deque<i32> = Deque::new(5);

        assert_eq!(deque.front(), None);
        assert_eq!(deque.back(), None);
    }

    #[test]
    fn test_get_mut_modification() {
        let mut deque = Deque::new(3);

        deque.push_back(10);
        deque.push_back(20);

        if let Some(value) = deque.get_mut(1) {
            *value = 25;
        }

        assert_eq!(deque.get(1), Some(&25));
    }
}
