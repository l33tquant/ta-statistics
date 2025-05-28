use core::cmp::Ordering;

use alloc::{boxed::Box, vec::Vec};

/// A fixed-size, circular buffer (ring buffer) for storing a sliding window of elements.
///
/// Maintains efficient insertions and overwriting behavior in a rolling window context.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    /// The underlying buffer holding the elements.
    /// The buffer has a fixed capacity and is allocated on the heap.
    data: Box<[T]>,
    /// The index of the oldest element in the buffer (the "head").
    /// This is where the next element will be overwritten when the buffer is full.
    index: usize,
    /// The current number of elements stored in the buffer.
    /// Always less than or equal to `data.len()`.
    len: usize,
}

impl<T: Default + Copy> RingBuffer<T> {
    /// Creates a new `RingBuffer` instance with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The capacity of the ring buffer
    ///
    /// # Returns
    ///
    /// * `Self` - The `RingBuffer` instance
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        vec.resize_with(capacity, T::default);
        Self {
            data: vec.into_boxed_slice(),
            index: 0,
            len: 0,
        }
    }

    /// Returns the capacity of the ring buffer
    ///
    /// # Returns
    ///
    /// * `usize` - The capacity of the ring buffer
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns the current number of elements stored in the ring buffer
    ///
    /// # Returns
    ///
    /// * `usize` - The current number of elements stored in the ring buffer
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the ring buffer is full
    ///
    /// # Returns
    ///
    /// * `bool` - True if the ring buffer is full
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    /// Pushes a new value into the ring buffer, overwriting the oldest value if the buffer is full
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push into the ring buffer
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The overwritten value if the buffer was full, otherwise None
    #[inline]
    pub const fn push(&mut self, value: T) -> Option<T> {
        let cap = self.capacity();

        if self.is_full() {
            let overwritten = core::mem::replace(&mut self.data[self.index], value);
            self.index = (self.index + 1) % cap;
            Some(overwritten)
        } else {
            let insert_at = (self.index + self.len) % cap;
            self.data[insert_at] = value;
            self.len += 1;
            None
        }
    }

    /// Resets the ring buffer to its initial state
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The ring buffer object
    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.index = 0;
        self.len = 0;
        self.data.fill(T::default());
        self
    }

    /// Returns an iterator over the elements in the ring buffer
    ///
    /// # Returns
    ///
    /// * `impl Iterator<Item = &T>` - An iterator over the elements in the ring buffer
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| &self.data[(self.index + i) % self.capacity()])
    }

    /// Returns an iterator over the elements in the buffer
    ///
    /// # Returns
    ///
    /// * `impl Iterator<Item = &mut T>` - An iterator over the elements in the buffer
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Returns a slice of the elements in the buffer   .
    ///
    /// # Returns
    ///
    /// * `&[T]` - A slice of the elements in the buffer
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Copies the elements from the slice into the buffer
    ///
    /// # Arguments
    ///
    /// * `slice` - The slice to copy from
    ///
    /// # Returns
    ///
    /// * `&mut Self` - The ring buffer object
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        self.data.copy_from_slice(slice)
    }

    /// Sorts the elements in the buffer
    ///
    /// # Returns
    ///
    /// * `&[T]` - A slice of the sorted elements in the buffer
    #[inline]
    pub fn sort(&mut self) -> &[T]
    where
        T: PartialOrd,
    {
        self.data
            .as_mut()
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        self.data.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn test_new_and_capacity() {
        let buf: RingBuffer<i32> = RingBuffer::new(4);
        assert_eq!(buf.capacity(), 4);
        assert_eq!(buf.len(), 0);
        assert!(!buf.is_full());
    }

    #[test]
    fn test_push_and_len() {
        let mut buf = RingBuffer::new(3);
        assert_eq!(buf.len(), 0);

        assert_eq!(buf.push(10), None);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.push(20), None);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.push(30), None);
        assert_eq!(buf.len(), 3);

        // Now buffer should be full
        assert!(buf.is_full());
    }

    #[test]
    fn test_push_overwrite() {
        let mut buf = RingBuffer::new(2);
        assert_eq!(buf.push(1), None);
        assert_eq!(buf.push(2), None);
        assert!(buf.is_full());

        // Next push overwrites the oldest element (1)
        let overwritten = buf.push(3);
        assert_eq!(overwritten, Some(1));
        assert_eq!(buf.len(), 2);

        // Next push overwrites element (2)
        let overwritten = buf.push(4);
        assert_eq!(overwritten, Some(2));

        // Buffer elements should be [3,4] in order
        let elems: Vec<_> = buf.iter().copied().collect();
        assert_eq!(elems, vec![3, 4]);
    }

    #[test]
    fn test_iter_order() {
        let mut buf = RingBuffer::new(3);
        buf.push(5);
        buf.push(6);
        let elems: Vec<_> = buf.iter().copied().collect();
        assert_eq!(elems, vec![5, 6]);

        buf.push(7);
        let elems: Vec<_> = buf.iter().copied().collect();
        assert_eq!(elems, vec![5, 6, 7]);

        // Overwrite oldest (5)
        buf.push(8);
        let elems: Vec<_> = buf.iter().copied().collect();
        assert_eq!(elems, vec![6, 7, 8]);
    }

    #[test]
    fn test_zero_capacity() {
        let result = std::panic::catch_unwind(|| {
            let mut buf: RingBuffer<i32> = RingBuffer::new(0);
            buf.push(1);
        });

        assert!(
            result.is_err(),
            "Pushing into zero-capacity buffer should panic or fail safely"
        );
    }

    #[test]
    fn test_full_cycle_push_pop() {
        let mut buf = RingBuffer::new(3);

        // Fill buffer
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert!(buf.is_full());

        // Overwrite in cycles
        assert_eq!(buf.push(4), Some(1));
        assert_eq!(buf.push(5), Some(2));
        assert_eq!(buf.push(6), Some(3));

        // Buffer elements are [4, 5, 6]
        let elems: Vec<_> = buf.iter().copied().collect();
        assert_eq!(elems, vec![4, 5, 6]);

        // Push more overwrites
        assert_eq!(buf.push(7), Some(4));
        assert_eq!(buf.push(8), Some(5));
    }

    #[test]
    fn test_reset_clears_internal_state() {
        let mut buf = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);

        buf.reset();

        // After reset, length should be zero, index back to 0
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.index, 0);

        // Pushing again should behave like from scratch
        buf.push(42);
        assert_eq!(buf.len(), 1);

        // Verify internal data is zeroed if T: Default + Copy = 0
        for i in 1..buf.capacity() {
            assert_eq!(buf.data[i], 0);
        }
    }
}
