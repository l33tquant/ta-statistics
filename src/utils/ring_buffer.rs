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
    pub fn new(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        vec.resize_with(capacity, T::default);
        Self {
            data: vec.into_boxed_slice(),
            index: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    pub fn push(&mut self, value: T) -> Option<T> {
        let cap = self.capacity();

        if self.is_full() {
            let overwritten = core::mem::replace(&mut self.data[self.index], value);
            self.index = (self.index + 1) % cap; // move head forward
            Some(overwritten)
        } else {
            let insert_at = (self.index + self.len) % cap;
            self.data[insert_at] = value;
            self.len += 1;
            None
        }
    }

    pub fn reset(&mut self) {
        self.index = 0;
        self.len = 0;
        self.data.fill(T::default());
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| &self.data[(self.index + i) % self.capacity()])
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
