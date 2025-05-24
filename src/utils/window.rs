use num_traits::Float;

use alloc::boxed::Box;

use core::{cmp::Ordering, fmt::Debug};

/// A fixed-size circular buffer that stores a sequence of values.
///
/// The buffer has a fixed size and can store a maximum of `period` values.
/// When the buffer is full, new values overwrite the oldest values in the buffer.
#[derive(Debug, Clone)]
pub struct Window<T> {
    /// The buffer that stores the values   
    buf: Box<[T]>,
    /// The period of the window
    period: usize,
    /// The current position in the buffer
    pos: usize,
    /// Whether the buffer has hit full once
    full: bool,
}

impl<T> Window<T> {
    /// Creates a new window with the specified period, returns Error if period is 0
    pub fn new(period: usize) -> Self
    where
        T: Default + Clone,
    {
        assert!(period > 0, "period can not be zero");

        Self {
            buf: vec![T::default(); period].into_boxed_slice(),
            pos: 0,
            full: false,
            period,
        }
    }

    /// Clears the buffer, resetting its state.
    pub fn reset(&mut self)
    where
        T: Default + Copy,
    {
        self.buf.fill(T::default());
        self.pos = 0;
        self.full = false;
    }

    /// Returns `true` if the buffer has hit full once
    ///
    /// # Returns
    ///
    /// * `bool` - True if the buffer has hit full once
    pub const fn is_full(&self) -> bool {
        self.full
    }

    /// Returns the current number of elements in the buffer.
    ///
    /// # Returns
    ///
    /// * `usize` - The current number of elements in the buffer
    pub const fn len(&self) -> usize {
        if self.full { self.period } else { self.pos }
    }

    /// Returns the current position in the buffer.
    ///
    /// # Returns
    ///
    /// * `usize` - The current position in the buffer
    pub const fn index(&self) -> usize {
        self.pos
    }

    /// Pushes a new value into the window and returns the value that was evicted
    ///
    /// # Arguments
    ///
    /// * `value` - The value to push into the window
    ///
    /// # Returns
    ///
    /// * `T` - The value that was evicted from the window
    pub const fn next(&mut self, value: T) -> T {
        let prev = core::mem::replace(&mut self.buf[self.pos], value);

        self.pos = (self.pos + 1) % self.period;

        if self.pos == 0 {
            self.full = true;
        }
        prev
    }

    /// Returns an iterator over the elements in logical (oldest to newest) order.
    ///
    /// # Returns
    ///
    /// * `impl Iterator<Item = &T>` - An iterator over the elements in the buffer
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len()).map(move |i| {
            let start = if self.full { self.pos } else { 0 };
            let idx = (start + i) % self.period;
            &self.buf[idx]
        })
    }

    /// Returns an iterator over the elements in the buffer
    ///
    /// # Returns
    ///
    /// * `impl Iterator<Item = &mut T>` - An iterator over the elements in the buffer
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.buf.iter_mut()
    }

    /// Returns a slice of the elements in the buffer   .
    ///
    /// # Returns
    ///
    /// * `&[T]` - A slice of the elements in the buffer
    pub fn as_slice(&self) -> &[T] {
        self.buf.as_ref()
    }

    /// Copies the elements from the slice into the buffer.
    ///
    /// # Arguments
    ///
    /// * `slice` - A slice of elements to copy into the buffer
    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        self.buf.copy_from_slice(slice)
    }

    /// Sorts the elements in the buffer in ascending order.
    ///
    /// # Returns
    ///
    /// * `&[T]` - A slice of the sorted elements in the buffer
    pub fn sort(&mut self) -> &[T]
    where
        T: PartialOrd,
    {
        self.buf
            .as_mut()
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        self.buf.as_ref()
    }

    /// Returns the maximum value in the buffer.
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The maximum value in the buffer, or `None` if the buffer is empty
    pub fn max(&self) -> Option<T>
    where
        T: Float,
    {
        self.buf.iter().copied().reduce(T::max)
    }

    /// Returns the minimum value in the buffer.
    ///
    /// # Returns
    ///
    /// * `Option<T>` - The minimum value in the buffer, or `None` if the buffer is empty
    pub fn min(&self) -> Option<T>
    where
        T: Float,
    {
        self.buf.iter().copied().reduce(T::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_next() {
        let mut window = Window::new(3);
        assert_eq!(window.next(1.0), 0.0);
        assert_eq!(window.next(2.0), 0.0);
        assert_eq!(window.next(3.0), 0.0);
        assert_eq!(window.next(4.0), 1.0);
        assert_eq!(window.next(5.0), 2.0);
    }

    #[test]
    fn test_window_is_full() {
        let mut window = Window::new(3);
        assert!(!window.is_full());
        window.next(1.0);
        assert!(!window.is_full());
        window.next(2.0);
        assert!(!window.is_full());
        window.next(3.0);
        assert!(window.is_full());
    }

    #[test]
    fn test_window_reset() {
        let mut window = Window::new(3);
        window.next(1.0);
        window.next(2.0);
        window.next(3.0);
        assert!(window.is_full());
        window.reset();
        assert!(!window.is_full());
    }
}
