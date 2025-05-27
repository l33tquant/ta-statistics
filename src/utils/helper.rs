use num_traits::Float;

/// Returns the median from a sorted slice
///
/// # Arguments
///
/// * `ss` - The sorted slice
///
/// # Returns
///
/// * `T` - The median
#[inline]
pub fn median_from_sorted_slice<T: Float>(ss: &[T]) -> T {
    let len = ss.len();
    let mid = len / 2;
    let _2 = T::one() + T::one();
    if len % 2 == 0 {
        (ss[mid - 1] + ss[mid]) / _2
    } else {
        ss[mid]
    }
}

/// Returns the quantile from a sorted slice
///
/// # Arguments
///
/// * `ss` - The sorted slice
/// * `q` - The quantile to calculate
/// * `period` - The period of the slice
///
/// # Returns
///
/// * `Option<T>` - The quantile, or `None` if the slice is empty
#[inline]
pub fn quantile_from_sorted_slice<T: Float>(ss: &[T], q: f64, period: usize) -> Option<T> {
    let pos = q * (period as f64 - 1.0);
    let lower_index = pos.floor() as usize;
    let upper_index = pos.ceil() as usize;

    if lower_index == upper_index {
        Some(ss[lower_index])
    } else {
        let lower_value = ss[lower_index];
        let upper_value = ss[upper_index];
        let weight = T::from(pos - lower_index as f64)?;

        T::from(lower_value + weight * (upper_value - lower_value))
    }
}
