## v0.2.6 (Jun 04, 2025)

- Added public exports for `minimum` and `maximum` modules to provide direct access to efficient rolling window extrema calculations

## v0.2.5 (Jun 02, 2025)

- Implemented a Red-Black Tree data structure with minimal unsafe code to efficiently compute median, quantiles, and interquartile range (IQR) with O(log n) time complexity for insertions, deletions, and lookups. Refactored Median/Mean Absolute Deviation (MAD) calculation from slice-based approach to Red-Black Tree implementation with O(n) complexity
- Minor refactoring and optimizations

## v0.2.4 (May 29, 2025)

- Optimized median with O(log n) time complexity for insertions and deletions, and O(1) median access using two balanced heaps with lazy deletions

## v0.2.3 (May 28, 2025)

- Optimized mode with O(1) lookup and amortized O(1) insertion/removal time using frequency bucket

## v0.2.2 (May 28, 2025)

- Optimized min and max with O(1) lookup and amortized O(1) insertion time using monotonic queue data structure

## v0.2.1 (May 27, 2025)

- Added recompute to both statistics
- Minor refactoring

## v0.2.0 (May 27, 2025)

- Implemented Kahan-Babuska-Neumaier algorithm for compensated summation to prevent catastrophic cancellation in floating-point calculations
- Added support for rolling higher-order moments (mean, variance, skewness, kurtosis) with numerically stable online computation

## v0.1.0 (May 24, 2025)

- Initial release.
