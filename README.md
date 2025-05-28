# ta-statistics

![Tests](https://github.com/l33tquant/ta-statistics/actions/workflows/ci.yml/badge.svg?branch=main)
[![Crate](https://img.shields.io/crates/v/ta-statistics.svg)](https://crates.io/crates/ta-statistics)
[![Documentation](https://docs.rs/ta-statistics/badge.svg)](https://docs.rs/ta-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A no_std, high-performance Rust library for computing rolling statistical measures on time series data, specifically designed for statistical analysis in backtesting and live trading systems.

## Features

- **no_std compatible**: Can be used in resource-constrained environments without the standard library
- **zero-runtime-allocation design**: Optimized for backtesting and algorithmic trading systems with minimal runtime overhead
- **Generic numeric support**: Works with any float type via the `num-traits` interface
- **Rolling window computations**: Efficiently calculates statistics over fixed-size windows
- **Comprehensive statistical measures**: Over 25 different statistical functions for both single and paired time series

### Single Time Series Statistics

| Category | Functions |
|----------|-----------|
| Basic Statistics | Sum, Mean, Mode, Median, Min, Max |
| Dispersion & Volatility | Variance, Standard Deviation, Mean Absolute Deviation, Median Absolute Deviation, IQR |
| Distribution Analysis | Z-Score, Skewness, Kurtosis, Quantile |
| Regression & Trend | Linear Regression (Slope/Intercept/Angle), Linear Fit |
| Trading-Specific | Drawdown, Maximum Drawdown, Percent Change, Log Return, Rolling Diff |

### Paired Time Series Statistics

| Category | Functions |
|----------|-----------|
| Relationship Metrics | Covariance, Correlation, Beta |
| Auxiliary Calculations | Mean Product, Mean of Squares |

## Installation

```bash
cargo add ta-statistics
```

Or add this to your `Cargo.toml`:

```toml
[dependencies]
ta-statistics = "*"
```

Replace `*` with the latest version number.

## Quick Start

For single statistics (like mean):

```rust
use ta_statistics::SingleStatistics;
let mut stats = SingleStatistics::new(20);
stats.next(105.43).mean();
```

For paired statistics (like correlation):

```rust
use ta_statistics::PairedStatistics;
let mut stats = PairedStatistics::new(20);
stats.next((105.43, 23.67)).corr();
```

## Use Cases

- **Technical Indicators**: Build standard and custom technical indicators based on statistical metrics
- **Alpha Generation**: Create statistical arbitrage models using correlation and covariance
- **Risk Management**: Monitor drawdowns, volatility, and beta for position sizing and risk control
- **Performance Analysis**: Calculate return statistics and risk-adjusted metrics for strategy evaluation
- **Market Regime Detection**: Use distributional statistics like skewness and kurtosis to identify market regimes

## Performance Considerations

- Memory usage is proportional to the window size
- Delta Degrees of Freedom correction can be applied with `set_ddof(true)` for sample statistics
- Uses KahanBabuskaNeumaier algorithm for compensated summation to prevent catastrophic cancellation in floating-point operations, ensuring numerical stability in rolling calculations
- Min and max are optimized with O(1) lookup and amortized O(1) insertion time using monotonic queue data structure

## Example: Real-time Volatility Analysis

```rust
use ta_statistics::SingleStatistics;

/// Calculate normalized volatility and detect regime changes in live market data
fn analyze_volatility_regime(
    price: f64, 
    stats: &mut SingleStatistics<f64>, 
    volatility_threshold: f64
) -> Option<VolatilityRegime> {
    // Update statistics with the latest price
    stats.next(price);
 
    // Calculate current volatility metrics
    let std_dev = stats.stddev()?;
    let kurt = stats.kurt()?;
    let skew = stats.skew()?;
    
    // Detect volatility regime
    let regime = if kurt > 3.0 && std_dev.abs() > volatility_threshold {
        // Fat tails and high volatility indicate stressed markets
        VolatilityRegime::Stressed
    } else if skew < -0.5 && std_dev > volatility_threshold * 0.7 {
        // Negative skew with moderately high volatility suggests caution
        VolatilityRegime::Cautious
    } else if std_dev < volatility_threshold * 0.5 && kurt < 3.0 {
        // Low volatility and normal kurtosis indicate calm markets
        VolatilityRegime::Normal
    } else {
        // Default regime when conditions are mixed
        VolatilityRegime::Transition
    };
    
    Some(regime)
}

enum VolatilityRegime {
    Normal,      // Low volatility, efficient price discovery
    Cautious,    // Increasing volatility, potential regime change
    Stressed,    // High volatility, fat tails, risk of extreme moves
    Transition,  // Mixed signals, regime in transition
}
```

## Example: Correlation-Based Pair Monitoring

```rust
use ta_statistics::PairedStatistics;

/// Monitor correlation stability between two instruments in real-time
/// and detect statistically significant relationship breakdowns
fn monitor_pair_relationship(
    returns: (f64, f64),
    stats: &mut PairedStatistics<f64>,
    historical_corr: f64,
    z_threshold: f64,
) -> Option<PairStatus> {
    // Update statistics with latest paired returns
    stats.next(returns);
     
    // Get current correlation and beta
    let current_corr = stats.corr()?;
    let current_beta = stats.beta()?;
    
    // Calculate Z-score of correlation deviation from historical norm
    // (simplified - in practice would use Fisher transformation)
    let corr_deviation = (current_corr - historical_corr).abs();
    let stddev_estimate = (1.0 - historical_corr * historical_corr) / 
                         (stats.period() as f64).sqrt();
    let correlation_z = corr_deviation / stddev_estimate;
    
    // Determine pair status based on statistical significance
    let status = if correlation_z > z_threshold {
        // Statistically significant breakdown in correlation
        PairStatus::RelationshipBreakdown {
            z_score: correlation_z,
            current_corr,
        }
    } else if current_beta.abs() > 1.5 * historical_corr.abs() {
        // Beta has increased but correlation remains stable
        // Indicates changing volatility dynamics
        PairStatus::IncreasedSensitivity {
            beta: current_beta,
            corr: current_corr,
        }
    } else {
        // Relationship is stable
        PairStatus::Stable {
            beta: current_beta,
            corr: current_corr,
        }
    };
    
    Some(status)
}

enum PairStatus {
    Stable { beta: f64, corr: f64 },
    RelationshipBreakdown { z_score: f64, current_corr: f64 },
    IncreasedSensitivity { beta: f64, corr: f64 },
}
```

## 📚 Documentation

For complete API documentation, examples, and explanations, visit:
[https://docs.rs/ta-statistics](https://docs.rs/ta-statistics)

## License

This project is licensed under the [MIT License](./LICENSE) - see the license file for details.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the license, shall be licensed as above, without any additional terms or conditions.

<div align="center">

### Disclaimer

*This software is for informational purposes only. It is not intended as trading or investment advice.*

</div>