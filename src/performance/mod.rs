// TODO: Fix compilation errors in these modules
// pub mod simd;
// pub mod memory;
// pub mod parallel;
// pub mod profiling;
// pub mod benchmarks;
pub mod simple_benchmarks;

// pub use simd::{SIMDGraphOps, VectorizedComputation};
// pub use memory::{MemoryMappedGraph, MemoryPool, CacheOptimizedStorage};
// pub use parallel::{ParallelGraphProcessor, ThreadPool, WorkStealingQueue};
// pub use profiling::{PerformanceProfiler, MetricsCollector, BenchmarkRunner};
// pub use benchmarks::{GraphBenchmarkSuite, BenchmarkSuiteConfig, TestGraph};
pub use simple_benchmarks::{SimpleBenchmark, SimpleBenchmarkResult, run_all_benchmarks, print_results};