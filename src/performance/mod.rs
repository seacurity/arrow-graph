// TODO: Fix compilation errors in these modules
pub mod simd;
pub mod memory;
// pub mod parallel; // Has compilation issues
// pub mod profiling; // Has compilation issues  
// pub mod benchmarks; // Has compilation issues
pub mod simple_benchmarks;

pub use simd::{SIMDGraphOps, VectorizedComputation};
pub use memory::{MemoryMappedGraph, MemoryPool};
// pub use parallel::{ParallelGraphProcessor, ThreadPool, WorkStealingQueue}; // Has compilation issues
// pub use profiling::{PerformanceProfiler, MetricsCollector, BenchmarkRunner}; // Has compilation issues  
// pub use benchmarks::{GraphBenchmarkSuite, BenchmarkSuiteConfig, TestGraph}; // Has compilation issues
pub use simple_benchmarks::{SimpleBenchmark, SimpleBenchmarkResult, run_all_benchmarks, print_results};