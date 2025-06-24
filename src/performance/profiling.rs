use crate::error::Result;
use crate::graph::ArrowGraph;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};

/// Performance profiler for graph operations
/// Provides detailed timing and resource usage analysis
#[derive(Debug)]
pub struct PerformanceProfiler {
    metrics: Arc<Mutex<ProfilerMetrics>>,
    active_sessions: Arc<Mutex<HashMap<String, ProfilingSession>>>,
    config: ProfilerConfig,
}

/// Configuration for performance profiling
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub enable_memory_tracking: bool,
    pub enable_cpu_tracking: bool,
    pub enable_io_tracking: bool,
    pub sample_rate: Duration,
    pub max_sessions: usize,
    pub auto_save_interval: Option<Duration>,
    pub output_format: OutputFormat,
}

/// Output formats for profiling data
#[derive(Debug, Clone)]
pub enum OutputFormat {
    JSON,
    CSV,
    Flamegraph,
    Perf,
    Chrome, // Chrome DevTools format
}

/// Metrics collector for system-wide performance monitoring
#[derive(Debug)]
pub struct MetricsCollector {
    system_metrics: Arc<Mutex<SystemMetrics>>,
    graph_metrics: Arc<Mutex<GraphMetrics>>,
    collection_interval: Duration,
    running: Arc<Mutex<bool>>,
}

/// Benchmark runner for performance testing
#[derive(Debug)]
pub struct BenchmarkRunner {
    benchmarks: Vec<Box<dyn Benchmark>>,
    results: Vec<BenchmarkResult>,
    config: BenchmarkConfig,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub timeout: Duration,
    pub statistical_analysis: bool,
    pub compare_baselines: bool,
}

/// Profiling session for tracking specific operations
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    pub session_id: String,
    pub operation_name: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub memory_snapshots: Vec<MemorySnapshot>,
    pub cpu_samples: Vec<CpuSample>,
    pub io_events: Vec<IoEvent>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: u64,
    pub heap_used: usize,
    pub heap_total: usize,
    pub stack_used: usize,
    pub peak_usage: usize,
    pub allocations: usize,
    pub deallocations: usize,
}

/// CPU usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    pub timestamp: u64,
    pub cpu_percent: f64,
    pub user_time: Duration,
    pub system_time: Duration,
    pub idle_time: Duration,
    pub context_switches: u64,
}

/// I/O event tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoEvent {
    pub timestamp: u64,
    pub event_type: IoEventType,
    pub bytes: usize,
    pub duration: Duration,
    pub path: Option<String>,
}

/// Types of I/O events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoEventType {
    FileRead,
    FileWrite,
    NetworkRead,
    NetworkWrite,
    MemoryMap,
    MemoryUnmap,
}

/// System-level performance metrics
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub memory_available: usize,
    pub disk_io_read: u64,
    pub disk_io_write: u64,
    pub network_io_read: u64,
    pub network_io_write: u64,
    pub load_average: [f64; 3], // 1, 5, 15 minute averages
    pub uptime: Duration,
}

/// Graph-specific performance metrics
#[derive(Debug, Clone, Default)]
pub struct GraphMetrics {
    pub operations_per_second: f64,
    pub query_latency_p95: Duration,
    pub query_latency_p99: Duration,
    pub memory_per_node: f64,
    pub memory_per_edge: f64,
    pub cache_hit_ratio: f64,
    pub index_efficiency: f64,
    pub parallel_efficiency: f64,
}

/// Overall profiler metrics
#[derive(Debug, Clone, Default)]
pub struct ProfilerMetrics {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub total_samples: usize,
    pub overhead_percentage: f64,
    pub collection_errors: usize,
}

/// Benchmark trait for performance testing
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    fn setup(&mut self, graph: &ArrowGraph) -> Result<()>;
    fn run(&self, graph: &ArrowGraph) -> Result<BenchmarkResult>;
    fn teardown(&mut self) -> Result<()>;
    fn expected_performance(&self) -> Option<PerformanceExpectation>;
}

/// Expected performance metrics for comparison
#[derive(Debug, Clone)]
pub struct PerformanceExpectation {
    pub max_duration: Duration,
    pub max_memory: usize,
    pub min_throughput: f64,
    pub max_cpu_usage: f64,
}

/// Benchmark execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub duration: Duration,
    pub memory_peak: usize,
    pub memory_average: usize,
    pub cpu_usage: f64,
    pub throughput: f64,
    pub success: bool,
    pub error_message: Option<String>,
    pub iterations: usize,
    pub statistical_summary: Option<StatisticalSummary>,
}

/// Statistical analysis of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
    pub percentile_95: Duration,
    pub percentile_99: Duration,
    pub coefficient_of_variation: f64,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(ProfilerMetrics::default())),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Start a new profiling session
    pub fn start_session(&self, session_id: String, operation_name: String) -> Result<String> {
        let session = ProfilingSession {
            session_id: session_id.clone(),
            operation_name,
            start_time: Instant::now(),
            end_time: None,
            memory_snapshots: Vec::new(),
            cpu_samples: Vec::new(),
            io_events: Vec::new(),
            custom_metrics: HashMap::new(),
        };

        let mut sessions = self.active_sessions.lock();
        
        // Check session limit
        if sessions.len() >= self.config.max_sessions {
            return Err(crate::error::GraphError::graph_construction("Maximum sessions exceeded"));
        }

        sessions.insert(session_id.clone(), session);

        // Update metrics
        let mut metrics = self.metrics.lock();
        metrics.active_sessions = sessions.len();
        metrics.total_sessions += 1;

        Ok(session_id)
    }

    /// End a profiling session
    pub fn end_session(&self, session_id: &str) -> Result<ProfilingSession> {
        let mut sessions = self.active_sessions.lock();
        
        if let Some(mut session) = sessions.remove(session_id) {
            session.end_time = Some(Instant::now());
            
            // Update metrics
            let mut metrics = self.metrics.lock();
            metrics.active_sessions = sessions.len();
            
            Ok(session)
        } else {
            Err(crate::error::GraphError::graph_construction("Session not found"))
        }
    }

    /// Record a memory snapshot for a session
    pub fn record_memory_snapshot(&self, session_id: &str, snapshot: MemorySnapshot) -> Result<()> {
        let mut sessions = self.active_sessions.lock();
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.memory_snapshots.push(snapshot);
            
            // Update sample count
            let mut metrics = self.metrics.lock();
            metrics.total_samples += 1;
            
            Ok(())
        } else {
            Err(crate::error::GraphError::graph_construction("Session not found"))
        }
    }

    /// Record a CPU sample for a session
    pub fn record_cpu_sample(&self, session_id: &str, sample: CpuSample) -> Result<()> {
        let mut sessions = self.active_sessions.lock();
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.cpu_samples.push(sample);
            
            let mut metrics = self.metrics.lock();
            metrics.total_samples += 1;
            
            Ok(())
        } else {
            Err(crate::error::GraphError::graph_construction("Session not found"))
        }
    }

    /// Record an I/O event for a session
    pub fn record_io_event(&self, session_id: &str, event: IoEvent) -> Result<()> {
        let mut sessions = self.active_sessions.lock();
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.io_events.push(event);
            
            let mut metrics = self.metrics.lock();
            metrics.total_samples += 1;
            
            Ok(())
        } else {
            Err(crate::error::GraphError::graph_construction("Session not found"))
        }
    }

    /// Add custom metric to a session
    pub fn add_custom_metric(&self, session_id: &str, name: String, value: f64) -> Result<()> {
        let mut sessions = self.active_sessions.lock();
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.custom_metrics.insert(name, value);
            Ok(())
        } else {
            Err(crate::error::GraphError::graph_construction("Session not found"))
        }
    }

    /// Get current profiler metrics
    pub fn get_metrics(&self) -> ProfilerMetrics {
        self.metrics.lock().clone()
    }

    /// Export profiling data
    pub fn export_data(&self, format: OutputFormat) -> Result<String> {
        let sessions = self.active_sessions.lock();
        
        match format {
            OutputFormat::JSON => {
                let data = serde_json::to_string_pretty(&*sessions)?;
                Ok(data)
            }
            OutputFormat::CSV => {
                self.export_csv(&sessions)
            }
            OutputFormat::Flamegraph => {
                self.export_flamegraph(&sessions)
            }
            OutputFormat::Chrome => {
                self.export_chrome_format(&sessions)
            }
            _ => {
                Err(crate::error::GraphError::graph_construction("Unsupported export format"))
            }
        }
    }

    /// Profile a specific operation
    pub fn profile_operation<F, R>(&self, operation_name: &str, operation: F) -> Result<(R, ProfilingSession)>
    where
        F: FnOnce() -> Result<R>,
    {
        let session_id = format!("op_{}_{}", operation_name, chrono::Utc::now().timestamp_millis());
        
        // Start profiling
        self.start_session(session_id.clone(), operation_name.to_string())?;
        
        // Take initial snapshots
        if self.config.enable_memory_tracking {
            let snapshot = self.take_memory_snapshot()?;
            self.record_memory_snapshot(&session_id, snapshot)?;
        }
        
        // Execute operation
        let start_time = Instant::now();
        let result = operation()?;
        let duration = start_time.elapsed();
        
        // Take final snapshots
        if self.config.enable_memory_tracking {
            let snapshot = self.take_memory_snapshot()?;
            self.record_memory_snapshot(&session_id, snapshot)?;
        }
        
        // Add duration metric
        self.add_custom_metric(&session_id, "duration_ms".to_string(), duration.as_millis() as f64)?;
        
        // End session
        let session = self.end_session(&session_id)?;
        
        Ok((result, session))
    }

    // Private helper methods
    fn export_csv(&self, _sessions: &HashMap<String, ProfilingSession>) -> Result<String> {
        // Implement CSV export
        Ok("session_id,operation,duration_ms,memory_peak\n".to_string())
    }

    fn export_flamegraph(&self, _sessions: &HashMap<String, ProfilingSession>) -> Result<String> {
        // Implement flamegraph export
        Ok("flamegraph data would go here".to_string())
    }

    fn export_chrome_format(&self, _sessions: &HashMap<String, ProfilingSession>) -> Result<String> {
        // Implement Chrome DevTools format
        Ok(r#"{"traceEvents":[]}"#.to_string())
    }

    fn take_memory_snapshot(&self) -> Result<MemorySnapshot> {
        // In a real implementation, this would query system memory usage
        Ok(MemorySnapshot {
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            heap_used: 1024 * 1024,      // 1MB placeholder
            heap_total: 10 * 1024 * 1024, // 10MB placeholder
            stack_used: 64 * 1024,        // 64KB placeholder
            peak_usage: 2 * 1024 * 1024,  // 2MB placeholder
            allocations: 100,
            deallocations: 50,
        })
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(collection_interval: Duration) -> Self {
        Self {
            system_metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            graph_metrics: Arc::new(Mutex::new(GraphMetrics::default())),
            collection_interval,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start collecting metrics
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.lock();
        if *running {
            return Err(crate::error::GraphError::graph_construction("Collector already running"));
        }
        *running = true;

        // In a real implementation, this would start a background thread
        // that periodically collects system and graph metrics
        
        Ok(())
    }

    /// Stop collecting metrics
    pub fn stop(&self) -> Result<()> {
        let mut running = self.running.lock();
        *running = false;
        Ok(())
    }

    /// Get current system metrics
    pub fn get_system_metrics(&self) -> SystemMetrics {
        self.system_metrics.lock().clone()
    }

    /// Get current graph metrics
    pub fn get_graph_metrics(&self) -> GraphMetrics {
        self.graph_metrics.lock().clone()
    }

    /// Update graph metrics
    pub fn update_graph_metrics(&self, metrics: GraphMetrics) {
        *self.graph_metrics.lock() = metrics;
    }

    /// Collect current system metrics
    fn collect_system_metrics(&self) -> SystemMetrics {
        // In a real implementation, this would query the operating system
        // for actual CPU, memory, disk, and network usage
        SystemMetrics {
            cpu_usage: 45.0,
            memory_usage: 8 * 1024 * 1024 * 1024, // 8GB
            memory_available: 16 * 1024 * 1024 * 1024, // 16GB
            disk_io_read: 1024 * 1024,
            disk_io_write: 512 * 1024,
            network_io_read: 100 * 1024,
            network_io_write: 50 * 1024,
            load_average: [1.2, 1.5, 1.8],
            uptime: Duration::from_secs(3600 * 24), // 1 day
        }
    }
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            benchmarks: Vec::new(),
            results: Vec::new(),
            config,
        }
    }

    /// Add a benchmark to the runner
    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }

    /// Run all benchmarks
    pub fn run_benchmarks(&mut self, graph: &ArrowGraph) -> Result<Vec<BenchmarkResult>> {
        self.results.clear();

        for benchmark in &mut self.benchmarks {
            println!("Running benchmark: {}", benchmark.name());
            
            // Setup
            benchmark.setup(graph)?;
            
            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = benchmark.run(graph);
            }
            
            // Measurements
            let mut durations = Vec::new();
            let mut success_count = 0;
            let mut last_error = None;

            for _ in 0..self.config.measurement_iterations {
                let start = Instant::now();
                match benchmark.run(graph) {
                    Ok(mut result) => {
                        let duration = start.elapsed();
                        durations.push(duration);
                        result.duration = duration;
                        result.iterations = self.config.measurement_iterations;
                        
                        if self.config.statistical_analysis {
                            result.statistical_summary = Some(self.calculate_statistics(&durations));
                        }
                        
                        self.results.push(result);
                        success_count += 1;
                    }
                    Err(e) => {
                        last_error = Some(e.to_string());
                    }
                }
            }

            // Teardown
            benchmark.teardown()?;

            // Create summary result if there were failures
            if success_count == 0 {
                let failed_result = BenchmarkResult {
                    benchmark_name: benchmark.name().to_string(),
                    duration: Duration::from_secs(0),
                    memory_peak: 0,
                    memory_average: 0,
                    cpu_usage: 0.0,
                    throughput: 0.0,
                    success: false,
                    error_message: last_error,
                    iterations: self.config.measurement_iterations,
                    statistical_summary: None,
                };
                self.results.push(failed_result);
            }
        }

        Ok(self.results.clone())
    }

    /// Calculate statistical summary
    fn calculate_statistics(&self, durations: &[Duration]) -> StatisticalSummary {
        if durations.is_empty() {
            return StatisticalSummary {
                mean: Duration::from_secs(0),
                median: Duration::from_secs(0),
                std_dev: Duration::from_secs(0),
                min: Duration::from_secs(0),
                max: Duration::from_secs(0),
                percentile_95: Duration::from_secs(0),
                percentile_99: Duration::from_secs(0),
                coefficient_of_variation: 0.0,
            };
        }

        let mut sorted_durations = durations.to_vec();
        sorted_durations.sort();

        let mean = Duration::from_nanos(
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128
        );

        let median = sorted_durations[durations.len() / 2];
        let min = sorted_durations[0];
        let max = sorted_durations[durations.len() - 1];

        // Calculate percentiles
        let p95_idx = (durations.len() * 95 / 100).min(durations.len() - 1);
        let p99_idx = (durations.len() * 99 / 100).min(durations.len() - 1);
        let percentile_95 = sorted_durations[p95_idx];
        let percentile_99 = sorted_durations[p99_idx];

        // Calculate standard deviation
        let mean_nanos = mean.as_nanos() as f64;
        let variance = durations.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>() / durations.len() as f64;
        
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);
        let coefficient_of_variation = if mean_nanos > 0.0 {
            variance.sqrt() / mean_nanos
        } else {
            0.0
        };

        StatisticalSummary {
            mean,
            median,
            std_dev,
            min,
            max,
            percentile_95,
            percentile_99,
            coefficient_of_variation,
        }
    }
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_cpu_tracking: true,
            enable_io_tracking: false,
            sample_rate: Duration::from_millis(100),
            max_sessions: 100,
            auto_save_interval: Some(Duration::from_secs(300)), // 5 minutes
            output_format: OutputFormat::JSON,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measurement_iterations: 10,
            timeout: Duration::from_secs(60),
            statistical_analysis: true,
            compare_baselines: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.active_sessions, 0);
        assert_eq!(metrics.total_sessions, 0);
    }

    #[test]
    fn test_profiling_session() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let session_id = profiler.start_session(
            "test_session".to_string(),
            "test_operation".to_string(),
        ).unwrap();
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.active_sessions, 1);
        assert_eq!(metrics.total_sessions, 1);
        
        let session = profiler.end_session(&session_id).unwrap();
        assert_eq!(session.operation_name, "test_operation");
        assert!(session.end_time.is_some());
    }

    #[test]
    fn test_memory_snapshot() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let session_id = profiler.start_session(
            "test_session".to_string(),
            "test_operation".to_string(),
        ).unwrap();
        
        let snapshot = MemorySnapshot {
            timestamp: 12345,
            heap_used: 1024,
            heap_total: 2048,
            stack_used: 512,
            peak_usage: 1536,
            allocations: 10,
            deallocations: 5,
        };
        
        profiler.record_memory_snapshot(&session_id, snapshot).unwrap();
        
        let session = profiler.end_session(&session_id).unwrap();
        assert_eq!(session.memory_snapshots.len(), 1);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(Duration::from_millis(100));
        
        collector.start().unwrap();
        
        let system_metrics = collector.get_system_metrics();
        assert!(system_metrics.memory_usage > 0);
        
        collector.stop().unwrap();
    }

    #[test]
    fn test_benchmark_runner() {
        let config = BenchmarkConfig::default();
        let mut runner = BenchmarkRunner::new(config);
        
        assert_eq!(runner.benchmarks.len(), 0);
        assert_eq!(runner.results.len(), 0);
    }

    #[test]
    fn test_profile_operation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let (result, session) = profiler.profile_operation("test_op", || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        }).unwrap();
        
        assert_eq!(result, 42);
        assert_eq!(session.operation_name, "test_op");
        assert!(session.end_time.is_some());
        assert!(!session.custom_metrics.is_empty());
    }

    #[test]
    fn test_statistical_summary() {
        let config = BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(config);
        
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(120),
            Duration::from_millis(130),
            Duration::from_millis(140),
        ];
        
        let summary = runner.calculate_statistics(&durations);
        assert_eq!(summary.min, Duration::from_millis(100));
        assert_eq!(summary.max, Duration::from_millis(140));
        assert_eq!(summary.median, Duration::from_millis(120));
    }
}