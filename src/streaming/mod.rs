pub mod incremental;
pub mod algorithms;
pub mod detection;
pub mod temporal;
pub mod events;
pub mod approximate;
pub mod sampling;

pub use incremental::{IncrementalGraphProcessor, UpdateOperation, UpdateResult, IncrementalStats};
pub use algorithms::{StreamingAlgorithm, StreamingPageRank, StreamingConnectedComponents};
pub use detection::{GraphChangeDetector, AnomalyEvent, AnomalyType, GraphChange, ChangeType};
pub use temporal::{SlidingWindowProcessor, GraphSnapshot, TemporalAnalytics, TemporalConfig, TrendInfo, ChangePoint, EvolutionPattern};
pub use events::{EventDrivenProcessor, EventRule, EventTrigger, EventAction, ProcessedEvent, AlgorithmStats};
pub use approximate::{ApproximateGraphProcessor, HyperLogLog, CountMinSketch, BloomFilter, ApproximateMetrics, ApproximateConfig};
pub use sampling::{GraphSamplingProcessor, NodeSampler, EdgeSampler, SubgraphSampler, SamplingConfig, GraphSample, SampledSubgraph, ReservoirSampler};