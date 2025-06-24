pub mod incremental;
pub mod algorithms;
pub mod detection;

pub use incremental::{IncrementalGraphProcessor, UpdateOperation, UpdateResult, IncrementalStats};
pub use algorithms::{StreamingAlgorithm, StreamingPageRank, StreamingConnectedComponents};
pub use detection::{GraphChangeDetector, AnomalyEvent, AnomalyType, GraphChange, ChangeType};