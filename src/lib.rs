pub mod graph;
pub mod sql;
pub mod algorithms;
pub mod error;
pub mod streaming;
// TODO: Fix compilation errors in ML module
// pub mod ml;
pub mod performance;

pub use graph::{ArrowGraph, GraphIndexes, StreamingGraphProcessor, StreamingGraphSystem, StreamUpdate, UpdateResult, IncrementalAlgorithmProcessor};
pub use sql::GraphSqlExtension;
pub use algorithms::{GraphAlgorithm, AlgorithmParams};
pub use streaming::{IncrementalGraphProcessor, UpdateOperation, StreamingAlgorithm, StreamingPageRank, StreamingConnectedComponents, GraphChangeDetector, AnomalyEvent, AnomalyType, SlidingWindowProcessor, EventDrivenProcessor, ApproximateGraphProcessor, GraphSamplingProcessor};
pub use algorithms::components::{WeaklyConnectedComponents, StronglyConnectedComponents};
pub use algorithms::community::LeidenCommunityDetection;
pub use algorithms::aggregation::{TriangleCount, ClusteringCoefficient};
pub use algorithms::centrality::{PageRank, BetweennessCentrality, EigenvectorCentrality, ClosenessCentrality};
pub use algorithms::vectorized::{VectorizedPageRank, VectorizedDistanceCalculator, VectorizedBatchOperations};
pub use algorithms::sampling::{RandomWalk, Node2VecWalk, GraphSampling};
pub use error::{GraphError, Result};

pub mod prelude {
    pub use crate::graph::{ArrowGraph, GraphIndexes, StreamingGraphProcessor, StreamingGraphSystem, StreamUpdate, UpdateResult, IncrementalAlgorithmProcessor};
    pub use crate::sql::GraphSqlExtension;
    pub use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
    pub use crate::streaming::{IncrementalGraphProcessor, UpdateOperation, StreamingAlgorithm, StreamingPageRank, StreamingConnectedComponents, GraphChangeDetector, AnomalyEvent, AnomalyType, SlidingWindowProcessor, EventDrivenProcessor, ApproximateGraphProcessor, GraphSamplingProcessor};
    pub use crate::algorithms::components::{WeaklyConnectedComponents, StronglyConnectedComponents};
    pub use crate::algorithms::community::LeidenCommunityDetection;
    pub use crate::algorithms::aggregation::{TriangleCount, ClusteringCoefficient};
    pub use crate::algorithms::centrality::{PageRank, BetweennessCentrality, EigenvectorCentrality, ClosenessCentrality};
    pub use crate::algorithms::vectorized::{VectorizedPageRank, VectorizedDistanceCalculator, VectorizedBatchOperations};
    pub use crate::algorithms::sampling::{RandomWalk, Node2VecWalk, GraphSampling};
    pub use crate::error::{GraphError, Result};
}