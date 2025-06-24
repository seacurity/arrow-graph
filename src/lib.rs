pub mod graph;
pub mod sql;
pub mod algorithms;
pub mod error;
pub mod streaming;
pub mod ml;
pub mod performance;
pub mod analytics; // New advanced analytics module for v0.7.0

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
// v0.7.0 Advanced Analytics Exports
pub use analytics::{
    AdvancedAnalyticsEngine, AnalyticsConfig,
    TemporalAnalyzer, GraphNeuralNetwork, AdvancedMLPipeline,
    GraphForecaster, GraphStatistics, AutoFeatureEngineering
};

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
    // v0.7.0 Advanced Analytics in Prelude
    pub use crate::analytics::{
        AdvancedAnalyticsEngine, AnalyticsConfig,
        TemporalAnalyzer, GraphNeuralNetwork, AdvancedMLPipeline,
        GraphForecaster, GraphStatistics, AutoFeatureEngineering
    };
}