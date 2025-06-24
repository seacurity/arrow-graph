/// Advanced Analytics Module for arrow-graph v0.7.0
/// 
/// This module provides sophisticated analytics capabilities including:
/// - Temporal query processing and time-series analysis
/// - Graph Neural Networks (GNN) framework  
/// - Advanced machine learning pipelines
/// - Predictive analytics and forecasting
/// - Statistical analysis tools
/// 
/// The analytics framework is built on Arrow's columnar format for high-performance
/// processing of large-scale temporal and multi-dimensional graph data.

pub mod temporal;
pub mod gnn;
pub mod ml_pipeline;
pub mod forecasting;
pub mod statistics;
pub mod feature_engineering;

// Re-export core analytics components
pub use temporal::{
    TemporalAnalyzer, TimeWindowQuery, TemporalPattern, 
    DynamicMetrics, TemporalAggregation
};

pub use gnn::{
    GraphNeuralNetwork, GCNLayer, GATLayer, GraphSAGELayer,
    NodeClassifier, EdgePredictor, GraphClassifier, GNNTrainer
};

pub use ml_pipeline::{
    AdvancedMLPipeline, AutoMLEngine, FeatureSelector,
    ModelEnsemble, HyperparameterOptimizer, CrossValidator
};

pub use forecasting::{
    GraphForecaster, TimeSeriesPredictor, AnomalyDetector,
    TrendAnalyzer, SeasonalityDetector, CausalInference
};

pub use statistics::{
    GraphStatistics, DistributionAnalyzer, HypothesisTester,
    CorrelationAnalyzer, DimensionalityReducer, ClusterAnalyzer
};

pub use feature_engineering::{
    AutoFeatureEngineering, TemporalFeatures, StructuralFeatures,
    CompositeFeatures, FeatureTransformer, FeatureValidator
};

/// Analytics configuration and global settings
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    pub enable_gpu_acceleration: bool,
    pub max_memory_usage: usize,  // in bytes
    pub parallel_workers: usize,
    pub cache_size: usize,
    pub precision: AnalyticsPrecision,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone)]
pub enum AnalyticsPrecision {
    Float32,
    Float64,
    Mixed,  // Use appropriate precision per algorithm
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    Debug,      // No optimizations, full debugging
    Balanced,   // Good performance with some debugging
    Performance, // Maximum performance, minimal debugging  
    Extreme,    // Unsafe optimizations for maximum speed
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_gpu_acceleration: false, // Default to CPU for compatibility
            max_memory_usage: 8 * 1024 * 1024 * 1024, // 8GB default
            parallel_workers: num_cpus::get(),
            cache_size: 512 * 1024 * 1024, // 512MB cache
            precision: AnalyticsPrecision::Float64,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

/// Central analytics engine that coordinates all advanced analytics capabilities
pub struct AdvancedAnalyticsEngine {
    config: AnalyticsConfig,
    temporal_analyzer: TemporalAnalyzer,
    gnn_framework: GraphNeuralNetwork,
    ml_pipeline: AdvancedMLPipeline,
    forecaster: GraphForecaster,
    statistics: GraphStatistics,
    feature_engineering: AutoFeatureEngineering,
}

impl AdvancedAnalyticsEngine {
    /// Create a new analytics engine with default configuration
    pub fn new() -> Self {
        Self::with_config(AnalyticsConfig::default())
    }
    
    /// Create a new analytics engine with custom configuration
    pub fn with_config(config: AnalyticsConfig) -> Self {
        Self {
            temporal_analyzer: TemporalAnalyzer::new(&config),
            gnn_framework: GraphNeuralNetwork::new(&config),
            ml_pipeline: AdvancedMLPipeline::new(&config),
            forecaster: GraphForecaster::new(&config),
            statistics: GraphStatistics::new(&config),
            feature_engineering: AutoFeatureEngineering::new(&config),
            config,
        }
    }
    
    /// Get a reference to the temporal analyzer
    pub fn temporal(&self) -> &TemporalAnalyzer {
        &self.temporal_analyzer
    }
    
    /// Get a reference to the GNN framework
    pub fn gnn(&self) -> &GraphNeuralNetwork {
        &self.gnn_framework
    }
    
    /// Get a reference to the ML pipeline
    pub fn ml_pipeline(&self) -> &AdvancedMLPipeline {
        &self.ml_pipeline
    }
    
    /// Get a reference to the forecaster
    pub fn forecaster(&self) -> &GraphForecaster {
        &self.forecaster
    }
    
    /// Get a reference to the statistics module
    pub fn statistics(&self) -> &GraphStatistics {
        &self.statistics
    }
    
    /// Get a reference to the feature engineering module
    pub fn feature_engineering(&self) -> &AutoFeatureEngineering {
        &self.feature_engineering
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &AnalyticsConfig {
        &self.config
    }
}

impl Default for AdvancedAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analytics_engine_creation() {
        let engine = AdvancedAnalyticsEngine::new();
        assert_eq!(engine.config().parallel_workers, num_cpus::get());
        assert!(matches!(engine.config().precision, AnalyticsPrecision::Float64));
    }
    
    #[test]
    fn test_custom_config() {
        let config = AnalyticsConfig {
            enable_gpu_acceleration: true,
            max_memory_usage: 16 * 1024 * 1024 * 1024, // 16GB
            parallel_workers: 16,
            cache_size: 1024 * 1024 * 1024, // 1GB
            precision: AnalyticsPrecision::Float32,
            optimization_level: OptimizationLevel::Performance,
        };
        
        let engine = AdvancedAnalyticsEngine::with_config(config.clone());
        assert!(engine.config().enable_gpu_acceleration);
        assert_eq!(engine.config().parallel_workers, 16);
        assert!(matches!(engine.config().precision, AnalyticsPrecision::Float32));
    }
}