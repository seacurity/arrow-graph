use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::collections::HashMap;
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

/// Streaming graph update operations for incremental processing
#[derive(Debug, Clone)]
pub enum StreamUpdate {
    /// Add a new node
    AddNode { node_id: String },
    /// Remove an existing node
    RemoveNode { node_id: String },
    /// Add a new edge
    AddEdge { source: String, target: String, weight: Option<f64> },
    /// Remove an existing edge
    RemoveEdge { source: String, target: String },
    /// Update edge weight
    UpdateEdgeWeight { source: String, target: String, weight: f64 },
    /// Batch of multiple operations
    Batch { operations: Vec<StreamUpdate> },
}

/// Streaming graph processor for handling incremental updates
pub struct StreamingGraphProcessor {
    graph: ArrowGraph,
    update_count: u64,
    change_log: Vec<(u64, StreamUpdate)>,
    enable_change_log: bool,
}

impl StreamingGraphProcessor {
    /// Create a new streaming processor with an initial graph
    pub fn new(initial_graph: ArrowGraph) -> Self {
        Self {
            graph: initial_graph,
            update_count: 0,
            change_log: Vec::new(),
            enable_change_log: false,
        }
    }

    /// Create an empty streaming processor
    pub fn empty() -> Result<Self> {
        let empty_graph = ArrowGraph::empty()?;
        Ok(Self::new(empty_graph))
    }

    /// Enable or disable change logging
    pub fn set_change_log_enabled(&mut self, enabled: bool) {
        self.enable_change_log = enabled;
        if !enabled {
            self.change_log.clear();
        }
    }

    /// Get current graph state
    pub fn graph(&self) -> &ArrowGraph {
        &self.graph
    }

    /// Get current update count
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Apply a streaming update to the graph
    pub fn apply_update(&mut self, update: StreamUpdate) -> Result<UpdateResult> {
        let start_time = std::time::Instant::now();
        let _initial_node_count = self.graph.node_count();
        let _initial_edge_count = self.graph.edge_count();

        // Log the update if enabled
        if self.enable_change_log {
            self.change_log.push((self.update_count, update.clone()));
        }

        let result = match update {
            StreamUpdate::AddNode { node_id } => {
                self.graph.add_node(node_id.clone())?;
                UpdateResult {
                    operation: "add_node".to_string(),
                    affected_nodes: vec![node_id],
                    affected_edges: Vec::new(),
                    nodes_added: 1,
                    nodes_removed: 0,
                    edges_added: 0,
                    edges_removed: 0,
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            }
            StreamUpdate::RemoveNode { node_id } => {
                // Count edges that will be removed
                let edges_to_remove = self.count_node_edges(&node_id);
                self.graph.remove_node(&node_id)?;
                UpdateResult {
                    operation: "remove_node".to_string(),
                    affected_nodes: vec![node_id],
                    affected_edges: Vec::new(),
                    nodes_added: 0,
                    nodes_removed: 1,
                    edges_added: 0,
                    edges_removed: edges_to_remove,
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            }
            StreamUpdate::AddEdge { source, target, weight } => {
                self.graph.add_edge(source.clone(), target.clone(), weight)?;
                UpdateResult {
                    operation: "add_edge".to_string(),
                    affected_nodes: vec![source.clone(), target.clone()],
                    affected_edges: vec![format!("{}→{}", source, target)],
                    nodes_added: 0,
                    nodes_removed: 0,
                    edges_added: 1,
                    edges_removed: 0,
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            }
            StreamUpdate::RemoveEdge { source, target } => {
                self.graph.remove_edge(&source, &target)?;
                UpdateResult {
                    operation: "remove_edge".to_string(),
                    affected_nodes: vec![source.clone(), target.clone()],
                    affected_edges: vec![format!("{}→{}", source, target)],
                    nodes_added: 0,
                    nodes_removed: 0,
                    edges_added: 0,
                    edges_removed: 1,
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            }
            StreamUpdate::UpdateEdgeWeight { source, target, weight } => {
                // Remove and re-add with new weight
                self.graph.remove_edge(&source, &target)?;
                self.graph.add_edge(source.clone(), target.clone(), Some(weight))?;
                UpdateResult {
                    operation: "update_edge_weight".to_string(),
                    affected_nodes: vec![source.clone(), target.clone()],
                    affected_edges: vec![format!("{}→{}", source, target)],
                    nodes_added: 0,
                    nodes_removed: 0,
                    edges_added: 0,
                    edges_removed: 0,
                    processing_time_ms: start_time.elapsed().as_millis() as f64,
                }
            }
            StreamUpdate::Batch { operations } => {
                let mut batch_result = UpdateResult {
                    operation: "batch".to_string(),
                    affected_nodes: Vec::new(),
                    affected_edges: Vec::new(),
                    nodes_added: 0,
                    nodes_removed: 0,
                    edges_added: 0,
                    edges_removed: 0,
                    processing_time_ms: 0.0,
                };

                for op in operations {
                    let op_result = self.apply_update(op)?;
                    batch_result.merge(op_result);
                }

                batch_result.processing_time_ms = start_time.elapsed().as_millis() as f64;
                batch_result
            }
        };

        self.update_count += 1;
        Ok(result)
    }

    /// Apply multiple updates in sequence
    pub fn apply_updates(&mut self, updates: Vec<StreamUpdate>) -> Result<Vec<UpdateResult>> {
        let mut results = Vec::new();
        for update in updates {
            results.push(self.apply_update(update)?);
        }
        Ok(results)
    }

    /// Get the change log since a specific update count
    pub fn get_change_log_since(&self, since_update: u64) -> Vec<(u64, StreamUpdate)> {
        self.change_log
            .iter()
            .filter(|(update_count, _)| *update_count >= since_update)
            .cloned()
            .collect()
    }

    /// Get statistics about the streaming processor
    pub fn get_statistics(&self) -> StreamingStatistics {
        StreamingStatistics {
            total_updates: self.update_count,
            current_node_count: self.graph.node_count(),
            current_edge_count: self.graph.edge_count(),
            change_log_size: self.change_log.len(),
            change_log_enabled: self.enable_change_log,
        }
    }

    /// Create a snapshot of the current graph state
    pub fn create_snapshot(&self) -> Result<GraphSnapshot> {
        Ok(GraphSnapshot {
            update_count: self.update_count,
            graph: self.graph.clone(),
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Restore from a snapshot
    pub fn restore_from_snapshot(&mut self, snapshot: GraphSnapshot) {
        self.graph = snapshot.graph;
        self.update_count = snapshot.update_count;
        // Clear change log when restoring
        self.change_log.clear();
    }

    /// Compact the change log (remove entries older than specified update count)
    pub fn compact_change_log(&mut self, keep_since: u64) {
        if self.enable_change_log {
            self.change_log.retain(|(update_count, _)| *update_count >= keep_since);
        }
    }

    /// Helper to count edges for a node (for removal statistics)
    fn count_node_edges(&self, node_id: &str) -> usize {
        let mut count = 0;
        
        // Count outgoing edges
        if let Some(neighbors) = self.graph.neighbors(node_id) {
            count += neighbors.len();
        }
        
        // Count incoming edges (this is approximate for undirected graphs)
        for other_node in self.graph.node_ids() {
            if other_node != node_id {
                if let Some(neighbors) = self.graph.neighbors(other_node) {
                    count += neighbors.iter().filter(|&n| n == node_id).count();
                }
            }
        }
        
        count
    }
}

/// Result of applying a streaming update
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub operation: String,
    pub affected_nodes: Vec<String>,
    pub affected_edges: Vec<String>,
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub processing_time_ms: f64,
}

impl UpdateResult {
    /// Merge another update result into this one (for batch operations)
    pub fn merge(&mut self, other: UpdateResult) {
        self.affected_nodes.extend(other.affected_nodes);
        self.affected_edges.extend(other.affected_edges);
        self.nodes_added += other.nodes_added;
        self.nodes_removed += other.nodes_removed;
        self.edges_added += other.edges_added;
        self.edges_removed += other.edges_removed;
        // Don't add processing time for individual operations in batch
    }

    /// Convert to Arrow RecordBatch for analysis
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("operation", DataType::Utf8, false),
            Field::new("nodes_added", DataType::UInt64, false),
            Field::new("nodes_removed", DataType::UInt64, false),
            Field::new("edges_added", DataType::UInt64, false),
            Field::new("edges_removed", DataType::UInt64, false),
            Field::new("processing_time_ms", DataType::Float64, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![self.operation.clone()])),
                Arc::new(UInt64Array::from(vec![self.nodes_added as u64])),
                Arc::new(UInt64Array::from(vec![self.nodes_removed as u64])),
                Arc::new(UInt64Array::from(vec![self.edges_added as u64])),
                Arc::new(UInt64Array::from(vec![self.edges_removed as u64])),
                Arc::new(Float64Array::from(vec![self.processing_time_ms])),
            ],
        ).map_err(GraphError::from)
    }
}

/// Statistics about the streaming processor
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    pub total_updates: u64,
    pub current_node_count: usize,
    pub current_edge_count: usize,
    pub change_log_size: usize,
    pub change_log_enabled: bool,
}

/// Snapshot of graph state at a specific point in time
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub update_count: u64,
    pub graph: ArrowGraph,
    pub timestamp: std::time::SystemTime,
}

/// Incremental algorithm processor for streaming updates
pub struct IncrementalAlgorithmProcessor {
    cached_results: HashMap<String, (u64, RecordBatch)>,
    invalidation_threshold: u64,
}

impl IncrementalAlgorithmProcessor {
    /// Create a new incremental algorithm processor
    pub fn new() -> Self {
        Self {
            cached_results: HashMap::new(),
            invalidation_threshold: 10, // Invalidate cache after 10 updates
        }
    }

    /// Set the cache invalidation threshold
    pub fn set_invalidation_threshold(&mut self, threshold: u64) {
        self.invalidation_threshold = threshold;
    }

    /// Check if cached result is still valid
    pub fn is_cache_valid(&self, algorithm_name: &str, current_update_count: u64) -> bool {
        if let Some((cached_update_count, _)) = self.cached_results.get(algorithm_name) {
            current_update_count - cached_update_count <= self.invalidation_threshold
        } else {
            false
        }
    }

    /// Cache algorithm result
    pub fn cache_result(&mut self, algorithm_name: String, update_count: u64, result: RecordBatch) {
        self.cached_results.insert(algorithm_name, (update_count, result));
    }

    /// Get cached result if valid
    pub fn get_cached_result(&self, algorithm_name: &str, current_update_count: u64) -> Option<RecordBatch> {
        if self.is_cache_valid(algorithm_name, current_update_count) {
            self.cached_results.get(algorithm_name).map(|(_, result)| result.clone())
        } else {
            None
        }
    }

    /// Invalidate cache for specific algorithm
    pub fn invalidate_cache(&mut self, algorithm_name: &str) {
        self.cached_results.remove(algorithm_name);
    }

    /// Clear all cached results
    pub fn clear_cache(&mut self) {
        self.cached_results.clear();
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        CacheStatistics {
            cached_algorithms: self.cached_results.len(),
            invalidation_threshold: self.invalidation_threshold,
            total_cache_size: self.cached_results.values()
                .map(|(_, batch)| batch.get_array_memory_size())
                .sum(),
        }
    }
}

impl Default for IncrementalAlgorithmProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about algorithm caching
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub cached_algorithms: usize,
    pub invalidation_threshold: u64,
    pub total_cache_size: usize,
}

/// Combined streaming processor with incremental algorithms
pub struct StreamingGraphSystem {
    graph_processor: StreamingGraphProcessor,
    algorithm_processor: IncrementalAlgorithmProcessor,
}

impl StreamingGraphSystem {
    /// Create a new streaming graph system
    pub fn new(initial_graph: ArrowGraph) -> Self {
        Self {
            graph_processor: StreamingGraphProcessor::new(initial_graph),
            algorithm_processor: IncrementalAlgorithmProcessor::new(),
        }
    }

    /// Create an empty streaming graph system
    pub fn empty() -> Result<Self> {
        Ok(Self {
            graph_processor: StreamingGraphProcessor::empty()?,
            algorithm_processor: IncrementalAlgorithmProcessor::new(),
        })
    }

    /// Get the graph processor
    pub fn graph_processor(&self) -> &StreamingGraphProcessor {
        &self.graph_processor
    }

    /// Get mutable access to the graph processor
    pub fn graph_processor_mut(&mut self) -> &mut StreamingGraphProcessor {
        &mut self.graph_processor
    }

    /// Get the algorithm processor
    pub fn algorithm_processor(&self) -> &IncrementalAlgorithmProcessor {
        &self.algorithm_processor
    }

    /// Get mutable access to the algorithm processor
    pub fn algorithm_processor_mut(&mut self) -> &mut IncrementalAlgorithmProcessor {
        &mut self.algorithm_processor
    }

    /// Apply update and invalidate relevant algorithm caches
    pub fn apply_update_with_cache_invalidation(&mut self, update: StreamUpdate) -> Result<UpdateResult> {
        let result = self.graph_processor.apply_update(update)?;
        
        // Invalidate caches for algorithms that might be affected
        match result.operation.as_str() {
            "add_node" | "remove_node" => {
                // Node changes affect most algorithms
                self.algorithm_processor.clear_cache();
            }
            "add_edge" | "remove_edge" | "update_edge_weight" => {
                // Edge changes affect connectivity and centrality algorithms
                self.algorithm_processor.invalidate_cache("pagerank");
                self.algorithm_processor.invalidate_cache("betweenness_centrality");
                self.algorithm_processor.invalidate_cache("closeness_centrality");
                self.algorithm_processor.invalidate_cache("eigenvector_centrality");
                self.algorithm_processor.invalidate_cache("shortest_path");
            }
            _ => {}
        }
        
        Ok(result)
    }
}