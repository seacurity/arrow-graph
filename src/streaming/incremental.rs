use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::{Array, StringArray, Float64Array, RecordBatch};
use arrow::datatypes::{Schema, Field, DataType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Incremental graph update operations for streaming graph processing
/// Supports efficient add/remove of vertices and edges with minimal recomputation
#[derive(Debug, Clone)]
pub struct IncrementalGraphProcessor {
    graph: ArrowGraph,
    vertex_cache: HashMap<String, usize>, // vertex_id -> internal_index
    edge_cache: HashMap<(String, String), f64>, // (source, target) -> weight
    pending_vertex_additions: Vec<String>,
    pending_vertex_removals: HashSet<String>,
    pending_edge_additions: Vec<(String, String, f64)>,
    pending_edge_removals: HashSet<(String, String)>,
    batch_size: usize, // Number of operations to batch before applying
}

/// Types of incremental updates
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateOperation {
    AddVertex(String),
    RemoveVertex(String),
    AddEdge(String, String, f64), // source, target, weight
    RemoveEdge(String, String),   // source, target
}

/// Result of applying incremental updates
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub vertices_added: usize,
    pub vertices_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub affected_components: Vec<String>, // Components that changed
    pub recomputation_needed: bool, // Whether algorithms need full recomputation
}

impl IncrementalGraphProcessor {
    /// Create a new incremental processor from an existing graph
    pub fn new(graph: ArrowGraph) -> Result<Self> {
        let mut vertex_cache = HashMap::new();
        let mut edge_cache = HashMap::new();

        // Build initial caches
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for (idx, node_id) in node_ids.iter().enumerate() {
                if let Some(id) = node_id {
                    vertex_cache.insert(id.to_string(), idx);
                }
            }
        }

        let edges_batch = &graph.edges;
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            let weights = edges_batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;

            for i in 0..source_ids.len() {
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                let weight = weights.value(i);
                edge_cache.insert((source.to_string(), target.to_string()), weight);
            }
        }

        Ok(Self {
            graph,
            vertex_cache,
            edge_cache,
            pending_vertex_additions: Vec::new(),
            pending_vertex_removals: HashSet::new(),
            pending_edge_additions: Vec::new(),
            pending_edge_removals: HashSet::new(),
            batch_size: 1000, // Default batch size
        })
    }

    /// Set the batch size for operations
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    /// Add a vertex to the graph (batched)
    pub fn add_vertex(&mut self, vertex_id: String) -> Result<()> {
        // Check if vertex already exists
        if self.vertex_cache.contains_key(&vertex_id) || 
           self.pending_vertex_additions.contains(&vertex_id) {
            return Ok(()); // Already exists, no-op
        }

        // Remove from pending removals if present
        self.pending_vertex_removals.remove(&vertex_id);
        
        // Add to pending additions
        self.pending_vertex_additions.push(vertex_id);

        // Auto-flush if batch size reached
        if self.pending_vertex_additions.len() >= self.batch_size {
            self.flush_vertex_operations()?;
        }

        Ok(())
    }

    /// Remove a vertex from the graph (batched)
    pub fn remove_vertex(&mut self, vertex_id: String) -> Result<()> {
        // Check if vertex exists
        if !self.vertex_cache.contains_key(&vertex_id) && 
           !self.pending_vertex_additions.contains(&vertex_id) {
            return Ok(()); // Doesn't exist, no-op
        }

        // Remove from pending additions if present
        self.pending_vertex_additions.retain(|v| v != &vertex_id);
        
        // Add to pending removals
        self.pending_vertex_removals.insert(vertex_id.clone());

        // Also remove all edges involving this vertex
        let edges_to_remove: Vec<(String, String)> = self.edge_cache
            .keys()
            .filter(|(source, target)| source == &vertex_id || target == &vertex_id)
            .cloned()
            .collect();

        for (source, target) in edges_to_remove {
            self.pending_edge_removals.insert((source, target));
        }

        // Auto-flush if batch size reached
        if self.pending_vertex_removals.len() >= self.batch_size {
            self.flush_all_operations()?;
        }

        Ok(())
    }

    /// Add an edge to the graph (batched)
    pub fn add_edge(&mut self, source: String, target: String, weight: f64) -> Result<()> {
        let edge_key = (source.clone(), target.clone());

        // Check if edge already exists
        if self.edge_cache.contains_key(&edge_key) {
            // Update weight if different
            if let Some(existing_weight) = self.edge_cache.get(&edge_key) {
                if (existing_weight - weight).abs() > f64::EPSILON {
                    self.edge_cache.insert(edge_key.clone(), weight);
                }
            }
            return Ok(());
        }

        // Remove from pending removals if present
        self.pending_edge_removals.remove(&edge_key);
        
        // Add vertices if they don't exist
        self.add_vertex(source.clone())?;
        self.add_vertex(target.clone())?;

        // Add to pending additions
        self.pending_edge_additions.push((source, target, weight));

        // Auto-flush if batch size reached
        if self.pending_edge_additions.len() >= self.batch_size {
            self.flush_edge_operations()?;
        }

        Ok(())
    }

    /// Remove an edge from the graph (batched)
    pub fn remove_edge(&mut self, source: String, target: String) -> Result<()> {
        let edge_key = (source.clone(), target.clone());

        // Check if edge exists
        if !self.edge_cache.contains_key(&edge_key) {
            return Ok(()); // Doesn't exist, no-op
        }

        // Remove from pending additions if present
        self.pending_edge_additions.retain(|(s, t, _)| (s, t) != (&source, &target));
        
        // Add to pending removals
        self.pending_edge_removals.insert(edge_key);

        // Auto-flush if batch size reached
        if self.pending_edge_removals.len() >= self.batch_size {
            self.flush_edge_operations()?;
        }

        Ok(())
    }

    /// Apply a batch of update operations
    pub fn apply_updates(&mut self, operations: Vec<UpdateOperation>) -> Result<UpdateResult> {
        for operation in operations {
            match operation {
                UpdateOperation::AddVertex(vertex_id) => self.add_vertex(vertex_id)?,
                UpdateOperation::RemoveVertex(vertex_id) => self.remove_vertex(vertex_id)?,
                UpdateOperation::AddEdge(source, target, weight) => self.add_edge(source, target, weight)?,
                UpdateOperation::RemoveEdge(source, target) => self.remove_edge(source, target)?,
            }
        }

        // Flush all pending operations
        self.flush_all_operations()
    }

    /// Flush all pending operations and rebuild the graph
    pub fn flush_all_operations(&mut self) -> Result<UpdateResult> {
        let mut result = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 0,
            edges_removed: 0,
            affected_components: Vec::new(),
            recomputation_needed: false,
        };

        // Flush vertices first
        let vertex_result = self.flush_vertex_operations()?;
        result.vertices_added += vertex_result.vertices_added;
        result.vertices_removed += vertex_result.vertices_removed;

        // Then flush edges
        let edge_result = self.flush_edge_operations()?;
        result.edges_added += edge_result.edges_added;
        result.edges_removed += edge_result.edges_removed;

        // Determine if recomputation is needed
        result.recomputation_needed = result.vertices_added > 0 || 
                                     result.vertices_removed > 0 || 
                                     result.edges_added > 0 || 
                                     result.edges_removed > 0;

        Ok(result)
    }

    /// Flush pending vertex operations
    fn flush_vertex_operations(&mut self) -> Result<UpdateResult> {
        let mut result = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 0,
            edges_removed: 0,
            affected_components: Vec::new(),
            recomputation_needed: false,
        };

        // Process additions
        for vertex_id in &self.pending_vertex_additions {
            if !self.vertex_cache.contains_key(vertex_id) {
                let new_index = self.vertex_cache.len();
                self.vertex_cache.insert(vertex_id.clone(), new_index);
                result.vertices_added += 1;
            }
        }

        // Process removals
        for vertex_id in &self.pending_vertex_removals {
            if self.vertex_cache.remove(vertex_id).is_some() {
                result.vertices_removed += 1;
            }
        }

        // Rebuild graph if there were changes
        if result.vertices_added > 0 || result.vertices_removed > 0 {
            self.rebuild_graph()?;
        }

        // Clear pending operations
        self.pending_vertex_additions.clear();
        self.pending_vertex_removals.clear();

        Ok(result)
    }

    /// Flush pending edge operations
    fn flush_edge_operations(&mut self) -> Result<UpdateResult> {
        let mut result = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 0,
            edges_removed: 0,
            affected_components: Vec::new(),
            recomputation_needed: false,
        };

        // Process additions
        for (source, target, weight) in &self.pending_edge_additions {
            let edge_key = (source.clone(), target.clone());
            if !self.edge_cache.contains_key(&edge_key) {
                self.edge_cache.insert(edge_key, *weight);
                result.edges_added += 1;
            }
        }

        // Process removals
        for edge_key in &self.pending_edge_removals {
            if self.edge_cache.remove(edge_key).is_some() {
                result.edges_removed += 1;
            }
        }

        // Rebuild graph if there were changes
        if result.edges_added > 0 || result.edges_removed > 0 {
            self.rebuild_graph()?;
        }

        // Clear pending operations
        self.pending_edge_additions.clear();
        self.pending_edge_removals.clear();

        Ok(result)
    }

    /// Rebuild the ArrowGraph from current caches
    fn rebuild_graph(&mut self) -> Result<()> {
        // Build nodes RecordBatch
        let node_ids: Vec<String> = self.vertex_cache.keys().cloned().collect();
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        
        let nodes_batch = if node_ids.is_empty() {
            None
        } else {
            let node_id_array = StringArray::from(node_ids);
            Some(RecordBatch::try_new(
                nodes_schema.clone(),
                vec![Arc::new(node_id_array)],
            )?)
        };

        // Build edges RecordBatch
        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));

        let edges_batch = if self.edge_cache.is_empty() {
            None
        } else {
            let mut sources = Vec::new();
            let mut targets = Vec::new();
            let mut weights = Vec::new();

            for ((source, target), weight) in &self.edge_cache {
                sources.push(source.clone());
                targets.push(target.clone());
                weights.push(*weight);
            }

            let source_array = StringArray::from(sources);
            let target_array = StringArray::from(targets);
            let weight_array = Float64Array::from(weights);

            Some(RecordBatch::try_new(
                edges_schema.clone(),
                vec![Arc::new(source_array), Arc::new(target_array), Arc::new(weight_array)],
            )?)
        };

        // Create new graph
        match (nodes_batch, edges_batch) {
            (Some(nodes), Some(edges)) => {
                self.graph = ArrowGraph::new(nodes, edges)?;
            }
            (Some(nodes), None) => {
                // Create empty edges batch
                let edges_schema = Arc::new(Schema::new(vec![
                    Field::new("source", DataType::Utf8, false),
                    Field::new("target", DataType::Utf8, false),
                    Field::new("weight", DataType::Float64, false),
                ]));
                let empty_edges = RecordBatch::new_empty(edges_schema);
                self.graph = ArrowGraph::new(nodes, empty_edges)?;
            }
            (None, Some(edges)) => {
                self.graph = ArrowGraph::from_edges(edges)?;
            }
            (None, None) => {
                self.graph = ArrowGraph::empty()?;
            }
        }

        Ok(())
    }

    /// Get the current graph
    pub fn graph(&self) -> &ArrowGraph {
        &self.graph
    }

    /// Get the number of pending operations
    pub fn pending_operations_count(&self) -> usize {
        self.pending_vertex_additions.len() + 
        self.pending_vertex_removals.len() + 
        self.pending_edge_additions.len() + 
        self.pending_edge_removals.len()
    }

    /// Check if there are pending operations
    pub fn has_pending_operations(&self) -> bool {
        self.pending_operations_count() > 0
    }

    /// Get statistics about the current state
    pub fn statistics(&self) -> IncrementalStats {
        IncrementalStats {
            vertex_count: self.vertex_cache.len(),
            edge_count: self.edge_cache.len(),
            pending_vertex_additions: self.pending_vertex_additions.len(),
            pending_vertex_removals: self.pending_vertex_removals.len(),
            pending_edge_additions: self.pending_edge_additions.len(),
            pending_edge_removals: self.pending_edge_removals.len(),
            batch_size: self.batch_size,
        }
    }
}

/// Statistics about the incremental processor state
#[derive(Debug, Clone)]
pub struct IncrementalStats {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub pending_vertex_additions: usize,
    pub pending_vertex_removals: usize,
    pub pending_edge_additions: usize,
    pub pending_edge_removals: usize,
    pub batch_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    fn create_test_graph() -> Result<ArrowGraph> {
        // Create nodes
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let node_ids = StringArray::from(vec!["A", "B", "C"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        // Create edges
        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["A", "B"]);
        let targets = StringArray::from(vec!["B", "C"]);
        let weights = Float64Array::from(vec![1.0, 2.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_incremental_processor_creation() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.pending_vertex_additions, 0);
        assert_eq!(stats.pending_edge_additions, 0);
    }

    #[test]
    fn test_add_vertex() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        processor.add_vertex("D".to_string()).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 4);
        assert!(processor.vertex_cache.contains_key("D"));
    }

    #[test]
    fn test_remove_vertex() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        processor.remove_vertex("C".to_string()).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 2);
        assert!(!processor.vertex_cache.contains_key("C"));
        // Edge B->C should also be removed
        assert!(!processor.edge_cache.contains_key(&("B".to_string(), "C".to_string())));
    }

    #[test]
    fn test_add_edge() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        processor.add_edge("A".to_string(), "C".to_string(), 3.0).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.edge_count, 3);
        assert_eq!(
            processor.edge_cache.get(&("A".to_string(), "C".to_string())),
            Some(&3.0)
        );
    }

    #[test]
    fn test_remove_edge() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        processor.remove_edge("A".to_string(), "B".to_string()).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.edge_count, 1);
        assert!(!processor.edge_cache.contains_key(&("A".to_string(), "B".to_string())));
    }

    #[test]
    fn test_batch_operations() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(10); // Large batch size to test batching
        
        // Add multiple operations without flushing
        processor.add_vertex("D".to_string()).unwrap();
        processor.add_vertex("E".to_string()).unwrap();
        processor.add_edge("D".to_string(), "E".to_string(), 4.0).unwrap();
        
        // Operations should be pending
        assert!(processor.has_pending_operations());
        assert_eq!(processor.pending_operations_count(), 3);
        
        // Flush all operations
        let result = processor.flush_all_operations().unwrap();
        
        assert_eq!(result.vertices_added, 2);
        assert_eq!(result.edges_added, 1);
        assert!(!processor.has_pending_operations());
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 5);
        assert_eq!(stats.edge_count, 3);
    }

    #[test]
    fn test_apply_updates() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let operations = vec![
            UpdateOperation::AddVertex("D".to_string()),
            UpdateOperation::AddVertex("E".to_string()),
            UpdateOperation::AddEdge("D".to_string(), "E".to_string(), 5.0),
            UpdateOperation::RemoveEdge("A".to_string(), "B".to_string()),
            UpdateOperation::RemoveVertex("C".to_string()),
        ];
        
        let result = processor.apply_updates(operations).unwrap();
        
        assert_eq!(result.vertices_added, 2);
        assert_eq!(result.vertices_removed, 1);
        assert_eq!(result.edges_added, 1);
        assert_eq!(result.edges_removed, 2); // A->B and B->C (removed with vertex C)
        assert!(result.recomputation_needed);
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 4); // A, B, D, E (C removed)
        assert_eq!(stats.edge_count, 1); // Only D->E remains
    }

    #[test]
    fn test_duplicate_operations() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        // Add same vertex multiple times
        processor.add_vertex("D".to_string()).unwrap();
        processor.add_vertex("D".to_string()).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 4); // Should only add once
        
        // Remove non-existent vertex
        processor.remove_vertex("Z".to_string()).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 4); // Should remain unchanged
    }

    #[test]
    fn test_edge_with_new_vertices() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        // Add edge with new vertices
        processor.add_edge("X".to_string(), "Y".to_string(), 10.0).unwrap();
        
        let stats = processor.statistics();
        assert_eq!(stats.vertex_count, 5); // Original 3 + X + Y
        assert_eq!(stats.edge_count, 3); // Original 2 + X->Y
        
        assert!(processor.vertex_cache.contains_key("X"));
        assert!(processor.vertex_cache.contains_key("Y"));
        assert_eq!(
            processor.edge_cache.get(&("X".to_string(), "Y".to_string())),
            Some(&10.0)
        );
    }
}