use crate::error::Result;
use crate::streaming::incremental::{IncrementalGraphProcessor, UpdateResult};
use arrow::array::Array;
use std::collections::HashMap;

/// Streaming algorithms that can update incrementally as the graph changes
/// These algorithms maintain state and update efficiently rather than recomputing from scratch
pub trait StreamingAlgorithm<T> {
    /// Initialize the algorithm with the current graph state
    fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()>;
    
    /// Update the algorithm state based on graph changes
    fn update(&mut self, processor: &IncrementalGraphProcessor, changes: &UpdateResult) -> Result<()>;
    
    /// Get the current result/state of the algorithm
    fn get_result(&self) -> &T;
    
    /// Force a full recomputation (fallback when incremental update is not sufficient)
    fn recompute(&mut self, processor: &IncrementalGraphProcessor) -> Result<()>;
    
    /// Check if the algorithm needs full recomputation
    fn needs_recomputation(&self, changes: &UpdateResult) -> bool;
}

/// Streaming PageRank algorithm that updates incrementally
#[derive(Debug, Clone)]
pub struct StreamingPageRank {
    scores: HashMap<String, f64>,
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
    iteration_count: usize,
    converged: bool,
}

impl StreamingPageRank {
    pub fn new(damping_factor: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            scores: HashMap::new(),
            damping_factor,
            max_iterations,
            tolerance,
            iteration_count: 0,
            converged: false,
        }
    }

    /// Default parameters for streaming PageRank
    pub fn default() -> Self {
        Self::new(0.85, 50, 1e-6)
    }

    /// Perform one iteration of PageRank updates
    fn iterate(&mut self, adjacency: &HashMap<String, Vec<(String, f64)>>, nodes: &[String]) -> Result<bool> {
        let node_count = nodes.len() as f64;
        let base_score = (1.0 - self.damping_factor) / node_count;
        
        let mut new_scores = HashMap::new();
        
        // Initialize all nodes with base score
        for node in nodes {
            new_scores.insert(node.clone(), base_score);
        }
        
        // Add contributions from incoming links
        for (source, targets) in adjacency {
            let source_score = self.scores.get(source).copied().unwrap_or(1.0 / node_count);
            let out_degree = targets.len() as f64;
            
            if out_degree > 0.0 {
                let contribution_per_link = self.damping_factor * source_score / out_degree;
                
                for (target, _weight) in targets {
                    *new_scores.entry(target.clone()).or_insert(base_score) += contribution_per_link;
                }
            }
        }
        
        // Check for convergence
        let mut max_change: f64 = 0.0;
        for (node, new_score) in &new_scores {
            let old_score = self.scores.get(node).copied().unwrap_or(1.0 / node_count);
            let change = (new_score - old_score).abs();
            max_change = max_change.max(change);
        }
        
        self.scores = new_scores;
        self.iteration_count += 1;
        
        let converged = max_change < self.tolerance;
        self.converged = converged;
        
        Ok(converged)
    }

    /// Get top-k nodes by PageRank score
    pub fn top_nodes(&self, k: usize) -> Vec<(String, f64)> {
        let mut node_scores: Vec<(String, f64)> = self.scores.iter()
            .map(|(node, score)| (node.clone(), *score))
            .collect();
            
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        node_scores.truncate(k);
        node_scores
    }

    /// Get score for a specific node
    pub fn node_score(&self, node_id: &str) -> Option<f64> {
        self.scores.get(node_id).copied()
    }
}

impl StreamingAlgorithm<HashMap<String, f64>> for StreamingPageRank {
    fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        // Build adjacency list from current graph
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;
        
        // Extract nodes
        let mut nodes = Vec::new();
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
                
            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }
        
        // Build adjacency map
        let mut adjacency: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            let weights = edges_batch.column(2)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;
                
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                let weight = weights.value(i);
                
                adjacency.entry(source).or_insert_with(Vec::new).push((target, weight));
            }
        }
        
        // Initialize scores
        let node_count = nodes.len() as f64;
        if node_count > 0.0 {
            let initial_score = 1.0 / node_count;
            for node in &nodes {
                self.scores.insert(node.clone(), initial_score);
            }
            
            // Run initial PageRank computation
            for _ in 0..self.max_iterations {
                if self.iterate(&adjacency, &nodes)? {
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self, processor: &IncrementalGraphProcessor, changes: &UpdateResult) -> Result<()> {
        // For significant changes, we recompute
        if self.needs_recomputation(changes) {
            return self.recompute(processor);
        }
        
        // For minor changes, we can do incremental updates
        // This is a simplified approach - full incremental PageRank is complex
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;
        
        // Update node scores for new nodes
        if changes.vertices_added > 0 {
            let node_count = processor.graph().node_count() as f64;
            let initial_score = 1.0 / node_count;
            
            // Normalize existing scores
            for score in self.scores.values_mut() {
                *score *= (node_count - changes.vertices_added as f64) / node_count;
            }
            
            // Add new nodes
            if nodes_batch.num_rows() > 0 {
                let node_ids = nodes_batch.column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
                    
                for i in 0..node_ids.len() {
                    let node = node_ids.value(i).to_string();
                    self.scores.entry(node).or_insert(initial_score);
                }
            }
        }
        
        // Remove deleted nodes
        if changes.vertices_removed > 0 {
            // Note: This is simplified - we'd need to track which specific nodes were removed
            // For now, we just clean up any orphaned scores
            let mut valid_nodes = std::collections::HashSet::new();
            if nodes_batch.num_rows() > 0 {
                let node_ids = nodes_batch.column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
                    
                for i in 0..node_ids.len() {
                    valid_nodes.insert(node_ids.value(i).to_string());
                }
            }
            
            self.scores.retain(|node, _| valid_nodes.contains(node));
        }
        
        // For edge changes, do a few iterations to re-stabilize
        if changes.edges_added > 0 || changes.edges_removed > 0 {
            // Build current adjacency map
            let mut adjacency: HashMap<String, Vec<(String, f64)>> = HashMap::new();
            if edges_batch.num_rows() > 0 {
                let source_ids = edges_batch.column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
                let target_ids = edges_batch.column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
                let weights = edges_batch.column(2)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;
                    
                for i in 0..source_ids.len() {
                    let source = source_ids.value(i).to_string();
                    let target = target_ids.value(i).to_string();
                    let weight = weights.value(i);
                    
                    adjacency.entry(source).or_insert_with(Vec::new).push((target, weight));
                }
            }
            
            let nodes: Vec<String> = self.scores.keys().cloned().collect();
            
            // Do a few iterations to re-stabilize
            let update_iterations = std::cmp::min(10, self.max_iterations);
            for _ in 0..update_iterations {
                if self.iterate(&adjacency, &nodes)? {
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    fn get_result(&self) -> &HashMap<String, f64> {
        &self.scores
    }
    
    fn recompute(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        self.scores.clear();
        self.iteration_count = 0;
        self.converged = false;
        self.initialize(processor)
    }
    
    fn needs_recomputation(&self, changes: &UpdateResult) -> bool {
        // Recompute for large changes or if we haven't converged
        let total_changes = changes.vertices_added + changes.vertices_removed + 
                           changes.edges_added + changes.edges_removed;
        
        total_changes > 10 || !self.converged
    }
}

/// Streaming Connected Components algorithm that updates incrementally
#[derive(Debug, Clone)]
pub struct StreamingConnectedComponents {
    components: HashMap<String, String>, // node -> component_id
    component_sizes: HashMap<String, usize>, // component_id -> size
}

impl StreamingConnectedComponents {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            component_sizes: HashMap::new(),
        }
    }

    /// Get the component ID for a node
    pub fn component_of(&self, node_id: &str) -> Option<&String> {
        self.components.get(node_id)
    }

    /// Get the size of the component containing a node
    pub fn component_size(&self, node_id: &str) -> Option<usize> {
        self.components.get(node_id)
            .and_then(|comp_id| self.component_sizes.get(comp_id))
            .copied()
    }

    /// Get all components and their sizes
    pub fn all_components(&self) -> Vec<(String, usize)> {
        self.component_sizes.iter()
            .map(|(id, size)| (id.clone(), *size))
            .collect()
    }

    /// Get number of components
    pub fn component_count(&self) -> usize {
        self.component_sizes.len()
    }

    /// Union-Find helper: find root with path compression
    #[allow(dead_code)]
    fn find_root(&self, mut node: String, temp_parents: &mut HashMap<String, String>) -> String {
        let mut path = Vec::new();
        
        // Find root
        while let Some(parent) = temp_parents.get(&node).or_else(|| self.components.get(&node)) {
            if parent == &node {
                break; // Found root
            }
            path.push(node.clone());
            node = parent.clone();
        }
        
        // Path compression
        for path_node in path {
            temp_parents.insert(path_node, node.clone());
        }
        
        node
    }

    /// Union two components
    fn union_components(&mut self, node1: &str, node2: &str) {
        let comp1 = self.components.get(node1).cloned().unwrap_or_else(|| node1.to_string());
        let comp2 = self.components.get(node2).cloned().unwrap_or_else(|| node2.to_string());
        
        if comp1 == comp2 {
            return; // Already in same component
        }
        
        // Merge smaller component into larger one
        let size1 = self.component_sizes.get(&comp1).copied().unwrap_or(1);
        let size2 = self.component_sizes.get(&comp2).copied().unwrap_or(1);
        
        let (smaller, larger, new_size) = if size1 <= size2 {
            (comp1, comp2, size1 + size2)
        } else {
            (comp2, comp1, size1 + size2)
        };
        
        // Update all nodes in smaller component
        let nodes_to_update: Vec<String> = self.components.iter()
            .filter(|(_, comp)| *comp == &smaller)
            .map(|(node, _)| node.clone())
            .collect();
            
        for node in nodes_to_update {
            self.components.insert(node, larger.clone());
        }
        
        // Update component sizes
        self.component_sizes.insert(larger, new_size);
        self.component_sizes.remove(&smaller);
    }
}

impl Default for StreamingConnectedComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingAlgorithm<HashMap<String, String>> for StreamingConnectedComponents {
    fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;
        
        self.components.clear();
        self.component_sizes.clear();
        
        // Initialize each node as its own component
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
                
            for i in 0..node_ids.len() {
                let node = node_ids.value(i).to_string();
                self.components.insert(node.clone(), node.clone());
                self.component_sizes.insert(node, 1);
            }
        }
        
        // Process edges to union components
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
                
            for i in 0..source_ids.len() {
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                self.union_components(source, target);
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self, processor: &IncrementalGraphProcessor, changes: &UpdateResult) -> Result<()> {
        // For large changes, recompute
        if self.needs_recomputation(changes) {
            return self.recompute(processor);
        }
        
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;
        
        // Handle new vertices
        if changes.vertices_added > 0 {
            if nodes_batch.num_rows() > 0 {
                let node_ids = nodes_batch.column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
                    
                for i in 0..node_ids.len() {
                    let node = node_ids.value(i).to_string();
                    if !self.components.contains_key(&node) {
                        self.components.insert(node.clone(), node.clone());
                        self.component_sizes.insert(node, 1);
                    }
                }
            }
        }
        
        // Handle removed vertices (simplified)
        if changes.vertices_removed > 0 {
            // For simplicity, we'll do a full recomputation for vertex removals
            // A full implementation would track specific removed vertices
            return self.recompute(processor);
        }
        
        // Handle new edges - union components
        if changes.edges_added > 0 {
            if edges_batch.num_rows() > 0 {
                let source_ids = edges_batch.column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
                let target_ids = edges_batch.column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
                    
                for i in 0..source_ids.len() {
                    let source = source_ids.value(i);
                    let target = target_ids.value(i);
                    
                    // Ensure both nodes exist in components
                    if !self.components.contains_key(source) {
                        self.components.insert(source.to_string(), source.to_string());
                        self.component_sizes.insert(source.to_string(), 1);
                    }
                    if !self.components.contains_key(target) {
                        self.components.insert(target.to_string(), target.to_string());
                        self.component_sizes.insert(target.to_string(), 1);
                    }
                    
                    self.union_components(source, target);
                }
            }
        }
        
        // Handle edge removals (complex - may split components)
        if changes.edges_removed > 0 {
            // For simplicity, recompute when edges are removed
            // A full implementation would check if removal splits components
            return self.recompute(processor);
        }
        
        Ok(())
    }
    
    fn get_result(&self) -> &HashMap<String, String> {
        &self.components
    }
    
    fn recompute(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        self.initialize(processor)
    }
    
    fn needs_recomputation(&self, changes: &UpdateResult) -> bool {
        // Recompute for vertex removals or edge removals (may split components)
        // or for very large changes
        let total_changes = changes.vertices_added + changes.vertices_removed + 
                           changes.edges_added + changes.edges_removed;
        
        changes.vertices_removed > 0 || changes.edges_removed > 0 || total_changes > 20
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ArrowGraph;
    use arrow::array::{StringArray, Float64Array};
    use arrow::record_batch::RecordBatch;
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    fn create_test_graph() -> Result<ArrowGraph> {
        // Create nodes
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let node_ids = StringArray::from(vec!["A", "B", "C", "D"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        // Create edges: A->B, B->C, D isolated
        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["A", "B"]);
        let targets = StringArray::from(vec!["B", "C"]);
        let weights = Float64Array::from(vec![1.0, 1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_streaming_pagerank_initialization() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut pagerank = StreamingPageRank::default();
        pagerank.initialize(&processor).unwrap();
        
        let scores = pagerank.get_result();
        assert_eq!(scores.len(), 4); // A, B, C, D
        
        // All nodes should have some score
        for node in ["A", "B", "C", "D"] {
            assert!(scores.contains_key(node));
            assert!(scores[node] > 0.0);
        }
        
        // B and C should have higher scores due to incoming links
        assert!(scores["B"] > scores["A"]);
        assert!(scores["C"] > scores["D"]);
    }

    #[test]
    fn test_streaming_pagerank_update() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        let mut pagerank = StreamingPageRank::default();
        pagerank.initialize(&processor).unwrap();
        
        let initial_scores = pagerank.get_result().clone();
        
        // Add new edge A->D
        processor.add_edge("A".to_string(), "D".to_string(), 1.0).unwrap();
        
        // Create fake update result
        let update_result = crate::streaming::incremental::UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 1,
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };
        
        pagerank.update(&processor, &update_result).unwrap();
        
        let updated_scores = pagerank.get_result();
        
        // D's score should have increased
        assert!(updated_scores["D"] > initial_scores["D"]);
    }

    #[test]
    fn test_streaming_connected_components_initialization() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut components = StreamingConnectedComponents::new();
        components.initialize(&processor).unwrap();
        
        let result = components.get_result();
        assert_eq!(result.len(), 4); // A, B, C, D
        
        // A, B, C should be in same component
        let comp_a = &result["A"];
        let comp_b = &result["B"];
        let comp_c = &result["C"];
        assert_eq!(comp_a, comp_b);
        assert_eq!(comp_b, comp_c);
        
        // D should be in its own component
        let comp_d = &result["D"];
        assert_ne!(comp_a, comp_d);
        
        // Should have 2 components total
        assert_eq!(components.component_count(), 2);
    }

    #[test]
    fn test_streaming_connected_components_update() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1); // Force immediate flush
        
        let mut components = StreamingConnectedComponents::new();
        components.initialize(&processor).unwrap();
        
        assert_eq!(components.component_count(), 2); // {A,B,C} and {D}
        
        // Add edge to connect D to the main component
        processor.add_edge("C".to_string(), "D".to_string(), 1.0).unwrap();
        
        let update_result = crate::streaming::incremental::UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 1,
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };
        
        components.update(&processor, &update_result).unwrap();
        
        // Now should have only 1 component
        assert_eq!(components.component_count(), 1);
        
        let result = components.get_result();
        let comp_a = &result["A"];
        let comp_d = &result["D"];
        assert_eq!(comp_a, comp_d); // All nodes in same component
    }

    #[test]
    fn test_streaming_algorithm_recomputation() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut pagerank = StreamingPageRank::default();
        pagerank.initialize(&processor).unwrap();
        
        // Simulate large changes that require recomputation
        let large_changes = crate::streaming::incremental::UpdateResult {
            vertices_added: 15,
            vertices_removed: 5,
            edges_added: 20,
            edges_removed: 10,
            affected_components: vec![],
            recomputation_needed: true,
        };
        
        assert!(pagerank.needs_recomputation(&large_changes));
        
        // Should trigger recomputation
        pagerank.update(&processor, &large_changes).unwrap();
        
        // Algorithm should still work after recomputation
        let scores = pagerank.get_result();
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_pagerank_top_nodes() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut pagerank = StreamingPageRank::default();
        pagerank.initialize(&processor).unwrap();
        
        let top_2 = pagerank.top_nodes(2);
        assert_eq!(top_2.len(), 2);
        
        // Should be sorted by score descending
        assert!(top_2[0].1 >= top_2[1].1);
        
        // Check specific node score
        assert!(pagerank.node_score("A").is_some());
        assert!(pagerank.node_score("nonexistent").is_none());
    }

    #[test]
    fn test_connected_components_queries() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut components = StreamingConnectedComponents::new();
        components.initialize(&processor).unwrap();
        
        // Test component queries
        assert!(components.component_of("A").is_some());
        assert!(components.component_of("nonexistent").is_none());
        
        assert!(components.component_size("A").is_some());
        assert_eq!(components.component_size("A"), Some(3)); // A, B, C
        assert_eq!(components.component_size("D"), Some(1)); // D alone
        
        let all_components = components.all_components();
        assert_eq!(all_components.len(), 2);
        
        // Check total sizes
        let total_size: usize = all_components.iter().map(|(_, size)| size).sum();
        assert_eq!(total_size, 4); // All 4 nodes
    }
}