use crate::error::Result;
use crate::streaming::incremental::IncrementalGraphProcessor;
use arrow::array::Array;
use std::collections::{HashMap, HashSet, VecDeque};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Graph sampling strategies for real-time analytics on large graphs
/// These strategies provide representative subgraphs for efficient processing

/// Random sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub seed: Option<u64>,
    pub sample_rate: f64,        // Fraction of elements to sample (0.0 - 1.0)
    pub min_sample_size: usize,  // Minimum number of elements to sample
    pub max_sample_size: usize,  // Maximum number of elements to sample
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: None,
            sample_rate: 0.1, // 10% sample
            min_sample_size: 100,
            max_sample_size: 10000,
        }
    }
}

/// Node sampling strategy that selects representative nodes
#[derive(Debug)]
pub struct NodeSampler {
    rng: Pcg64,
    config: SamplingConfig,
    sampled_nodes: HashSet<String>,
    #[allow(dead_code)]
    node_scores: HashMap<String, f64>, // Scores for importance-based sampling
}

impl NodeSampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            Pcg64::seed_from_u64(seed)
        } else {
            Pcg64::from_entropy()
        };

        Self {
            rng,
            config,
            sampled_nodes: HashSet::new(),
            node_scores: HashMap::new(),
        }
    }

    /// Uniform random node sampling
    pub fn uniform_sample(&mut self, processor: &IncrementalGraphProcessor) -> Result<HashSet<String>> {
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        
        let mut all_nodes = Vec::new();
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for i in 0..node_ids.len() {
                all_nodes.push(node_ids.value(i).to_string());
            }
        }

        let sample_size = self.calculate_sample_size(all_nodes.len());
        let mut sampled = HashSet::new();

        while sampled.len() < sample_size && sampled.len() < all_nodes.len() {
            let index = self.rng.gen_range(0..all_nodes.len());
            sampled.insert(all_nodes[index].clone());
        }

        self.sampled_nodes = sampled.clone();
        Ok(sampled)
    }

    /// Degree-based importance sampling (higher degree = higher probability)
    pub fn degree_based_sample(&mut self, processor: &IncrementalGraphProcessor) -> Result<HashSet<String>> {
        let _graph = processor.graph();
        let degree_map = self.calculate_node_degrees(processor)?;
        
        // Calculate sampling probabilities based on degree
        let total_degree: u32 = degree_map.values().sum();
        let mut cumulative_probs = Vec::new();
        let mut nodes = Vec::new();
        let mut cumulative = 0.0;
        
        for (node, degree) in &degree_map {
            nodes.push(node.clone());
            cumulative += (*degree as f64) / (total_degree as f64);
            cumulative_probs.push(cumulative);
        }

        let sample_size = self.calculate_sample_size(nodes.len());
        let mut sampled = HashSet::new();

        for _ in 0..sample_size {
            let rand_val = self.rng.gen::<f64>();
            
            // Find first cumulative probability >= rand_val
            for (i, &prob) in cumulative_probs.iter().enumerate() {
                if rand_val <= prob {
                    sampled.insert(nodes[i].clone());
                    break;
                }
            }
        }

        self.sampled_nodes = sampled.clone();
        Ok(sampled)
    }

    /// PageRank-based importance sampling
    pub fn pagerank_based_sample(&mut self, _processor: &IncrementalGraphProcessor, pagerank_scores: &HashMap<String, f64>) -> Result<HashSet<String>> {
        // Calculate sampling probabilities based on PageRank scores
        let total_score: f64 = pagerank_scores.values().sum();
        let mut cumulative_probs = Vec::new();
        let mut nodes = Vec::new();
        let mut cumulative = 0.0;
        
        for (node, score) in pagerank_scores {
            nodes.push(node.clone());
            cumulative += score / total_score;
            cumulative_probs.push(cumulative);
        }

        let sample_size = self.calculate_sample_size(nodes.len());
        let mut sampled = HashSet::new();

        for _ in 0..sample_size {
            let rand_val = self.rng.gen::<f64>();
            
            for (i, &prob) in cumulative_probs.iter().enumerate() {
                if rand_val <= prob {
                    sampled.insert(nodes[i].clone());
                    break;
                }
            }
        }

        self.sampled_nodes = sampled.clone();
        Ok(sampled)
    }

    /// Calculate node degrees
    fn calculate_node_degrees(&self, processor: &IncrementalGraphProcessor) -> Result<HashMap<String, u32>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        let mut degrees = HashMap::new();

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
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                *degrees.entry(source).or_insert(0) += 1;
                *degrees.entry(target).or_insert(0) += 1;
            }
        }

        Ok(degrees)
    }

    fn calculate_sample_size(&self, total_size: usize) -> usize {
        let target_size = (total_size as f64 * self.config.sample_rate) as usize;
        target_size.clamp(self.config.min_sample_size, self.config.max_sample_size.min(total_size))
    }

    pub fn sampled_nodes(&self) -> &HashSet<String> {
        &self.sampled_nodes
    }
}

/// Edge sampling strategy that selects representative edges
#[derive(Debug)]
pub struct EdgeSampler {
    rng: Pcg64,
    config: SamplingConfig,
    sampled_edges: HashSet<(String, String)>,
}

impl EdgeSampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            Pcg64::seed_from_u64(seed)
        } else {
            Pcg64::from_entropy()
        };

        Self {
            rng,
            config,
            sampled_edges: HashSet::new(),
        }
    }

    /// Uniform random edge sampling
    pub fn uniform_sample(&mut self, processor: &IncrementalGraphProcessor) -> Result<HashSet<(String, String)>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        
        let mut all_edges = Vec::new();
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
                all_edges.push((source_ids.value(i).to_string(), target_ids.value(i).to_string()));
            }
        }

        let sample_size = self.calculate_sample_size(all_edges.len());
        let mut sampled = HashSet::new();

        while sampled.len() < sample_size && sampled.len() < all_edges.len() {
            let index = self.rng.gen_range(0..all_edges.len());
            sampled.insert(all_edges[index].clone());
        }

        self.sampled_edges = sampled.clone();
        Ok(sampled)
    }

    /// Weight-based edge sampling (higher weight = higher probability)
    pub fn weight_based_sample(&mut self, processor: &IncrementalGraphProcessor) -> Result<HashSet<(String, String)>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        
        let mut edges_with_weights = Vec::new();
        let mut total_weight = 0.0;

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
                let weight = weights.value(i).abs(); // Use absolute weight
                edges_with_weights.push(((source_ids.value(i).to_string(), target_ids.value(i).to_string()), weight));
                total_weight += weight;
            }
        }

        // Calculate cumulative probabilities
        let mut cumulative_probs = Vec::new();
        let mut cumulative = 0.0;
        
        for (_, weight) in &edges_with_weights {
            cumulative += weight / total_weight;
            cumulative_probs.push(cumulative);
        }

        let sample_size = self.calculate_sample_size(edges_with_weights.len());
        let mut sampled = HashSet::new();

        for _ in 0..sample_size {
            let rand_val = self.rng.gen::<f64>();
            
            for (i, &prob) in cumulative_probs.iter().enumerate() {
                if rand_val <= prob {
                    sampled.insert(edges_with_weights[i].0.clone());
                    break;
                }
            }
        }

        self.sampled_edges = sampled.clone();
        Ok(sampled)
    }

    fn calculate_sample_size(&self, total_size: usize) -> usize {
        let target_size = (total_size as f64 * self.config.sample_rate) as usize;
        target_size.clamp(self.config.min_sample_size, self.config.max_sample_size.min(total_size))
    }

    pub fn sampled_edges(&self) -> &HashSet<(String, String)> {
        &self.sampled_edges
    }
}

/// Subgraph sampling strategy that extracts connected subgraphs
#[derive(Debug)]
pub struct SubgraphSampler {
    rng: Pcg64,
    config: SamplingConfig,
    sampled_subgraph: SampledSubgraph,
}

/// A sampled subgraph containing nodes and edges
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    pub nodes: HashSet<String>,
    pub edges: HashSet<(String, String)>,
    pub sampling_method: String,
}

impl SubgraphSampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            Pcg64::seed_from_u64(seed)
        } else {
            Pcg64::from_entropy()
        };

        Self {
            rng,
            config,
            sampled_subgraph: SampledSubgraph {
                nodes: HashSet::new(),
                edges: HashSet::new(),
                sampling_method: "none".to_string(),
            },
        }
    }

    /// Random walk sampling starting from a random node
    pub fn random_walk_sample(&mut self, processor: &IncrementalGraphProcessor, walk_length: usize) -> Result<SampledSubgraph> {
        let _graph = processor.graph();
        let adjacency = self.build_adjacency_list(processor)?;
        
        // Get all nodes
        let all_nodes: Vec<String> = adjacency.keys().cloned().collect();
        if all_nodes.is_empty() {
            return Ok(SampledSubgraph {
                nodes: HashSet::new(),
                edges: HashSet::new(),
                sampling_method: "random_walk".to_string(),
            });
        }

        // Start from random node
        let start_node = &all_nodes[self.rng.gen_range(0..all_nodes.len())];
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut current_node = start_node.clone();

        visited_nodes.insert(current_node.clone());

        for _ in 0..walk_length {
            if let Some(neighbors) = adjacency.get(&current_node) {
                if neighbors.is_empty() {
                    break; // Dead end
                }
                
                let next_node = &neighbors[self.rng.gen_range(0..neighbors.len())];
                visited_edges.insert((current_node.clone(), next_node.clone()));
                visited_nodes.insert(next_node.clone());
                current_node = next_node.clone();
            } else {
                break; // No neighbors
            }
        }

        let subgraph = SampledSubgraph {
            nodes: visited_nodes,
            edges: visited_edges,
            sampling_method: "random_walk".to_string(),
        };

        self.sampled_subgraph = subgraph.clone();
        Ok(subgraph)
    }

    /// Breadth-First Search (BFS) sampling from multiple seed nodes
    pub fn bfs_sample(&mut self, processor: &IncrementalGraphProcessor, num_seeds: usize, max_depth: usize) -> Result<SampledSubgraph> {
        let adjacency = self.build_adjacency_list(processor)?;
        let all_nodes: Vec<String> = adjacency.keys().cloned().collect();
        
        if all_nodes.is_empty() {
            return Ok(SampledSubgraph {
                nodes: HashSet::new(),
                edges: HashSet::new(),
                sampling_method: "bfs".to_string(),
            });
        }

        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut queue = VecDeque::new();

        // Select random seed nodes
        let actual_seeds = num_seeds.min(all_nodes.len());
        let mut selected_seeds = HashSet::new();
        
        while selected_seeds.len() < actual_seeds {
            let seed = &all_nodes[self.rng.gen_range(0..all_nodes.len())];
            if selected_seeds.insert(seed.clone()) {
                queue.push_back((seed.clone(), 0)); // (node, depth)
                visited_nodes.insert(seed.clone());
            }
        }

        // BFS exploration
        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(neighbors) = adjacency.get(&current_node) {
                for neighbor in neighbors {
                    visited_edges.insert((current_node.clone(), neighbor.clone()));
                    
                    if visited_nodes.insert(neighbor.clone()) {
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }
        }

        let subgraph = SampledSubgraph {
            nodes: visited_nodes,
            edges: visited_edges,
            sampling_method: "bfs".to_string(),
        };

        self.sampled_subgraph = subgraph.clone();
        Ok(subgraph)
    }

    /// Forest Fire sampling (spreads like fire with burning probability)
    pub fn forest_fire_sample(&mut self, processor: &IncrementalGraphProcessor, burn_probability: f64) -> Result<SampledSubgraph> {
        let adjacency = self.build_adjacency_list(processor)?;
        let all_nodes: Vec<String> = adjacency.keys().cloned().collect();
        
        if all_nodes.is_empty() {
            return Ok(SampledSubgraph {
                nodes: HashSet::new(),
                edges: HashSet::new(),
                sampling_method: "forest_fire".to_string(),
            });
        }

        // Start from random node
        let start_node = &all_nodes[self.rng.gen_range(0..all_nodes.len())];
        let mut visited_nodes = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut burning_queue = VecDeque::new();

        visited_nodes.insert(start_node.clone());
        burning_queue.push_back(start_node.clone());

        let target_size = self.calculate_sample_size(all_nodes.len());

        while !burning_queue.is_empty() && visited_nodes.len() < target_size {
            let current_node = burning_queue.pop_front().unwrap();
            
            if let Some(neighbors) = adjacency.get(&current_node) {
                for neighbor in neighbors {
                    // Fire spreads with burn_probability
                    if self.rng.gen::<f64>() < burn_probability {
                        visited_edges.insert((current_node.clone(), neighbor.clone()));
                        
                        if visited_nodes.insert(neighbor.clone()) {
                            burning_queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        let subgraph = SampledSubgraph {
            nodes: visited_nodes,
            edges: visited_edges,
            sampling_method: "forest_fire".to_string(),
        };

        self.sampled_subgraph = subgraph.clone();
        Ok(subgraph)
    }

    /// Build adjacency list representation
    fn build_adjacency_list(&self, processor: &IncrementalGraphProcessor) -> Result<HashMap<String, Vec<String>>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        let mut adjacency = HashMap::new();

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
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                adjacency.entry(source).or_insert_with(Vec::new).push(target);
            }
        }

        Ok(adjacency)
    }

    fn calculate_sample_size(&self, total_size: usize) -> usize {
        let target_size = (total_size as f64 * self.config.sample_rate) as usize;
        target_size.clamp(self.config.min_sample_size, self.config.max_sample_size.min(total_size))
    }

    pub fn sampled_subgraph(&self) -> &SampledSubgraph {
        &self.sampled_subgraph
    }
}

/// Reservoir sampling for streaming scenarios
#[derive(Debug)]
pub struct ReservoirSampler<T> {
    reservoir: Vec<T>,
    capacity: usize,
    count: usize,
    rng: Pcg64,
}

impl<T> ReservoirSampler<T> {
    pub fn new(capacity: usize, seed: Option<u64>) -> Self {
        let rng = if let Some(seed) = seed {
            Pcg64::seed_from_u64(seed)
        } else {
            Pcg64::from_entropy()
        };

        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            rng,
        }
    }

    /// Add an item to the reservoir sample
    pub fn add(&mut self, item: T) {
        self.count += 1;

        if self.reservoir.len() < self.capacity {
            // Reservoir not full, just add
            self.reservoir.push(item);
        } else {
            // Reservoir full, decide whether to replace
            let j = self.rng.gen_range(0..self.count);
            if j < self.capacity {
                self.reservoir[j] = item;
            }
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Get sample size
    pub fn sample_size(&self) -> usize {
        self.reservoir.len()
    }

    /// Get total items seen
    pub fn total_count(&self) -> usize {
        self.count
    }

    /// Reset the sampler
    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }
}

/// Combined graph sampling processor
#[derive(Debug)]
pub struct GraphSamplingProcessor {
    node_sampler: NodeSampler,
    edge_sampler: EdgeSampler,
    subgraph_sampler: SubgraphSampler,
    reservoir_nodes: ReservoirSampler<String>,
    reservoir_edges: ReservoirSampler<(String, String)>,
    config: SamplingConfig,
}

impl GraphSamplingProcessor {
    pub fn new(config: SamplingConfig) -> Self {
        let reservoir_capacity = config.max_sample_size;
        
        Self {
            node_sampler: NodeSampler::new(config.clone()),
            edge_sampler: EdgeSampler::new(config.clone()),
            subgraph_sampler: SubgraphSampler::new(config.clone()),
            reservoir_nodes: ReservoirSampler::new(reservoir_capacity, config.seed),
            reservoir_edges: ReservoirSampler::new(reservoir_capacity, config.seed),
            config,
        }
    }

    /// Perform comprehensive sampling of the graph
    pub fn sample_graph(&mut self, processor: &IncrementalGraphProcessor) -> Result<GraphSample> {
        // Sample nodes using different strategies
        let uniform_nodes = self.node_sampler.uniform_sample(processor)?;
        let degree_nodes = self.node_sampler.degree_based_sample(processor)?;
        
        // Sample edges
        let uniform_edges = self.edge_sampler.uniform_sample(processor)?;
        let weight_edges = self.edge_sampler.weight_based_sample(processor)?;
        
        // Sample subgraphs
        let random_walk_subgraph = self.subgraph_sampler.random_walk_sample(processor, 100)?;
        let bfs_subgraph = self.subgraph_sampler.bfs_sample(processor, 5, 3)?;
        let forest_fire_subgraph = self.subgraph_sampler.forest_fire_sample(processor, 0.7)?;

        Ok(GraphSample {
            uniform_nodes,
            degree_based_nodes: degree_nodes,
            uniform_edges,
            weight_based_edges: weight_edges,
            random_walk_subgraph,
            bfs_subgraph,
            forest_fire_subgraph,
            reservoir_nodes: self.reservoir_nodes.sample().to_vec(),
            reservoir_edges: self.reservoir_edges.sample().to_vec(),
        })
    }

    /// Update reservoir samplers with new graph elements
    pub fn update_reservoirs(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        let graph = processor.graph();
        
        // Update node reservoir
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for i in 0..node_ids.len() {
                self.reservoir_nodes.add(node_ids.value(i).to_string());
            }
        }

        // Update edge reservoir
        let edges_batch = &graph.edges;
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
                self.reservoir_edges.add((source_ids.value(i).to_string(), target_ids.value(i).to_string()));
            }
        }

        Ok(())
    }

    /// Get sampling statistics
    pub fn sampling_stats(&self) -> SamplingStats {
        SamplingStats {
            reservoir_node_count: self.reservoir_nodes.sample_size(),
            reservoir_edge_count: self.reservoir_edges.sample_size(),
            total_nodes_seen: self.reservoir_nodes.total_count(),
            total_edges_seen: self.reservoir_edges.total_count(),
            config: self.config.clone(),
        }
    }
}

/// Comprehensive graph sample containing multiple sampling strategies
#[derive(Debug, Clone)]
pub struct GraphSample {
    pub uniform_nodes: HashSet<String>,
    pub degree_based_nodes: HashSet<String>,
    pub uniform_edges: HashSet<(String, String)>,
    pub weight_based_edges: HashSet<(String, String)>,
    pub random_walk_subgraph: SampledSubgraph,
    pub bfs_subgraph: SampledSubgraph,
    pub forest_fire_subgraph: SampledSubgraph,
    pub reservoir_nodes: Vec<String>,
    pub reservoir_edges: Vec<(String, String)>,
}

/// Statistics about sampling performance
#[derive(Debug, Clone)]
pub struct SamplingStats {
    pub reservoir_node_count: usize,
    pub reservoir_edge_count: usize,
    pub total_nodes_seen: usize,
    pub total_edges_seen: usize,
    pub config: SamplingConfig,
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
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let node_ids = StringArray::from(vec!["A", "B", "C", "D", "E"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["A", "B", "C", "D", "A", "B"]);
        let targets = StringArray::from(vec!["B", "C", "D", "E", "C", "D"]);
        let weights = Float64Array::from(vec![1.0, 2.0, 3.0, 1.0, 4.0, 2.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_node_uniform_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.6, // 60%
            min_sample_size: 1,
            max_sample_size: 3,
        };
        
        let mut sampler = NodeSampler::new(config);
        let sampled = sampler.uniform_sample(&processor).unwrap();
        
        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 3); // Max sample size
        
        // All sampled nodes should be valid
        let valid_nodes = ["A", "B", "C", "D", "E"];
        for node in &sampled {
            assert!(valid_nodes.contains(&node.as_str()));
        }
    }

    #[test]
    fn test_edge_uniform_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.5,
            min_sample_size: 1,
            max_sample_size: 3,
        };
        
        let mut sampler = EdgeSampler::new(config);
        let sampled = sampler.uniform_sample(&processor).unwrap();
        
        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 3);
    }

    #[test]
    fn test_degree_based_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.6,
            min_sample_size: 1,
            max_sample_size: 3,
        };
        
        let mut sampler = NodeSampler::new(config);
        let sampled = sampler.degree_based_sample(&processor).unwrap();
        
        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 3);
    }

    #[test]
    fn test_random_walk_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.5,
            min_sample_size: 1,
            max_sample_size: 10,
        };
        
        let mut sampler = SubgraphSampler::new(config);
        let subgraph = sampler.random_walk_sample(&processor, 5).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert_eq!(subgraph.sampling_method, "random_walk");
        
        // Edges should connect nodes in the subgraph
        for (source, target) in &subgraph.edges {
            assert!(subgraph.nodes.contains(source));
            assert!(subgraph.nodes.contains(target));
        }
    }

    #[test]
    fn test_bfs_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.8,
            min_sample_size: 1,
            max_sample_size: 10,
        };
        
        let mut sampler = SubgraphSampler::new(config);
        let subgraph = sampler.bfs_sample(&processor, 2, 2).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert_eq!(subgraph.sampling_method, "bfs");
    }

    #[test]
    fn test_forest_fire_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.6,
            min_sample_size: 1,
            max_sample_size: 10,
        };
        
        let mut sampler = SubgraphSampler::new(config);
        let subgraph = sampler.forest_fire_sample(&processor, 0.7).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert_eq!(subgraph.sampling_method, "forest_fire");
    }

    #[test]
    fn test_reservoir_sampling() {
        let mut reservoir = ReservoirSampler::new(3, Some(42));
        
        // Add more items than capacity
        for i in 0..10 {
            reservoir.add(format!("item_{}", i));
        }
        
        assert_eq!(reservoir.sample_size(), 3); // Should maintain capacity
        assert_eq!(reservoir.total_count(), 10); // Should track all items seen
        
        // Sample should contain valid items
        for item in reservoir.sample() {
            assert!(item.starts_with("item_"));
        }
    }

    #[test]
    fn test_weight_based_edge_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.5,
            min_sample_size: 1,
            max_sample_size: 3,
        };
        
        let mut sampler = EdgeSampler::new(config);
        let sampled = sampler.weight_based_sample(&processor).unwrap();
        
        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 3);
    }

    #[test]
    fn test_comprehensive_graph_sampling() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.6,
            min_sample_size: 1,
            max_sample_size: 5,
        };
        
        let mut sampling_processor = GraphSamplingProcessor::new(config);
        let sample = sampling_processor.sample_graph(&processor).unwrap();
        
        // Check that all sampling methods produced results
        assert!(!sample.uniform_nodes.is_empty());
        assert!(!sample.degree_based_nodes.is_empty());
        assert!(!sample.uniform_edges.is_empty());
        assert!(!sample.weight_based_edges.is_empty());
        assert!(!sample.random_walk_subgraph.nodes.is_empty());
        assert!(!sample.bfs_subgraph.nodes.is_empty());
        assert!(!sample.forest_fire_subgraph.nodes.is_empty());
    }

    #[test]
    fn test_reservoir_updates() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let config = SamplingConfig {
            seed: Some(42),
            sample_rate: 0.5,
            min_sample_size: 1,
            max_sample_size: 3,
        };
        
        let mut sampling_processor = GraphSamplingProcessor::new(config);
        sampling_processor.update_reservoirs(&processor).unwrap();
        
        let stats = sampling_processor.sampling_stats();
        assert!(stats.reservoir_node_count > 0);
        assert!(stats.reservoir_edge_count > 0);
        assert!(stats.total_nodes_seen > 0);
        assert!(stats.total_edges_seen > 0);
    }
}