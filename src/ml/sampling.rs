use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::Array;
use std::collections::{HashMap, HashSet, VecDeque};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};

/// GraphSAINT sampler for scalable graph neural network training
/// Implements sampling for inductive training on large graphs
#[derive(Debug)]
pub struct GraphSAINTSampler {
    sampling_strategy: SAINTStrategy,
    num_subgraphs: usize,
    subgraph_size: usize,
    walk_length: usize,
    num_walks: usize,
    normalization: NormalizationType,
    rng: Pcg64,
}

/// GraphSAINT sampling strategies
#[derive(Debug, Clone)]
pub enum SAINTStrategy {
    Node,           // Node-wise sampling
    Edge,           // Edge-wise sampling  
    RandomWalk,     // Random walk sampling
    NodeSampling,   // Pure node sampling
    GraphSAGE,      // GraphSAGE-style sampling
}

/// FastGCN sampler for efficient graph convolution training
/// Implements importance sampling for large-scale GCN training
#[derive(Debug)]
pub struct FastGCNSampler {
    layer_sizes: Vec<usize>,
    importance_sampling: bool,
    use_variance_reduction: bool,
    batch_size: usize,
    num_layers: usize,
    rng: Pcg64,
}

/// Control variate sampler for variance reduction
#[derive(Debug)]
pub struct ControlVariateSampler {
    base_sampler: Box<dyn MLSamplingStrategy>,
    control_function: ControlFunction,
    history_size: usize,
    adaptation_rate: f64,
}

/// ML sampling strategy trait
pub trait MLSamplingStrategy: std::fmt::Debug {
    /// Sample a subgraph for training
    fn sample_subgraph(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph>;
    
    /// Get sampling parameters
    fn get_parameters(&self) -> SamplingParameters;
    
    /// Update sampling parameters based on training feedback
    fn update_parameters(&mut self, feedback: &TrainingFeedback) -> Result<()>;
}

/// Sampled subgraph result
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String)>,
    pub node_weights: HashMap<String, f64>,
    pub edge_weights: HashMap<(String, String), f64>,
    pub sampling_probability: f64,
    pub original_graph_size: (usize, usize), // (nodes, edges)
    pub subgraph_size: (usize, usize),
}

/// Sampling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub num_samples: usize,
    pub sample_size: usize,
    pub sampling_rate: f64,
    pub importance_weights: HashMap<String, f64>,
    pub variance_reduction: bool,
    pub adaptive_sampling: bool,
}

/// Training feedback for adaptive sampling
#[derive(Debug, Clone)]
pub struct TrainingFeedback {
    pub loss: f64,
    pub gradient_variance: f64,
    pub convergence_rate: f64,
    pub node_importance: HashMap<String, f64>,
    pub edge_importance: HashMap<(String, String), f64>,
}

/// Normalization types for sampling
#[derive(Debug, Clone)]
pub enum NormalizationType {
    None,
    Degree,
    PPR,        // Personalized PageRank
    Heat,       // Heat kernel
    Symmetric,  // Symmetric normalization
}

/// Control functions for variance reduction
#[derive(Debug, Clone)]
pub enum ControlFunction {
    Linear,
    Quadratic,
    Exponential,
    Adaptive,
}

/// Importance sampling result
#[derive(Debug, Clone)]
pub struct ImportanceSamplingResult {
    pub samples: Vec<String>,
    pub weights: Vec<f64>,
    pub total_weight: f64,
    pub effective_sample_size: f64,
}

/// Multi-layer sampling for hierarchical graph learning
#[derive(Debug)]
pub struct HierarchicalSampler {
    layer_configs: Vec<LayerSamplingConfig>,
    aggregation_strategy: AggregationStrategy,
    coarsening_ratio: f64,
}

/// Configuration for each layer in hierarchical sampling
#[derive(Debug, Clone)]
pub struct LayerSamplingConfig {
    pub sample_size: usize,
    pub sampling_strategy: SAINTStrategy,
    pub normalization: NormalizationType,
}

/// Aggregation strategies for hierarchical sampling
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    Mean,
    Max,
    Attention,
    GraphSAGE,
}

impl GraphSAINTSampler {
    /// Create a new GraphSAINT sampler
    pub fn new(strategy: SAINTStrategy, num_subgraphs: usize, subgraph_size: usize) -> Self {
        Self {
            sampling_strategy: strategy,
            num_subgraphs,
            subgraph_size,
            walk_length: 10,
            num_walks: 100,
            normalization: NormalizationType::Degree,
            rng: Pcg64::from_entropy(),
        }
    }

    /// Configure sampling parameters
    pub fn with_config(mut self, walk_length: usize, num_walks: usize, normalization: NormalizationType) -> Self {
        self.walk_length = walk_length;
        self.num_walks = num_walks;
        self.normalization = normalization;
        self
    }

    /// Sample multiple subgraphs for mini-batch training
    pub fn sample_batch(&mut self, graph: &ArrowGraph) -> Result<Vec<SampledSubgraph>> {
        let mut batch = Vec::new();
        
        for _ in 0..self.num_subgraphs {
            let subgraph = self.sample_subgraph(graph)?;
            batch.push(subgraph);
        }
        
        Ok(batch)
    }

    /// Sample a single subgraph using the configured strategy
    fn sample_single_subgraph(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        match &self.sampling_strategy {
            SAINTStrategy::Node => self.node_sampling(graph),
            SAINTStrategy::Edge => self.edge_sampling(graph),
            SAINTStrategy::RandomWalk => self.random_walk_sampling(graph),
            SAINTStrategy::NodeSampling => self.pure_node_sampling(graph),
            SAINTStrategy::GraphSAGE => self.graphsage_sampling(graph),
        }
    }

    /// Node-wise sampling strategy
    fn node_sampling(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let all_nodes = self.get_all_nodes(graph)?;
        let adjacency = self.build_adjacency_list(graph)?;
        
        // Sample nodes with probability proportional to their importance
        let node_probabilities = self.compute_node_importance(&all_nodes, &adjacency)?;
        let sampled_nodes = self.importance_sample_nodes(&all_nodes, &node_probabilities, self.subgraph_size)?;
        
        // Include all edges between sampled nodes
        let sampled_edges = self.get_induced_edges(graph, &sampled_nodes)?;
        
        // Compute sampling weights for normalization
        let node_weights = self.compute_node_weights(&sampled_nodes, &node_probabilities)?;
        let edge_weights = self.compute_edge_weights(&sampled_edges, graph)?;
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes,
            edges: sampled_edges,
            node_weights,
            edge_weights,
            sampling_probability: self.subgraph_size as f64 / all_nodes.len() as f64,
            original_graph_size: (all_nodes.len(), self.count_edges(graph)?),
            subgraph_size: (self.subgraph_size, 0), // Will be updated with actual edge count
        })
    }

    /// Edge-wise sampling strategy
    fn edge_sampling(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let all_edges = self.get_all_edges(graph)?;
        let all_nodes = self.get_all_nodes(graph)?;
        
        // Sample edges uniformly
        let num_edges_to_sample = (all_edges.len() as f64 * (self.subgraph_size as f64 / all_nodes.len() as f64)) as usize;
        let sampled_edges = self.uniform_sample_edges(&all_edges, num_edges_to_sample)?;
        
        // Collect all nodes from sampled edges
        let mut sampled_nodes = HashSet::new();
        for (source, target) in &sampled_edges {
            sampled_nodes.insert(source.clone());
            sampled_nodes.insert(target.clone());
        }
        
        let sampled_nodes: Vec<String> = sampled_nodes.into_iter().collect();
        
        // Uniform weights for edge sampling
        let node_weights = sampled_nodes.iter()
            .map(|node| (node.clone(), 1.0))
            .collect();
        let edge_weights = sampled_edges.iter()
            .map(|edge| (edge.clone(), 1.0))
            .collect();
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes,
            edges: sampled_edges,
            node_weights,
            edge_weights,
            sampling_probability: num_edges_to_sample as f64 / all_edges.len() as f64,
            original_graph_size: (all_nodes.len(), all_edges.len()),
            subgraph_size: (sampled_nodes.len(), sampled_edges.len()),
        })
    }

    /// Random walk sampling strategy
    fn random_walk_sampling(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let all_nodes = self.get_all_nodes(graph)?;
        let adjacency = self.build_adjacency_list(graph)?;
        
        let mut sampled_nodes = HashSet::new();
        
        // Perform multiple random walks
        for _ in 0..self.num_walks {
            if all_nodes.is_empty() {
                continue;
            }
            
            let start_node = &all_nodes[self.rng.gen_range(0..all_nodes.len())];
            let walk = self.perform_random_walk(start_node, &adjacency)?;
            
            for node in walk {
                sampled_nodes.insert(node);
                if sampled_nodes.len() >= self.subgraph_size {
                    break;
                }
            }
            
            if sampled_nodes.len() >= self.subgraph_size {
                break;
            }
        }
        
        let sampled_nodes: Vec<String> = sampled_nodes.into_iter().collect();
        let sampled_edges = self.get_induced_edges(graph, &sampled_nodes)?;
        
        // Compute weights based on random walk probabilities
        let node_weights = self.compute_random_walk_weights(&sampled_nodes, &adjacency)?;
        let edge_weights = sampled_edges.iter()
            .map(|edge| (edge.clone(), 1.0))
            .collect();
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes,
            edges: sampled_edges,
            node_weights,
            edge_weights,
            sampling_probability: sampled_nodes.len() as f64 / all_nodes.len() as f64,
            original_graph_size: (all_nodes.len(), self.count_edges(graph)?),
            subgraph_size: (sampled_nodes.len(), sampled_edges.len()),
        })
    }

    /// Pure node sampling (uniform)
    fn pure_node_sampling(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let all_nodes = self.get_all_nodes(graph)?;
        
        // Sample nodes uniformly without replacement
        let sampled_nodes = self.uniform_sample_nodes(&all_nodes, self.subgraph_size)?;
        let sampled_edges = self.get_induced_edges(graph, &sampled_nodes)?;
        
        // Uniform weights
        let node_weights = sampled_nodes.iter()
            .map(|node| (node.clone(), 1.0))
            .collect();
        let edge_weights = sampled_edges.iter()
            .map(|edge| (edge.clone(), 1.0))
            .collect();
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes,
            edges: sampled_edges,
            node_weights,
            edge_weights,
            sampling_probability: self.subgraph_size as f64 / all_nodes.len() as f64,
            original_graph_size: (all_nodes.len(), self.count_edges(graph)?),
            subgraph_size: (sampled_nodes.len(), sampled_edges.len()),
        })
    }

    /// GraphSAGE-style neighborhood sampling
    fn graphsage_sampling(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let all_nodes = self.get_all_nodes(graph)?;
        let adjacency = self.build_adjacency_list(graph)?;
        
        // Start with a small set of nodes
        let seed_size = (self.subgraph_size as f64 * 0.3) as usize;
        let seed_nodes = self.uniform_sample_nodes(&all_nodes, seed_size)?;
        
        let mut sampled_nodes = HashSet::new();
        for node in seed_nodes {
            sampled_nodes.insert(node);
        }
        
        // Expand by sampling neighbors
        let target_neighbors = 5; // Number of neighbors per node
        let mut queue = VecDeque::new();
        for node in &sampled_nodes.clone() {
            queue.push_back(node.clone());
        }
        
        while let Some(current_node) = queue.pop_front() {
            if sampled_nodes.len() >= self.subgraph_size {
                break;
            }
            
            if let Some(neighbors) = adjacency.get(&current_node) {
                let sample_size = target_neighbors.min(neighbors.len());
                let sampled_neighbors = self.uniform_sample_from_vec(neighbors, sample_size)?;
                
                for neighbor in sampled_neighbors {
                    if sampled_nodes.len() >= self.subgraph_size {
                        break;
                    }
                    if sampled_nodes.insert(neighbor.clone()) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        let sampled_nodes: Vec<String> = sampled_nodes.into_iter().collect();
        let sampled_edges = self.get_induced_edges(graph, &sampled_nodes)?;
        
        // Compute GraphSAGE-style weights
        let node_weights = self.compute_graphsage_weights(&sampled_nodes, &adjacency)?;
        let edge_weights = sampled_edges.iter()
            .map(|edge| (edge.clone(), 1.0))
            .collect();
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes,
            edges: sampled_edges,
            node_weights,
            edge_weights,
            sampling_probability: sampled_nodes.len() as f64 / all_nodes.len() as f64,
            original_graph_size: (all_nodes.len(), self.count_edges(graph)?),
            subgraph_size: (sampled_nodes.len(), sampled_edges.len()),
        })
    }

    /// Helper methods
    fn get_all_nodes(&self, graph: &ArrowGraph) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }
        
        Ok(nodes)
    }

    fn get_all_edges(&self, graph: &ArrowGraph) -> Result<Vec<(String, String)>> {
        let mut edges = Vec::new();
        
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
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                edges.push((source, target));
            }
        }
        
        Ok(edges)
    }

    fn build_adjacency_list(&self, graph: &ArrowGraph) -> Result<HashMap<String, Vec<String>>> {
        let mut adjacency = HashMap::new();
        
        // Initialize with all nodes
        let nodes = self.get_all_nodes(graph)?;
        for node in nodes {
            adjacency.insert(node, Vec::new());
        }
        
        // Add edges
        let edges = self.get_all_edges(graph)?;
        for (source, target) in edges {
            adjacency.entry(source.clone()).or_default().push(target.clone());
            adjacency.entry(target).or_default().push(source); // Undirected
        }
        
        Ok(adjacency)
    }

    fn compute_node_importance(&self, nodes: &[String], adjacency: &HashMap<String, Vec<String>>) -> Result<HashMap<String, f64>> {
        let mut importance = HashMap::new();
        
        match &self.normalization {
            NormalizationType::Degree => {
                for node in nodes {
                    let degree = adjacency.get(node).map(|neighbors| neighbors.len()).unwrap_or(0);
                    importance.insert(node.clone(), degree as f64 + 1.0); // +1 to avoid zero
                }
            }
            NormalizationType::PPR => {
                // Simplified PPR - in practice would use power iteration
                for node in nodes {
                    importance.insert(node.clone(), 1.0 / nodes.len() as f64);
                }
            }
            _ => {
                // Uniform importance
                for node in nodes {
                    importance.insert(node.clone(), 1.0);
                }
            }
        }
        
        Ok(importance)
    }

    fn importance_sample_nodes(&mut self, nodes: &[String], probabilities: &HashMap<String, f64>, sample_size: usize) -> Result<Vec<String>> {
        let total_weight: f64 = probabilities.values().sum();
        let mut sampled = Vec::new();
        let mut used_indices = HashSet::new();
        
        for _ in 0..sample_size.min(nodes.len()) {
            let mut cumulative = 0.0;
            let target = self.rng.gen::<f64>() * total_weight;
            
            for (i, node) in nodes.iter().enumerate() {
                if used_indices.contains(&i) {
                    continue;
                }
                
                cumulative += probabilities.get(node).unwrap_or(&1.0);
                if cumulative >= target {
                    sampled.push(node.clone());
                    used_indices.insert(i);
                    break;
                }
            }
        }
        
        Ok(sampled)
    }

    fn uniform_sample_nodes(&mut self, nodes: &[String], sample_size: usize) -> Result<Vec<String>> {
        let mut sampled = Vec::new();
        let mut used_indices = HashSet::new();
        
        for _ in 0..sample_size.min(nodes.len()) {
            let mut idx = self.rng.gen_range(0..nodes.len());
            while used_indices.contains(&idx) {
                idx = self.rng.gen_range(0..nodes.len());
            }
            
            sampled.push(nodes[idx].clone());
            used_indices.insert(idx);
        }
        
        Ok(sampled)
    }

    fn uniform_sample_edges(&mut self, edges: &[(String, String)], sample_size: usize) -> Result<Vec<(String, String)>> {
        let mut sampled = Vec::new();
        let mut used_indices = HashSet::new();
        
        for _ in 0..sample_size.min(edges.len()) {
            let mut idx = self.rng.gen_range(0..edges.len());
            while used_indices.contains(&idx) {
                idx = self.rng.gen_range(0..edges.len());
            }
            
            sampled.push(edges[idx].clone());
            used_indices.insert(idx);
        }
        
        Ok(sampled)
    }

    fn uniform_sample_from_vec(&mut self, items: &[String], sample_size: usize) -> Result<Vec<String>> {
        let mut sampled = Vec::new();
        let mut used_indices = HashSet::new();
        
        for _ in 0..sample_size.min(items.len()) {
            let mut idx = self.rng.gen_range(0..items.len());
            while used_indices.contains(&idx) {
                idx = self.rng.gen_range(0..items.len());
            }
            
            sampled.push(items[idx].clone());
            used_indices.insert(idx);
        }
        
        Ok(sampled)
    }

    fn get_induced_edges(&self, graph: &ArrowGraph, nodes: &[String]) -> Result<Vec<(String, String)>> {
        let node_set: HashSet<_> = nodes.iter().collect();
        let mut induced_edges = Vec::new();
        
        let edges = self.get_all_edges(graph)?;
        for (source, target) in edges {
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push((source, target));
            }
        }
        
        Ok(induced_edges)
    }

    fn count_edges(&self, graph: &ArrowGraph) -> Result<usize> {
        Ok(graph.edges.num_rows())
    }

    fn perform_random_walk(&mut self, start_node: &str, adjacency: &HashMap<String, Vec<String>>) -> Result<Vec<String>> {
        let mut walk = vec![start_node.to_string()];
        let mut current = start_node;
        
        for _ in 1..self.walk_length {
            if let Some(neighbors) = adjacency.get(current) {
                if neighbors.is_empty() {
                    break;
                }
                current = &neighbors[self.rng.gen_range(0..neighbors.len())];
                walk.push(current.to_string());
            } else {
                break;
            }
        }
        
        Ok(walk)
    }

    fn compute_node_weights(&self, nodes: &[String], probabilities: &HashMap<String, f64>) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        let total_prob: f64 = probabilities.values().sum();
        
        for node in nodes {
            let prob = probabilities.get(node).unwrap_or(&1.0);
            let weight = total_prob / (prob * nodes.len() as f64);
            weights.insert(node.clone(), weight);
        }
        
        Ok(weights)
    }

    fn compute_edge_weights(&self, edges: &[(String, String)], _graph: &ArrowGraph) -> Result<HashMap<(String, String), f64>> {
        let mut weights = HashMap::new();
        
        for edge in edges {
            weights.insert(edge.clone(), 1.0); // Simplified - uniform weights
        }
        
        Ok(weights)
    }

    fn compute_random_walk_weights(&self, nodes: &[String], adjacency: &HashMap<String, Vec<String>>) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        
        for node in nodes {
            let degree = adjacency.get(node).map(|neighbors| neighbors.len()).unwrap_or(0);
            let weight = if degree > 0 { 1.0 / degree as f64 } else { 1.0 };
            weights.insert(node.clone(), weight);
        }
        
        Ok(weights)
    }

    fn compute_graphsage_weights(&self, nodes: &[String], adjacency: &HashMap<String, Vec<String>>) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        
        for node in nodes {
            let degree = adjacency.get(node).map(|neighbors| neighbors.len()).unwrap_or(0);
            // GraphSAGE-style normalization
            let weight = 1.0 / (degree as f64).sqrt().max(1.0);
            weights.insert(node.clone(), weight);
        }
        
        Ok(weights)
    }
}

impl MLSamplingStrategy for GraphSAINTSampler {
    fn sample_subgraph(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        self.sample_single_subgraph(graph)
    }

    fn get_parameters(&self) -> SamplingParameters {
        SamplingParameters {
            num_samples: self.num_subgraphs,
            sample_size: self.subgraph_size,
            sampling_rate: self.subgraph_size as f64 / 1000.0, // Placeholder
            importance_weights: HashMap::new(),
            variance_reduction: false,
            adaptive_sampling: false,
        }
    }

    fn update_parameters(&mut self, feedback: &TrainingFeedback) -> Result<()> {
        // Adaptive sampling based on training feedback
        if feedback.gradient_variance > 0.1 {
            // Increase sample size if variance is high
            self.subgraph_size = (self.subgraph_size as f64 * 1.1) as usize;
        } else if feedback.gradient_variance < 0.01 {
            // Decrease sample size if variance is low
            self.subgraph_size = (self.subgraph_size as f64 * 0.9) as usize;
        }
        
        // Adjust number of walks based on convergence
        if feedback.convergence_rate < 0.01 {
            self.num_walks = (self.num_walks as f64 * 1.2) as usize;
        }
        
        Ok(())
    }
}

impl FastGCNSampler {
    /// Create a new FastGCN sampler
    pub fn new(layer_sizes: Vec<usize>, batch_size: usize) -> Self {
        Self {
            layer_sizes,
            importance_sampling: true,
            use_variance_reduction: true,
            batch_size,
            num_layers: 2,
            rng: Pcg64::from_entropy(),
        }
    }

    /// Sample nodes for each layer using importance sampling
    pub fn sample_layers(&mut self, graph: &ArrowGraph) -> Result<Vec<ImportanceSamplingResult>> {
        let mut layer_samples = Vec::new();
        let adjacency = self.build_adjacency_matrix(graph)?;
        
        for (layer_idx, &layer_size) in self.layer_sizes.iter().enumerate() {
            let importance_weights = if self.importance_sampling {
                self.compute_importance_weights(graph, layer_idx)?
            } else {
                self.uniform_weights(graph)?
            };
            
            let sample_result = self.importance_sample(&importance_weights, layer_size)?;
            layer_samples.push(sample_result);
        }
        
        Ok(layer_samples)
    }

    /// Compute importance weights based on node degrees and layer position
    fn compute_importance_weights(&self, graph: &ArrowGraph, layer_idx: usize) -> Result<HashMap<String, f64>> {
        let nodes = self.get_all_nodes(graph)?;
        let adjacency = self.build_adjacency_matrix(graph)?;
        let mut weights = HashMap::new();
        
        for node in nodes {
            let degree = adjacency.get(&node).map(|neighbors| neighbors.len()).unwrap_or(0);
            
            // Importance decreases with layer depth
            let layer_factor = 1.0 / (layer_idx + 1) as f64;
            let importance = (degree as f64 + 1.0) * layer_factor;
            
            weights.insert(node, importance);
        }
        
        Ok(weights)
    }

    /// Create uniform weights for all nodes
    fn uniform_weights(&self, graph: &ArrowGraph) -> Result<HashMap<String, f64>> {
        let nodes = self.get_all_nodes(graph)?;
        let mut weights = HashMap::new();
        
        for node in nodes {
            weights.insert(node, 1.0);
        }
        
        Ok(weights)
    }

    /// Perform importance sampling
    fn importance_sample(&mut self, weights: &HashMap<String, f64>, sample_size: usize) -> Result<ImportanceSamplingResult> {
        let total_weight: f64 = weights.values().sum();
        let mut samples = Vec::new();
        let mut sample_weights = Vec::new();
        
        let nodes: Vec<_> = weights.keys().cloned().collect();
        let effective_sample_size = sample_size.min(nodes.len());
        
        for _ in 0..effective_sample_size {
            let target = self.rng.gen::<f64>() * total_weight;
            let mut cumulative = 0.0;
            
            for node in &nodes {
                cumulative += weights.get(node).unwrap_or(&1.0);
                if cumulative >= target {
                    let weight = total_weight / (weights.get(node).unwrap_or(&1.0) * effective_sample_size as f64);
                    samples.push(node.clone());
                    sample_weights.push(weight);
                    break;
                }
            }
        }
        
        let effective_sample_size = if sample_weights.is_empty() {
            0.0
        } else {
            let sum_weights: f64 = sample_weights.iter().sum();
            let sum_squared: f64 = sample_weights.iter().map(|w| w * w).sum();
            sum_weights * sum_weights / sum_squared
        };
        
        Ok(ImportanceSamplingResult {
            samples,
            weights: sample_weights,
            total_weight,
            effective_sample_size,
        })
    }

    fn get_all_nodes(&self, graph: &ArrowGraph) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }
        
        Ok(nodes)
    }

    fn build_adjacency_matrix(&self, graph: &ArrowGraph) -> Result<HashMap<String, Vec<String>>> {
        let mut adjacency = HashMap::new();
        
        // Initialize with all nodes
        let nodes = self.get_all_nodes(graph)?;
        for node in nodes {
            adjacency.insert(node, Vec::new());
        }
        
        // Add edges
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
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                adjacency.entry(source.clone()).or_default().push(target.clone());
                adjacency.entry(target).or_default().push(source); // Undirected
            }
        }
        
        Ok(adjacency)
    }
}

impl MLSamplingStrategy for FastGCNSampler {
    fn sample_subgraph(&mut self, graph: &ArrowGraph) -> Result<SampledSubgraph> {
        let layer_samples = self.sample_layers(graph)?;
        
        // Combine samples from all layers
        let mut all_nodes = HashSet::new();
        let mut node_weights = HashMap::new();
        
        for sample_result in layer_samples {
            for (node, weight) in sample_result.samples.iter().zip(sample_result.weights.iter()) {
                all_nodes.insert(node.clone());
                node_weights.insert(node.clone(), *weight);
            }
        }
        
        let sampled_nodes: Vec<String> = all_nodes.into_iter().collect();
        let sampled_edges = self.get_induced_edges(graph, &sampled_nodes)?;
        let edge_weights = sampled_edges.iter()
            .map(|edge| (edge.clone(), 1.0))
            .collect();
        
        let original_size = (self.get_all_nodes(graph)?.len(), graph.edges.num_rows());
        
        Ok(SampledSubgraph {
            nodes: sampled_nodes.clone(),
            edges: sampled_edges.clone(),
            node_weights,
            edge_weights,
            sampling_probability: sampled_nodes.len() as f64 / original_size.0 as f64,
            original_graph_size: original_size,
            subgraph_size: (sampled_nodes.len(), sampled_edges.len()),
        })
    }

    fn get_parameters(&self) -> SamplingParameters {
        SamplingParameters {
            num_samples: self.batch_size,
            sample_size: self.layer_sizes.iter().sum(),
            sampling_rate: 0.1, // Placeholder
            importance_weights: HashMap::new(),
            variance_reduction: self.use_variance_reduction,
            adaptive_sampling: true,
        }
    }

    fn update_parameters(&mut self, feedback: &TrainingFeedback) -> Result<()> {
        // Adjust layer sizes based on gradient variance
        if feedback.gradient_variance > 0.1 {
            for size in &mut self.layer_sizes {
                *size = (*size as f64 * 1.1) as usize;
            }
        } else if feedback.gradient_variance < 0.01 {
            for size in &mut self.layer_sizes {
                *size = (*size as f64 * 0.9) as usize;
            }
        }
        
        Ok(())
    }
}

impl FastGCNSampler {
    fn get_induced_edges(&self, graph: &ArrowGraph, nodes: &[String]) -> Result<Vec<(String, String)>> {
        let node_set: HashSet<_> = nodes.iter().collect();
        let mut induced_edges = Vec::new();
        
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
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                
                if node_set.contains(&source.to_string()) && node_set.contains(&target.to_string()) {
                    induced_edges.push((source.to_string(), target.to_string()));
                }
            }
        }
        
        Ok(induced_edges)
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
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let node_ids = StringArray::from(vec!["A", "B", "C", "D", "E", "F"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["A", "B", "C", "D", "E"]);
        let targets = StringArray::from(vec!["B", "C", "D", "E", "F"]);
        let weights = Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_graphsaint_node_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Node, 1, 3);
        
        let subgraph = sampler.sample_subgraph(&graph).unwrap();
        
        assert_eq!(subgraph.nodes.len(), 3);
        assert!(subgraph.sampling_probability > 0.0);
        assert_eq!(subgraph.original_graph_size.0, 6); // 6 nodes in test graph
    }

    #[test]
    fn test_graphsaint_edge_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Edge, 1, 4);
        
        let subgraph = sampler.sample_subgraph(&graph).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert!(!subgraph.edges.is_empty());
        assert_eq!(subgraph.original_graph_size.1, 5); // 5 edges in test graph
    }

    #[test]
    fn test_graphsaint_random_walk_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::RandomWalk, 1, 4)
            .with_config(5, 3, NormalizationType::Degree);
        
        let subgraph = sampler.sample_subgraph(&graph).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert!(subgraph.nodes.len() <= 4);
        assert!(!subgraph.node_weights.is_empty());
    }

    #[test]
    fn test_graphsaint_graphsage_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::GraphSAGE, 1, 4);
        
        let subgraph = sampler.sample_subgraph(&graph).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert!(subgraph.nodes.len() <= 4);
        // GraphSAGE sampling should produce connected subgraphs
        assert!(!subgraph.edges.is_empty() || subgraph.nodes.len() == 1);
    }

    #[test]
    fn test_graphsaint_batch_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Node, 3, 2);
        
        let batch = sampler.sample_batch(&graph).unwrap();
        
        assert_eq!(batch.len(), 3);
        for subgraph in batch {
            assert_eq!(subgraph.nodes.len(), 2);
        }
    }

    #[test]
    fn test_fastgcn_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = FastGCNSampler::new(vec![3, 2], 1);
        
        let subgraph = sampler.sample_subgraph(&graph).unwrap();
        
        assert!(!subgraph.nodes.is_empty());
        assert!(!subgraph.node_weights.is_empty());
        assert!(subgraph.sampling_probability > 0.0);
    }

    #[test]
    fn test_fastgcn_layer_sampling() {
        let graph = create_test_graph().unwrap();
        let mut sampler = FastGCNSampler::new(vec![4, 3, 2], 1);
        
        let layer_samples = sampler.sample_layers(&graph).unwrap();
        
        assert_eq!(layer_samples.len(), 3);
        assert_eq!(layer_samples[0].samples.len(), 4);
        assert_eq!(layer_samples[1].samples.len(), 3);
        assert_eq!(layer_samples[2].samples.len(), 2);
    }

    #[test]
    fn test_importance_sampling_weights() {
        let graph = create_test_graph().unwrap();
        let mut sampler = FastGCNSampler::new(vec![3], 1);
        
        let weights = sampler.compute_importance_weights(&graph, 0).unwrap();
        
        assert_eq!(weights.len(), 6); // 6 nodes
        for weight in weights.values() {
            assert!(*weight > 0.0);
        }
    }

    #[test]
    fn test_sampling_parameters() {
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Node, 5, 10);
        let params = sampler.get_parameters();
        
        assert_eq!(params.num_samples, 5);
        assert_eq!(params.sample_size, 10);
        assert!(!params.variance_reduction);
        assert!(!params.adaptive_sampling);
    }

    #[test]
    fn test_adaptive_parameter_update() {
        let graph = create_test_graph().unwrap();
        let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Node, 1, 3);
        
        let feedback = TrainingFeedback {
            loss: 0.5,
            gradient_variance: 0.15, // High variance
            convergence_rate: 0.005, // Slow convergence
            node_importance: HashMap::new(),
            edge_importance: HashMap::new(),
        };
        
        let original_size = sampler.subgraph_size;
        let original_walks = sampler.num_walks;
        
        sampler.update_parameters(&feedback).unwrap();
        
        // Should increase sample size due to high variance
        assert!(sampler.subgraph_size > original_size);
        // Should increase walks due to slow convergence
        assert!(sampler.num_walks > original_walks);
    }

    #[test]
    fn test_normalization_types() {
        let graph = create_test_graph().unwrap();
        
        for norm_type in [NormalizationType::Degree, NormalizationType::PPR, NormalizationType::None] {
            let mut sampler = GraphSAINTSampler::new(SAINTStrategy::Node, 1, 3)
                .with_config(5, 3, norm_type);
            
            let subgraph = sampler.sample_subgraph(&graph).unwrap();
            assert!(!subgraph.nodes.is_empty());
            assert!(!subgraph.node_weights.is_empty());
        }
    }

    #[test]
    fn test_effective_sample_size() {
        let graph = create_test_graph().unwrap();
        let mut sampler = FastGCNSampler::new(vec![4], 1);
        
        let layer_samples = sampler.sample_layers(&graph).unwrap();
        let sample_result = &layer_samples[0];
        
        assert!(sample_result.effective_sample_size > 0.0);
        assert!(sample_result.effective_sample_size <= sample_result.samples.len() as f64);
    }
}