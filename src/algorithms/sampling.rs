use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, UInt32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

/// Random Walk implementation for graph sampling and ML feature generation
pub struct RandomWalk;

impl RandomWalk {
    /// Perform random walks starting from specified nodes
    fn compute_random_walks(
        &self,
        graph: &ArrowGraph,
        start_nodes: &[String],
        walk_length: usize,
        num_walks: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Vec<String>>> {
        if walk_length == 0 {
            return Err(GraphError::invalid_parameter(
                "walk_length must be greater than 0"
            ));
        }

        if num_walks == 0 {
            return Err(GraphError::invalid_parameter(
                "num_walks must be greater than 0"
            ));
        }

        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut all_walks = Vec::new();

        for start_node in start_nodes {
            if !graph.has_node(start_node) {
                return Err(GraphError::node_not_found(start_node.clone()));
            }

            for _ in 0..num_walks {
                let walk = self.single_random_walk(graph, start_node, walk_length, &mut rng)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Perform a single random walk from a starting node
    fn single_random_walk(
        &self,
        graph: &ArrowGraph,
        start_node: &str,
        walk_length: usize,
        rng: &mut Pcg64,
    ) -> Result<Vec<String>> {
        let mut walk = Vec::with_capacity(walk_length);
        let mut current_node = start_node.to_string();

        walk.push(current_node.clone());

        for _ in 1..walk_length {
            let neighbors = match graph.neighbors(&current_node) {
                Some(neighbors) => neighbors,
                None => break, // Dead end - end walk early
            };

            if neighbors.is_empty() {
                break; // No neighbors - end walk early
            }

            // Choose random neighbor
            let next_node = neighbors.choose(rng).unwrap();
            current_node = next_node.clone();
            walk.push(current_node.clone());
        }

        Ok(walk)
    }
}

impl GraphAlgorithm for RandomWalk {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let walk_length: usize = params.get("walk_length").unwrap_or(10);
        let num_walks: usize = params.get("num_walks").unwrap_or(10);
        let seed: Option<u64> = params.get("seed");
        
        // Get start nodes - if not specified, use all nodes
        let start_nodes: Vec<String> = if let Some(nodes) = params.get::<Vec<String>>("start_nodes") {
            nodes
        } else {
            graph.node_ids().cloned().collect()
        };

        let walks = self.compute_random_walks(graph, &start_nodes, walk_length, num_walks, seed)?;

        // Convert walks to Arrow format
        let schema = Arc::new(Schema::new(vec![
            Field::new("walk_id", DataType::UInt32, false),
            Field::new("step", DataType::UInt32, false),
            Field::new("node_id", DataType::Utf8, false),
        ]));

        let mut walk_ids = Vec::new();
        let mut steps = Vec::new();
        let mut node_ids = Vec::new();

        for (walk_id, walk) in walks.iter().enumerate() {
            for (step, node_id) in walk.iter().enumerate() {
                walk_ids.push(walk_id as u32);
                steps.push(step as u32);
                node_ids.push(node_id.clone());
            }
        }

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt32Array::from(walk_ids)),
                Arc::new(UInt32Array::from(steps)),
                Arc::new(StringArray::from(node_ids)),
            ],
        ).map_err(GraphError::from)
    }

    fn name(&self) -> &'static str {
        "random_walk"
    }

    fn description(&self) -> &'static str {
        "Generate random walks from specified nodes for graph sampling and ML feature generation"
    }
}

/// Node2Vec-style biased random walks with return parameter p and in-out parameter q
pub struct Node2VecWalk;

impl Node2VecWalk {
    /// Perform Node2Vec-style biased random walks
    fn compute_node2vec_walks(
        &self,
        graph: &ArrowGraph,
        start_nodes: &[String],
        walk_length: usize,
        num_walks: usize,
        p: f64, // Return parameter (controls likelihood of returning to previous node)
        q: f64, // In-out parameter (controls likelihood of exploring vs. staying local)
        seed: Option<u64>,
    ) -> Result<Vec<Vec<String>>> {
        if walk_length < 2 {
            return Err(GraphError::invalid_parameter(
                "walk_length must be at least 2 for Node2Vec walks"
            ));
        }

        if p <= 0.0 || q <= 0.0 {
            return Err(GraphError::invalid_parameter(
                "p and q parameters must be positive"
            ));
        }

        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut all_walks = Vec::new();

        for start_node in start_nodes {
            if !graph.has_node(start_node) {
                return Err(GraphError::node_not_found(start_node.clone()));
            }

            for _ in 0..num_walks {
                let walk = self.single_node2vec_walk(graph, start_node, walk_length, p, q, &mut rng)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Perform a single Node2Vec-style biased random walk
    fn single_node2vec_walk(
        &self,
        graph: &ArrowGraph,
        start_node: &str,
        walk_length: usize,
        p: f64,
        q: f64,
        rng: &mut Pcg64,
    ) -> Result<Vec<String>> {
        let mut walk = Vec::with_capacity(walk_length);
        walk.push(start_node.to_string());

        // First step is uniform random
        let first_neighbors = match graph.neighbors(start_node) {
            Some(neighbors) if !neighbors.is_empty() => neighbors,
            _ => return Ok(walk), // No neighbors - return single node walk
        };

        let second_node = first_neighbors.choose(rng).unwrap().clone();
        walk.push(second_node.clone());

        // Subsequent steps use biased probabilities
        for _ in 2..walk_length {
            let current_node = &walk[walk.len() - 1];
            let previous_node = &walk[walk.len() - 2];

            let neighbors = match graph.neighbors(current_node) {
                Some(neighbors) if !neighbors.is_empty() => neighbors,
                _ => break, // No neighbors - end walk
            };

            let next_node = self.choose_next_node_biased(
                graph, previous_node, current_node, &neighbors, p, q, rng
            )?;
            walk.push(next_node);
        }

        Ok(walk)
    }

    /// Choose next node based on Node2Vec biased probabilities
    fn choose_next_node_biased(
        &self,
        graph: &ArrowGraph,
        previous_node: &str,
        current_node: &str,
        neighbors: &[String],
        p: f64,
        q: f64,
        rng: &mut Pcg64,
    ) -> Result<String> {
        let mut probabilities = Vec::new();
        let mut cumulative_prob = 0.0;

        // Get neighbors of previous node for distance calculation
        let previous_neighbors: std::collections::HashSet<_> = match graph.neighbors(previous_node) {
            Some(neighbors) => neighbors.iter().collect(),
            None => std::collections::HashSet::new(),
        };

        for neighbor in neighbors {
            let weight = if neighbor == previous_node {
                // Return to previous node - controlled by p
                1.0 / p
            } else if previous_neighbors.contains(neighbor) {
                // Stay within local neighborhood - weight = 1
                1.0
            } else {
                // Explore further - controlled by q
                1.0 / q
            };

            // Apply edge weight if available
            let edge_weight = graph.edge_weight(current_node, neighbor).unwrap_or(1.0);
            let final_weight = weight * edge_weight;

            cumulative_prob += final_weight;
            probabilities.push((neighbor, cumulative_prob));
        }

        // Sample from the probability distribution
        let random_val = rng.gen::<f64>() * cumulative_prob;
        
        for (neighbor, cum_prob) in probabilities {
            if random_val <= cum_prob {
                return Ok(neighbor.clone());
            }
        }

        // Fallback - should not happen with proper probability distribution
        Ok(neighbors[0].clone())
    }
}

impl GraphAlgorithm for Node2VecWalk {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let walk_length: usize = params.get("walk_length").unwrap_or(80);
        let num_walks: usize = params.get("num_walks").unwrap_or(10);
        let p: f64 = params.get("p").unwrap_or(1.0);
        let q: f64 = params.get("q").unwrap_or(1.0);
        let seed: Option<u64> = params.get("seed");
        
        // Get start nodes - if not specified, use all nodes
        let start_nodes: Vec<String> = if let Some(nodes) = params.get::<Vec<String>>("start_nodes") {
            nodes
        } else {
            graph.node_ids().cloned().collect()
        };

        let walks = self.compute_node2vec_walks(graph, &start_nodes, walk_length, num_walks, p, q, seed)?;

        // Convert walks to Arrow format with additional Node2Vec metadata
        let schema = Arc::new(Schema::new(vec![
            Field::new("walk_id", DataType::UInt32, false),
            Field::new("step", DataType::UInt32, false),
            Field::new("node_id", DataType::Utf8, false),
            Field::new("p_param", DataType::Float64, false),
            Field::new("q_param", DataType::Float64, false),
        ]));

        let mut walk_ids = Vec::new();
        let mut steps = Vec::new();
        let mut node_ids = Vec::new();
        let mut p_params = Vec::new();
        let mut q_params = Vec::new();

        for (walk_id, walk) in walks.iter().enumerate() {
            for (step, node_id) in walk.iter().enumerate() {
                walk_ids.push(walk_id as u32);
                steps.push(step as u32);
                node_ids.push(node_id.clone());
                p_params.push(p);
                q_params.push(q);
            }
        }

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt32Array::from(walk_ids)),
                Arc::new(UInt32Array::from(steps)),
                Arc::new(StringArray::from(node_ids)),
                Arc::new(Float64Array::from(p_params)),
                Arc::new(Float64Array::from(q_params)),
            ],
        ).map_err(GraphError::from)
    }

    fn name(&self) -> &'static str {
        "node2vec"
    }

    fn description(&self) -> &'static str {
        "Generate Node2Vec-style biased random walks with return (p) and in-out (q) parameters"
    }
}

/// Graph sampling using various strategies
pub struct GraphSampling;

impl GraphSampling {
    /// Random node sampling
    pub fn random_node_sampling(
        &self,
        graph: &ArrowGraph,
        sample_size: usize,
        seed: Option<u64>,
    ) -> Result<Vec<String>> {
        let all_nodes: Vec<String> = graph.node_ids().cloned().collect();
        
        if sample_size >= all_nodes.len() {
            return Ok(all_nodes);
        }

        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let sampled_nodes = all_nodes.choose_multiple(&mut rng, sample_size).cloned().collect();
        Ok(sampled_nodes)
    }

    /// Random edge sampling
    pub fn random_edge_sampling(
        &self,
        graph: &ArrowGraph,
        sample_ratio: f64,
        seed: Option<u64>,
    ) -> Result<RecordBatch> {
        if !(0.0..=1.0).contains(&sample_ratio) {
            return Err(GraphError::invalid_parameter(
                "sample_ratio must be between 0.0 and 1.0"
            ));
        }

        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut sampled_sources = Vec::new();
        let mut sampled_targets = Vec::new();
        let mut sampled_weights = Vec::new();

        // Sample edges based on ratio
        for node_id in graph.node_ids() {
            if let Some(neighbors) = graph.neighbors(node_id) {
                for neighbor in neighbors {
                    if rng.gen::<f64>() < sample_ratio {
                        sampled_sources.push(node_id.clone());
                        sampled_targets.push(neighbor.clone());
                        sampled_weights.push(graph.edge_weight(node_id, neighbor).unwrap_or(1.0));
                    }
                }
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(sampled_sources)),
                Arc::new(StringArray::from(sampled_targets)),
                Arc::new(Float64Array::from(sampled_weights)),
            ],
        ).map_err(GraphError::from)
    }

    /// Snowball sampling (BFS-based expansion)
    pub fn snowball_sampling(
        &self,
        graph: &ArrowGraph,
        seed_nodes: &[String],
        k_hops: usize,
        max_nodes: Option<usize>,
    ) -> Result<Vec<String>> {
        if k_hops == 0 {
            return Ok(seed_nodes.to_vec());
        }

        let mut sampled_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut current_frontier: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Initialize with seed nodes
        for seed_node in seed_nodes {
            if !graph.has_node(seed_node) {
                return Err(GraphError::node_not_found(seed_node.clone()));
            }
            sampled_nodes.insert(seed_node.clone());
            current_frontier.insert(seed_node.clone());
        }

        // Expand k hops
        for _ in 0..k_hops {
            let mut next_frontier = std::collections::HashSet::new();

            for node in &current_frontier {
                if let Some(neighbors) = graph.neighbors(node) {
                    for neighbor in neighbors {
                        if !sampled_nodes.contains(neighbor) {
                            sampled_nodes.insert(neighbor.clone());
                            next_frontier.insert(neighbor.clone());

                            // Check if we've reached the maximum number of nodes
                            if let Some(max) = max_nodes {
                                if sampled_nodes.len() >= max {
                                    return Ok(sampled_nodes.into_iter().collect());
                                }
                            }
                        }
                    }
                }
            }

            current_frontier = next_frontier;
            if current_frontier.is_empty() {
                break; // No more nodes to expand
            }
        }

        Ok(sampled_nodes.into_iter().collect())
    }
}

impl GraphAlgorithm for GraphSampling {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let sampling_method: String = params.get("method").unwrap_or("random_node".to_string());

        match sampling_method.as_str() {
            "random_node" => {
                let sample_size: usize = params.get("sample_size").unwrap_or(graph.node_count() / 2);
                let seed: Option<u64> = params.get("seed");
                
                let sampled_nodes = self.random_node_sampling(graph, sample_size, seed)?;

                let schema = Arc::new(Schema::new(vec![
                    Field::new("node_id", DataType::Utf8, false),
                ]));

                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(StringArray::from(sampled_nodes))],
                ).map_err(GraphError::from)
            }
            "random_edge" => {
                let sample_ratio: f64 = params.get("sample_ratio").unwrap_or(0.5);
                let seed: Option<u64> = params.get("seed");
                
                self.random_edge_sampling(graph, sample_ratio, seed)
            }
            "snowball" => {
                let seed_nodes: Vec<String> = params.get("seed_nodes")
                    .unwrap_or_else(|| vec![graph.node_ids().next().unwrap().clone()]);
                let k_hops: usize = params.get("k_hops").unwrap_or(2);
                let max_nodes: Option<usize> = params.get("max_nodes");
                
                let sampled_nodes = self.snowball_sampling(graph, &seed_nodes, k_hops, max_nodes)?;

                let schema = Arc::new(Schema::new(vec![
                    Field::new("node_id", DataType::Utf8, false),
                ]));

                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(StringArray::from(sampled_nodes))],
                ).map_err(GraphError::from)
            }
            _ => Err(GraphError::invalid_parameter(format!(
                "Unknown sampling method: {}. Supported methods: random_node, random_edge, snowball",
                sampling_method
            )))
        }
    }

    fn name(&self) -> &'static str {
        "graph_sampling"
    }

    fn description(&self) -> &'static str {
        "Perform various graph sampling strategies including random node/edge sampling and snowball sampling"
    }
}