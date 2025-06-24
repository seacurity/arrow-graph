use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::collections::HashMap;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

pub struct PageRank;

impl PageRank {
    /// PageRank algorithm with power iteration and early termination
    fn compute_pagerank(
        &self,
        graph: &ArrowGraph,
        damping_factor: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<HashMap<String, f64>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(HashMap::new());
        }
        
        // Initialize PageRank scores
        let initial_score = 1.0 / node_count as f64;
        let mut current_scores: HashMap<String, f64> = HashMap::new();
        let mut next_scores: HashMap<String, f64> = HashMap::new();
        
        // Initialize all nodes with equal probability
        for node_id in graph.node_ids() {
            current_scores.insert(node_id.clone(), initial_score);
            next_scores.insert(node_id.clone(), 0.0);
        }
        
        // Calculate out-degrees for each node
        let mut out_degrees: HashMap<String, usize> = HashMap::new();
        for node_id in graph.node_ids() {
            let degree = graph.neighbors(node_id).map(|n| n.len()).unwrap_or(0);
            out_degrees.insert(node_id.clone(), degree);
        }
        
        // Power iteration
        for iteration in 0..max_iterations {
            // Reset next scores
            for score in next_scores.values_mut() {
                *score = (1.0 - damping_factor) / node_count as f64;
            }
            
            // Distribute PageRank scores
            for node_id in graph.node_ids() {
                let current_score = current_scores.get(node_id).unwrap_or(&0.0);
                let out_degree = out_degrees.get(node_id).unwrap_or(&0);
                
                if *out_degree > 0 {
                    let contribution = current_score * damping_factor / *out_degree as f64;
                    
                    if let Some(neighbors) = graph.neighbors(node_id) {
                        for neighbor in neighbors {
                            if let Some(neighbor_score) = next_scores.get_mut(neighbor) {
                                *neighbor_score += contribution;
                            }
                        }
                    }
                } else {
                    // Handle dangling nodes - distribute equally to all nodes
                    let dangling_contribution = current_score * damping_factor / node_count as f64;
                    for score in next_scores.values_mut() {
                        *score += dangling_contribution;
                    }
                }
            }
            
            // Check for convergence
            let mut diff = 0.0;
            for node_id in graph.node_ids() {
                let old_score = current_scores.get(node_id).unwrap_or(&0.0);
                let new_score = next_scores.get(node_id).unwrap_or(&0.0);
                diff += (new_score - old_score).abs();
            }
            
            // Early termination if converged
            if diff < tolerance {
                log::debug!("PageRank converged after {} iterations", iteration + 1);
                break;
            }
            
            // Swap scores for next iteration
            std::mem::swap(&mut current_scores, &mut next_scores);
        }
        
        Ok(current_scores)
    }
}

impl GraphAlgorithm for PageRank {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let damping_factor: f64 = params.get("damping_factor").unwrap_or(0.85);
        let max_iterations: usize = params.get("max_iterations").unwrap_or(100);
        let tolerance: f64 = params.get("tolerance").unwrap_or(1e-6);
        
        // Validate parameters
        if !(0.0..=1.0).contains(&damping_factor) {
            return Err(GraphError::invalid_parameter(
                "damping_factor must be between 0.0 and 1.0"
            ));
        }
        
        if max_iterations == 0 {
            return Err(GraphError::invalid_parameter(
                "max_iterations must be greater than 0"
            ));
        }
        
        if tolerance <= 0.0 {
            return Err(GraphError::invalid_parameter(
                "tolerance must be greater than 0.0"
            ));
        }
        
        let scores = self.compute_pagerank(graph, damping_factor, max_iterations, tolerance)?;
        
        // Convert to Arrow RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("pagerank_score", DataType::Float64, false),
        ]));
        
        let mut node_ids = Vec::new();
        let mut pagerank_scores = Vec::new();
        
        // Sort by PageRank score (descending) for consistent output
        let mut sorted_scores: Vec<(&String, &f64)> = scores.iter().collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        for (node_id, score) in sorted_scores {
            node_ids.push(node_id.clone());
            pagerank_scores.push(*score);
        }
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(Float64Array::from(pagerank_scores)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "pagerank"
    }
    
    fn description(&self) -> &'static str {
        "Calculate PageRank scores using power iteration with early termination"
    }
}

pub struct BetweennessCentrality;

impl BetweennessCentrality {
    /// Calculate betweenness centrality using Brandes' algorithm
    fn compute_betweenness_centrality(&self, graph: &ArrowGraph) -> Result<HashMap<String, f64>> {
        let mut centrality: HashMap<String, f64> = HashMap::new();
        
        // Initialize centrality scores
        for node_id in graph.node_ids() {
            centrality.insert(node_id.clone(), 0.0);
        }
        
        // For each node as source
        for source in graph.node_ids() {
            let mut stack = Vec::new();
            let mut paths: HashMap<String, Vec<String>> = HashMap::new();
            let mut num_paths: HashMap<String, f64> = HashMap::new();
            let mut distances: HashMap<String, i32> = HashMap::new();
            let mut delta: HashMap<String, f64> = HashMap::new();
            
            // Initialize
            for node_id in graph.node_ids() {
                paths.insert(node_id.clone(), Vec::new());
                num_paths.insert(node_id.clone(), 0.0);
                distances.insert(node_id.clone(), -1);
                delta.insert(node_id.clone(), 0.0);
            }
            
            num_paths.insert(source.clone(), 1.0);
            distances.insert(source.clone(), 0);
            
            // BFS
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(source.clone());
            
            while let Some(current) = queue.pop_front() {
                stack.push(current.clone());
                
                if let Some(neighbors) = graph.neighbors(&current) {
                    for neighbor in neighbors {
                        let current_dist = *distances.get(&current).unwrap_or(&-1);
                        let neighbor_dist = *distances.get(neighbor).unwrap_or(&-1);
                        
                        // First time we reach this neighbor
                        if neighbor_dist < 0 {
                            queue.push_back(neighbor.clone());
                            distances.insert(neighbor.clone(), current_dist + 1);
                        }
                        
                        // Shortest path to neighbor via current
                        if neighbor_dist == current_dist + 1 {
                            let current_paths = *num_paths.get(&current).unwrap_or(&0.0);
                            let neighbor_paths = num_paths.get_mut(neighbor).unwrap();
                            *neighbor_paths += current_paths;
                            
                            paths.get_mut(neighbor).unwrap().push(current.clone());
                        }
                    }
                }
            }
            
            // Accumulation
            while let Some(w) = stack.pop() {
                if let Some(predecessors) = paths.get(&w) {
                    for predecessor in predecessors {
                        let w_delta = *delta.get(&w).unwrap_or(&0.0);
                        let w_paths = *num_paths.get(&w).unwrap_or(&0.0);
                        let pred_paths = *num_paths.get(predecessor).unwrap_or(&0.0);
                        
                        if pred_paths > 0.0 {
                            let contribution = (pred_paths / w_paths) * (1.0 + w_delta);
                            *delta.get_mut(predecessor).unwrap() += contribution;
                        }
                    }
                }
                
                if w != *source {
                    let w_delta = *delta.get(&w).unwrap_or(&0.0);
                    *centrality.get_mut(&w).unwrap() += w_delta;
                }
            }
        }
        
        // Normalize for undirected graphs
        let node_count = graph.node_count() as f64;
        if node_count > 2.0 {
            let normalization = 2.0 / ((node_count - 1.0) * (node_count - 2.0));
            for score in centrality.values_mut() {
                *score *= normalization;
            }
        }
        
        Ok(centrality)
    }
}

impl GraphAlgorithm for BetweennessCentrality {
    fn execute(&self, graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        let centrality = self.compute_betweenness_centrality(graph)?;
        
        if centrality.is_empty() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("betweenness_centrality", DataType::Float64, false),
            ]));
            
            return RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(Vec::<String>::new())),
                    Arc::new(Float64Array::from(Vec::<f64>::new())),
                ],
            ).map_err(GraphError::from);
        }
        
        // Sort by centrality score (descending)
        let mut sorted_nodes: Vec<(&String, &f64)> = centrality.iter().collect();
        sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
        let scores: Vec<f64> = sorted_nodes.iter().map(|(_, &score)| score).collect();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("betweenness_centrality", DataType::Float64, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(Float64Array::from(scores)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "betweenness_centrality"
    }
    
    fn description(&self) -> &'static str {
        "Calculate betweenness centrality using Brandes' algorithm"
    }
}

pub struct EigenvectorCentrality;

impl EigenvectorCentrality {
    /// Calculate eigenvector centrality using power iteration
    fn compute_eigenvector_centrality(
        &self,
        graph: &ArrowGraph,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<HashMap<String, f64>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(HashMap::new());
        }
        
        let node_ids: Vec<String> = graph.node_ids().cloned().collect();
        let mut centrality: HashMap<String, f64> = HashMap::new();
        let mut new_centrality: HashMap<String, f64> = HashMap::new();
        
        // Initialize with equal values
        let initial_value = 1.0 / (node_count as f64).sqrt();
        for node_id in &node_ids {
            centrality.insert(node_id.clone(), initial_value);
            new_centrality.insert(node_id.clone(), 0.0);
        }
        
        // Power iteration
        for iteration in 0..max_iterations {
            // Reset new centrality values
            for value in new_centrality.values_mut() {
                *value = 0.0;
            }
            
            // Compute new centrality values
            for node_id in &node_ids {
                let current_score = *centrality.get(node_id).unwrap_or(&0.0);
                
                if let Some(neighbors) = graph.neighbors(node_id) {
                    for neighbor in neighbors {
                        if let Some(neighbor_score) = new_centrality.get_mut(neighbor) {
                            *neighbor_score += current_score;
                        }
                    }
                }
            }
            
            // Normalize to prevent overflow
            let norm: f64 = new_centrality.values().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in new_centrality.values_mut() {
                    *value /= norm;
                }
            }
            
            // Check for convergence
            let mut diff = 0.0;
            for node_id in &node_ids {
                let old_score = *centrality.get(node_id).unwrap_or(&0.0);
                let new_score = *new_centrality.get(node_id).unwrap_or(&0.0);
                diff += (new_score - old_score).abs();
            }
            
            if diff < tolerance {
                log::debug!("Eigenvector centrality converged after {} iterations", iteration + 1);
                break;
            }
            
            // Swap for next iteration
            std::mem::swap(&mut centrality, &mut new_centrality);
        }
        
        Ok(centrality)
    }
}

impl GraphAlgorithm for EigenvectorCentrality {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let max_iterations: usize = params.get("max_iterations").unwrap_or(100);
        let tolerance: f64 = params.get("tolerance").unwrap_or(1e-6);
        
        // Validate parameters
        if max_iterations == 0 {
            return Err(GraphError::invalid_parameter(
                "max_iterations must be greater than 0"
            ));
        }
        
        if tolerance <= 0.0 {
            return Err(GraphError::invalid_parameter(
                "tolerance must be greater than 0.0"
            ));
        }
        
        let centrality = self.compute_eigenvector_centrality(graph, max_iterations, tolerance)?;
        
        if centrality.is_empty() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("eigenvector_centrality", DataType::Float64, false),
            ]));
            
            return RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(Vec::<String>::new())),
                    Arc::new(Float64Array::from(Vec::<f64>::new())),
                ],
            ).map_err(GraphError::from);
        }
        
        // Sort by centrality score (descending)
        let mut sorted_nodes: Vec<(&String, &f64)> = centrality.iter().collect();
        sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
        let scores: Vec<f64> = sorted_nodes.iter().map(|(_, &score)| score).collect();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("eigenvector_centrality", DataType::Float64, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(Float64Array::from(scores)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "eigenvector_centrality"
    }
    
    fn description(&self) -> &'static str {
        "Calculate eigenvector centrality using power iteration"
    }
}

pub struct ClosenessCentrality;

impl ClosenessCentrality {
    /// Calculate closeness centrality with batched distance calculations
    fn compute_closeness_centrality(&self, graph: &ArrowGraph) -> Result<HashMap<String, f64>> {
        let mut centrality: HashMap<String, f64> = HashMap::new();
        let node_count = graph.node_count();
        
        if node_count <= 1 {
            for node_id in graph.node_ids() {
                centrality.insert(node_id.clone(), 0.0);
            }
            return Ok(centrality);
        }
        
        // For each node, calculate shortest paths to all other nodes
        for source in graph.node_ids() {
            let distances = self.single_source_shortest_path_lengths(graph, source)?;
            
            // Calculate sum of distances and count reachable nodes
            let mut total_distance = 0.0;
            let mut reachable_count = 0;
            
            for (target, distance) in &distances {
                if target != source && *distance >= 0.0 {
                    total_distance += distance;
                    reachable_count += 1;
                }
            }
            
            // Calculate closeness centrality
            let closeness = if total_distance > 0.0 && reachable_count > 0 {
                let avg_distance = total_distance / reachable_count as f64;
                // Normalize by the fraction of nodes that are reachable
                let connectivity = reachable_count as f64 / (node_count - 1) as f64;
                connectivity / avg_distance
            } else {
                0.0
            };
            
            centrality.insert(source.clone(), closeness);
        }
        
        Ok(centrality)
    }
    
    /// Single-source shortest path lengths using BFS
    fn single_source_shortest_path_lengths(
        &self,
        graph: &ArrowGraph,
        source: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Initialize distances
        for node_id in graph.node_ids() {
            distances.insert(node_id.clone(), -1.0); // -1 means unreachable
        }
        
        // Start BFS from source
        distances.insert(source.to_string(), 0.0);
        queue.push_back(source.to_string());
        
        while let Some(current) = queue.pop_front() {
            let current_distance = *distances.get(&current).unwrap_or(&-1.0);
            
            if let Some(neighbors) = graph.neighbors(&current) {
                for neighbor in neighbors {
                    let neighbor_distance = *distances.get(neighbor).unwrap_or(&-1.0);
                    
                    // If neighbor not visited yet
                    if neighbor_distance < 0.0 {
                        distances.insert(neighbor.clone(), current_distance + 1.0);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        
        Ok(distances)
    }
}

impl GraphAlgorithm for ClosenessCentrality {
    fn execute(&self, graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        let centrality = self.compute_closeness_centrality(graph)?;
        
        if centrality.is_empty() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("closeness_centrality", DataType::Float64, false),
            ]));
            
            return RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(Vec::<String>::new())),
                    Arc::new(Float64Array::from(Vec::<f64>::new())),
                ],
            ).map_err(GraphError::from);
        }
        
        // Sort by centrality score (descending)
        let mut sorted_nodes: Vec<(&String, &f64)> = centrality.iter().collect();
        sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
        let scores: Vec<f64> = sorted_nodes.iter().map(|(_, &score)| score).collect();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("closeness_centrality", DataType::Float64, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(Float64Array::from(scores)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "closeness_centrality"
    }
    
    fn description(&self) -> &'static str {
        "Calculate closeness centrality using batched distance calculations"
    }
}