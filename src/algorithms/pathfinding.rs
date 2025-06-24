use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, Float64Array, UInt32Array, ListArray, ArrayRef};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::array::builder::{ListBuilder, StringBuilder};
use std::sync::Arc;
use std::collections::{HashMap, BinaryHeap, VecDeque};
use std::cmp::Ordering;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

#[derive(Debug, Clone, PartialEq)]
struct DijkstraNode {
    node_id: String,
    distance: f64,
    previous: Option<String>,
}

impl Eq for DijkstraNode {}

impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ShortestPath;

impl ShortestPath {
    /// Dijkstra's algorithm implementation optimized for Arrow
    fn dijkstra(
        &self,
        graph: &ArrowGraph,
        source: &str,
        target: Option<&str>,
    ) -> Result<HashMap<String, (f64, Option<String>)>> {
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, Option<String>> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        // Initialize distances
        for node_id in graph.node_ids() {
            let dist = if node_id == source { 0.0 } else { f64::INFINITY };
            distances.insert(node_id.clone(), dist);
            previous.insert(node_id.clone(), None);
        }
        
        heap.push(DijkstraNode {
            node_id: source.to_string(),
            distance: 0.0,
            previous: None,
        });
        
        while let Some(current) = heap.pop() {
            // Early termination if we're looking for a specific target
            if let Some(target_node) = target {
                if current.node_id == target_node {
                    break;
                }
            }
            
            // Skip if we've already found a better path
            if current.distance > *distances.get(&current.node_id).unwrap_or(&f64::INFINITY) {
                continue;
            }
            
            // Check neighbors
            if let Some(neighbors) = graph.neighbors(&current.node_id) {
                for neighbor in neighbors {
                    let edge_weight = graph.edge_weight(&current.node_id, neighbor).unwrap_or(1.0);
                    let new_distance = current.distance + edge_weight;
                    
                    if new_distance < *distances.get(neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor.clone(), new_distance);
                        previous.insert(neighbor.clone(), Some(current.node_id.clone()));
                        
                        heap.push(DijkstraNode {
                            node_id: neighbor.clone(),
                            distance: new_distance,
                            previous: Some(current.node_id.clone()),
                        });
                    }
                }
            }
        }
        
        // Combine results
        let mut result = HashMap::new();
        for node_id in graph.node_ids() {
            let dist = *distances.get(node_id).unwrap_or(&f64::INFINITY);
            let prev = previous.get(node_id).cloned().flatten();
            result.insert(node_id.clone(), (dist, prev));
        }
        
        Ok(result)
    }
    
    /// Reconstruct path from source to target
    fn reconstruct_path(
        &self,
        target: &str,
        previous: &HashMap<String, Option<String>>,
    ) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = Some(target.to_string());
        
        while let Some(node) = current {
            path.push(node.clone());
            current = previous.get(&node).cloned().flatten();
        }
        
        path.reverse();
        path
    }
}

impl GraphAlgorithm for ShortestPath {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let source: String = params.get("source")
            .ok_or_else(|| GraphError::invalid_parameter("source parameter required"))?;
        
        let target: Option<String> = params.get("target");
        
        match target {
            Some(target_node) => {
                // Single source-target shortest path
                let results = self.dijkstra(graph, &source, Some(&target_node))?;
                
                if let Some((distance, _)) = results.get(&target_node) {
                    if distance.is_infinite() {
                        return Err(GraphError::algorithm("No path found between source and target"));
                    }
                    
                    let path = self.reconstruct_path(&target_node, &results.iter()
                        .map(|(k, (_, prev))| (k.clone(), prev.clone()))
                        .collect());
                    
                    // Create Arrow RecordBatch with path and distance
                    let schema = Arc::new(Schema::new(vec![
                        Field::new("source", DataType::Utf8, false),
                        Field::new("target", DataType::Utf8, false),
                        Field::new("distance", DataType::Float64, false),
                        Field::new("path", DataType::List(
                            Arc::new(Field::new("item", DataType::Utf8, true))
                        ), false),
                    ]));
                    
                    // Build path array using ListBuilder
                    let mut list_builder = ListBuilder::new(StringBuilder::new());
                    for node in &path {
                        list_builder.values().append_value(node);
                    }
                    list_builder.append(true);
                    let path_array = list_builder.finish();
                    
                    RecordBatch::try_new(
                        schema,
                        vec![
                            Arc::new(StringArray::from(vec![source])),
                            Arc::new(StringArray::from(vec![target_node])),
                            Arc::new(Float64Array::from(vec![*distance])),
                            Arc::new(path_array),
                        ],
                    ).map_err(GraphError::from)
                } else {
                    Err(GraphError::node_not_found(target_node))
                }
            }
            None => {
                // Single source shortest paths to all nodes
                let results = self.dijkstra(graph, &source, None)?;
                
                let mut targets = Vec::new();
                let mut distances = Vec::new();
                
                for (node_id, (distance, _)) in results.iter() {
                    if node_id != &source && !distance.is_infinite() {
                        targets.push(node_id.clone());
                        distances.push(*distance);
                    }
                }
                
                let schema = Arc::new(Schema::new(vec![
                    Field::new("source", DataType::Utf8, false),
                    Field::new("target", DataType::Utf8, false),
                    Field::new("distance", DataType::Float64, false),
                ]));
                
                let sources = vec![source; targets.len()];
                
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(StringArray::from(sources)),
                        Arc::new(StringArray::from(targets)),
                        Arc::new(Float64Array::from(distances)),
                    ],
                ).map_err(GraphError::from)
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "shortest_path"
    }
    
    fn description(&self) -> &'static str {
        "Find the shortest path between nodes using Dijkstra's algorithm"
    }
}

pub struct AllPaths;

impl AllPaths {
    /// BFS-based all paths implementation with hop limit
    fn find_all_paths(
        &self,
        graph: &ArrowGraph,
        source: &str,
        target: &str,
        max_hops: usize,
    ) -> Result<Vec<Vec<String>>> {
        let mut all_paths = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with the source node
        queue.push_back((vec![source.to_string()], 0));
        
        while let Some((current_path, hops)) = queue.pop_front() {
            let current_node = current_path.last().unwrap();
            
            if current_node == target {
                all_paths.push(current_path);
                continue;
            }
            
            if hops >= max_hops {
                continue;
            }
            
            if let Some(neighbors) = graph.neighbors(current_node) {
                for neighbor in neighbors {
                    // Avoid cycles
                    if !current_path.contains(neighbor) {
                        let mut new_path = current_path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back((new_path, hops + 1));
                    }
                }
            }
        }
        
        Ok(all_paths)
    }
}

impl GraphAlgorithm for AllPaths {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let source: String = params.get("source")
            .ok_or_else(|| GraphError::invalid_parameter("source parameter required"))?;
        
        let target: String = params.get("target")
            .ok_or_else(|| GraphError::invalid_parameter("target parameter required"))?;
        
        let max_hops: usize = params.get("max_hops").unwrap_or(10);
        
        let paths = self.find_all_paths(graph, &source, &target, max_hops)?;
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("path_length", DataType::UInt32, false),
            Field::new("path", DataType::List(
                Arc::new(Field::new("item", DataType::Utf8, true))
            ), false),
        ]));
        
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut path_lengths = Vec::new();
        let mut path_arrays = Vec::new();
        
        for path in paths {
            sources.push(source.clone());
            targets.push(target.clone());
            path_lengths.push(path.len() as u32 - 1); // Number of edges
            
            let path_values: Vec<Option<String>> = path.into_iter().map(Some).collect();
            path_arrays.push(Some(path_values));
        }
        
        // Build all paths using ListBuilder
        let mut list_builder = ListBuilder::new(StringBuilder::new());
        for path_values in path_arrays {
            if let Some(path) = path_values {
                for node in path {
                    if let Some(node_str) = node {
                        list_builder.values().append_value(&node_str);
                    }
                }
                list_builder.append(true);
            } else {
                list_builder.append(false);
            }
        }
        let list_array = list_builder.finish();
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(sources)),
                Arc::new(StringArray::from(targets)),
                Arc::new(UInt32Array::from(path_lengths)),
                Arc::new(list_array),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "all_paths"
    }
    
    fn description(&self) -> &'static str {
        "Find all paths between two nodes with optional hop limit"
    }
}