use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::collections::HashMap;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

pub struct GraphDensity;

impl GraphAlgorithm for GraphDensity {
    fn execute(&self, _graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Calculate graph density metric")
    }
    
    fn name(&self) -> &'static str {
        "graph_density"
    }
    
    fn description(&self) -> &'static str {
        "Calculate the density of the graph"
    }
}

pub struct TriangleCount;

impl TriangleCount {
    /// Count triangles in the graph using node enumeration method
    fn count_triangles(&self, graph: &ArrowGraph) -> Result<u64> {
        let mut triangle_count = 0u64;
        let node_ids: Vec<String> = graph.node_ids().cloned().collect();
        
        // For each triple of nodes, check if they form a triangle
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                for k in (j + 1)..node_ids.len() {
                    let node_a = &node_ids[i];
                    let node_b = &node_ids[j];
                    let node_c = &node_ids[k];
                    
                    // Check if all three edges exist (undirected): A-B, B-C, C-A
                    let has_ab = (graph.neighbors(node_a)
                        .map(|neighbors| neighbors.contains(node_b))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_b)
                        .map(|neighbors| neighbors.contains(node_a))
                        .unwrap_or(false));
                    
                    let has_bc = (graph.neighbors(node_b)
                        .map(|neighbors| neighbors.contains(node_c))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_c)
                        .map(|neighbors| neighbors.contains(node_b))
                        .unwrap_or(false));
                    
                    let has_ac = (graph.neighbors(node_a)
                        .map(|neighbors| neighbors.contains(node_c))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_c)
                        .map(|neighbors| neighbors.contains(node_a))
                        .unwrap_or(false));
                    
                    if has_ab && has_bc && has_ac {
                        triangle_count += 1;
                    }
                }
            }
        }
        
        Ok(triangle_count)
    }
    
    /// Count triangles per node for local clustering coefficient
    fn count_triangles_per_node(&self, graph: &ArrowGraph) -> Result<HashMap<String, u64>> {
        let mut node_triangles: HashMap<String, u64> = HashMap::new();
        let node_ids: Vec<String> = graph.node_ids().cloned().collect();
        
        // Initialize all nodes with 0 triangles
        for node_id in graph.node_ids() {
            node_triangles.insert(node_id.clone(), 0);
        }
        
        // Find all triangles and count them for each participating node
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                for k in (j + 1)..node_ids.len() {
                    let node_a = &node_ids[i];
                    let node_b = &node_ids[j];
                    let node_c = &node_ids[k];
                    
                    // Check if all three edges exist (undirected): A-B, B-C, C-A
                    let has_ab = (graph.neighbors(node_a)
                        .map(|neighbors| neighbors.contains(node_b))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_b)
                        .map(|neighbors| neighbors.contains(node_a))
                        .unwrap_or(false));
                    
                    let has_bc = (graph.neighbors(node_b)
                        .map(|neighbors| neighbors.contains(node_c))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_c)
                        .map(|neighbors| neighbors.contains(node_b))
                        .unwrap_or(false));
                    
                    let has_ac = (graph.neighbors(node_a)
                        .map(|neighbors| neighbors.contains(node_c))
                        .unwrap_or(false)) ||
                        (graph.neighbors(node_c)
                        .map(|neighbors| neighbors.contains(node_a))
                        .unwrap_or(false));
                    
                    if has_ab && has_bc && has_ac {
                        // Triangle found, increment count for all three nodes
                        *node_triangles.get_mut(node_a).unwrap() += 1;
                        *node_triangles.get_mut(node_b).unwrap() += 1;
                        *node_triangles.get_mut(node_c).unwrap() += 1;
                    }
                }
            }
        }
        
        Ok(node_triangles)
    }
}

impl GraphAlgorithm for TriangleCount {
    fn execute(&self, graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        let total_triangles = self.count_triangles(graph)?;
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::UInt64, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["triangle_count"])),
                Arc::new(UInt64Array::from(vec![total_triangles])),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "triangle_count"
    }
    
    fn description(&self) -> &'static str {
        "Count the total number of triangles in the graph"
    }
}

pub struct ClusteringCoefficient;

impl ClusteringCoefficient {
    /// Calculate local clustering coefficient for each node
    fn calculate_local_clustering(&self, graph: &ArrowGraph) -> Result<HashMap<String, f64>> {
        let mut clustering: HashMap<String, f64> = HashMap::new();
        let triangle_counter = TriangleCount;
        let node_triangles = triangle_counter.count_triangles_per_node(graph)?;
        
        for node_id in graph.node_ids() {
            if let Some(neighbors) = graph.neighbors(node_id) {
                let degree = neighbors.len();
                
                if degree < 2 {
                    // Nodes with degree < 2 cannot form triangles
                    clustering.insert(node_id.clone(), 0.0);
                } else {
                    let triangles = *node_triangles.get(node_id).unwrap_or(&0);
                    let possible_triangles = (degree * (degree - 1)) / 2;
                    let coefficient = triangles as f64 / possible_triangles as f64;
                    clustering.insert(node_id.clone(), coefficient);
                }
            } else {
                clustering.insert(node_id.clone(), 0.0);
            }
        }
        
        Ok(clustering)
    }
    
    /// Calculate global clustering coefficient (transitivity)
    fn calculate_global_clustering(&self, graph: &ArrowGraph) -> Result<f64> {
        let triangle_counter = TriangleCount;
        let total_triangles = triangle_counter.count_triangles(graph)? as f64;
        
        // Count total number of connected triples (paths of length 2)
        let mut total_triples = 0u64;
        
        for node_id in graph.node_ids() {
            if let Some(neighbors) = graph.neighbors(node_id) {
                let degree = neighbors.len();
                if degree >= 2 {
                    // Number of connected triples centered at this node
                    // Each pair of neighbors forms a triple with this node as center
                    total_triples += (degree * (degree - 1)) as u64 / 2;
                }
            }
        }
        
        if total_triples == 0 {
            Ok(0.0)
        } else {
            // Global clustering coefficient = 3 * triangles / triples
            // Note: Each triangle contributes 3 triples, so we multiply by 3
            let coefficient = 3.0 * total_triangles / total_triples as f64;
            // Ensure coefficient is within valid range [0, 1]
            Ok(coefficient.min(1.0))
        }
    }
}

impl GraphAlgorithm for ClusteringCoefficient {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        let mode: String = params.get("mode").unwrap_or("local".to_string());
        
        match mode.as_str() {
            "local" => {
                let clustering = self.calculate_local_clustering(graph)?;
                
                if clustering.is_empty() {
                    let schema = Arc::new(Schema::new(vec![
                        Field::new("node_id", DataType::Utf8, false),
                        Field::new("clustering_coefficient", DataType::Float64, false),
                    ]));
                    
                    return RecordBatch::try_new(
                        schema,
                        vec![
                            Arc::new(StringArray::from(Vec::<String>::new())),
                            Arc::new(Float64Array::from(Vec::<f64>::new())),
                        ],
                    ).map_err(GraphError::from);
                }
                
                // Sort by clustering coefficient (descending)
                let mut sorted_nodes: Vec<(&String, &f64)> = clustering.iter().collect();
                sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
                let coefficients: Vec<f64> = sorted_nodes.iter().map(|(_, &coeff)| coeff).collect();
                
                let schema = Arc::new(Schema::new(vec![
                    Field::new("node_id", DataType::Utf8, false),
                    Field::new("clustering_coefficient", DataType::Float64, false),
                ]));
                
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(StringArray::from(node_ids)),
                        Arc::new(Float64Array::from(coefficients)),
                    ],
                ).map_err(GraphError::from)
            },
            "global" => {
                let global_coefficient = self.calculate_global_clustering(graph)?;
                
                let schema = Arc::new(Schema::new(vec![
                    Field::new("metric", DataType::Utf8, false),
                    Field::new("value", DataType::Float64, false),
                ]));
                
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(StringArray::from(vec!["global_clustering_coefficient"])),
                        Arc::new(Float64Array::from(vec![global_coefficient])),
                    ],
                ).map_err(GraphError::from)
            },
            _ => Err(GraphError::invalid_parameter(
                "mode must be 'local' or 'global'"
            ))
        }
    }
    
    fn name(&self) -> &'static str {
        "clustering_coefficient"
    }
    
    fn description(&self) -> &'static str {
        "Calculate local or global clustering coefficient"
    }
}