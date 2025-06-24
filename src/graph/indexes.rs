use arrow::record_batch::RecordBatch;
use arrow::array::{Array, StringArray, Float64Array};
use hashbrown::HashMap;
use crate::error::{GraphError, Result};

#[derive(Debug, Clone)]
pub struct GraphIndexes {
    pub adjacency_list: HashMap<String, Vec<String>>,
    pub reverse_adjacency_list: HashMap<String, Vec<String>>,
    pub node_index: HashMap<String, usize>,
    pub edge_weights: HashMap<(String, String), f64>,
    pub node_count: usize,
    pub edge_count: usize,
}

impl GraphIndexes {
    pub fn build(nodes: &RecordBatch, edges: &RecordBatch) -> Result<Self> {
        let mut adjacency_list: HashMap<String, Vec<String>> = HashMap::new();
        let mut reverse_adjacency_list: HashMap<String, Vec<String>> = HashMap::new();
        let mut node_index: HashMap<String, usize> = HashMap::new();
        let mut edge_weights: HashMap<(String, String), f64> = HashMap::new();
        
        // Build node index from nodes RecordBatch
        // Expected schema: id (String), [label (String)], [properties (JSON)]
        if nodes.num_columns() > 0 {
            let id_column = nodes.column(0);
            if let Some(string_array) = id_column.as_any().downcast_ref::<StringArray>() {
                for (idx, node_id_opt) in string_array.iter().enumerate() {
                    if let Some(node_id) = node_id_opt {
                        node_index.insert(node_id.to_string(), idx);
                        adjacency_list.insert(node_id.to_string(), Vec::new());
                        reverse_adjacency_list.insert(node_id.to_string(), Vec::new());
                    }
                }
            } else {
                return Err(GraphError::graph_construction(
                    "First column of nodes table must be String (node ID)"
                ));
            }
        }
        
        // Build adjacency lists from edges RecordBatch
        // Expected schema: source (String), target (String), [weight (Float64)], [label (String)]
        if edges.num_columns() >= 2 {
            let source_column = edges.column(0);
            let target_column = edges.column(1);
            
            let source_array = source_column.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| GraphError::graph_construction(
                    "First column of edges table must be String (source)"
                ))?;
            
            let target_array = target_column.as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| GraphError::graph_construction(
                    "Second column of edges table must be String (target)"
                ))?;
            
            // Handle optional weight column
            let weight_array = if edges.num_columns() >= 3 {
                edges.column(2).as_any().downcast_ref::<Float64Array>()
            } else {
                None
            };
            
            for i in 0..edges.num_rows() {
                let source_opt = source_array.value(i);
                let target_opt = target_array.value(i);
                let (source, target) = (source_opt, target_opt);
                let source_str = source.to_string();
                let target_str = target.to_string();
                
                // Add to adjacency lists (create nodes if they don't exist)
                adjacency_list.entry(source_str.clone())
                    .or_default()
                    .push(target_str.clone());
                
                reverse_adjacency_list.entry(target_str.clone())
                    .or_default()
                    .push(source_str.clone());
                
                // Add to node index if not exists
                if !node_index.contains_key(&source_str) {
                    let idx = node_index.len();
                    node_index.insert(source_str.clone(), idx);
                }
                if !node_index.contains_key(&target_str) {
                    let idx = node_index.len();
                    node_index.insert(target_str.clone(), idx);
                }
                
                // Handle edge weights
                let weight = if let Some(weights) = weight_array {
                    weights.value(i)
                } else {
                    1.0 // Default weight
                };
                
                edge_weights.insert((source_str, target_str), weight);
            }
        }
        
        let node_count = node_index.len();
        let edge_count = edges.num_rows();
        
        Ok(GraphIndexes {
            adjacency_list,
            reverse_adjacency_list,
            node_index,
            edge_weights,
            node_count,
            edge_count,
        })
    }
    
    pub fn neighbors(&self, node_id: &str) -> Option<&Vec<String>> {
        self.adjacency_list.get(node_id)
    }
    
    pub fn predecessors(&self, node_id: &str) -> Option<&Vec<String>> {
        self.reverse_adjacency_list.get(node_id)
    }
    
    pub fn has_node(&self, node_id: &str) -> bool {
        self.node_index.contains_key(node_id)
    }
    
    pub fn edge_weight(&self, source: &str, target: &str) -> Option<f64> {
        self.edge_weights.get(&(source.to_string(), target.to_string())).copied()
    }
    
    pub fn all_nodes(&self) -> impl Iterator<Item = &String> {
        self.node_index.keys()
    }
}