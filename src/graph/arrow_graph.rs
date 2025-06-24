use arrow::record_batch::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::path::Path;
use crate::error::{GraphError, Result};
use crate::graph::GraphIndexes;

#[derive(Debug, Clone)]
pub struct ArrowGraph {
    pub nodes: RecordBatch,
    pub edges: RecordBatch,
    pub indexes: GraphIndexes,
}

impl ArrowGraph {
    /// Create a new graph from nodes and edges RecordBatches
    pub fn new(nodes: RecordBatch, edges: RecordBatch) -> Result<Self> {
        let indexes = GraphIndexes::build(&nodes, &edges)?;
        
        Ok(ArrowGraph {
            nodes,
            edges,
            indexes,
        })
    }
    
    /// Create a graph from just edges (nodes will be inferred)
    pub fn from_edges(edges: RecordBatch) -> Result<Self> {
        // Create an empty nodes RecordBatch with proper schema
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        
        let empty_nodes = RecordBatch::new_empty(nodes_schema);
        Self::new(empty_nodes, edges)
    }

    /// Create an empty graph with no nodes or edges
    pub fn empty() -> Result<Self> {
        // Create empty nodes RecordBatch
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let empty_nodes = RecordBatch::new_empty(nodes_schema);

        // Create empty edges RecordBatch
        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));
        let empty_edges = RecordBatch::new_empty(edges_schema);

        Self::new(empty_nodes, empty_edges)
    }
    
    /// Load graph from Arrow/Parquet files
    pub async fn from_files<P: AsRef<Path>>(
        _nodes_path: P,
        _edges_path: P,
    ) -> Result<Self> {
        todo!("Implement loading from files - will read Arrow/Parquet files")
    }
    
    /// Create graph from RecordBatches (alias for new)
    pub fn from_tables(
        nodes: RecordBatch,
        edges: RecordBatch,
    ) -> Result<Self> {
        Self::new(nodes, edges)
    }
    
    /// Execute SQL query with graph functions
    pub async fn sql(&self, _query: &str) -> Result<RecordBatch> {
        todo!("Implement SQL execution using DataFusion with graph functions")
    }
    
    /// Get number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.indexes.node_count
    }
    
    /// Get number of edges in the graph  
    pub fn edge_count(&self) -> usize {
        self.indexes.edge_count
    }
    
    /// Calculate graph density (edges / max_possible_edges)
    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        let m = self.edge_count() as f64;
        
        if n <= 1.0 {
            0.0
        } else {
            m / (n * (n - 1.0))
        }
    }
    
    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: &str) -> Option<&Vec<String>> {
        self.indexes.neighbors(node_id)
    }
    
    /// Get predecessors of a node (incoming edges)
    pub fn predecessors(&self, node_id: &str) -> Option<&Vec<String>> {
        self.indexes.predecessors(node_id)
    }
    
    /// Check if node exists in graph
    pub fn has_node(&self, node_id: &str) -> bool {
        self.indexes.has_node(node_id)
    }
    
    /// Get edge weight between two nodes
    pub fn edge_weight(&self, source: &str, target: &str) -> Option<f64> {
        self.indexes.edge_weight(source, target)
    }
    
    /// Get all node IDs
    pub fn node_ids(&self) -> impl Iterator<Item = &String> {
        self.indexes.all_nodes()
    }
    
    /// Add a new node to the graph
    pub fn add_node(&mut self, node_id: String) -> Result<()> {
        // Check if node already exists
        if self.has_node(&node_id) {
            return Err(GraphError::invalid_parameter(
                &format!("Node '{}' already exists in the graph", node_id)
            ));
        }
        
        // Create new nodes RecordBatch with the added node
        let mut node_ids: Vec<String> = self.node_ids().cloned().collect();
        node_ids.push(node_id);
        
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        
        let new_nodes = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(arrow::array::StringArray::from(node_ids))],
        ).map_err(GraphError::from)?;
        
        // Rebuild graph with new nodes
        self.nodes = new_nodes;
        self.indexes = GraphIndexes::build(&self.nodes, &self.edges)?;
        
        Ok(())
    }
    
    /// Remove a node from the graph (also removes all incident edges)
    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        // Check if node exists
        if !self.has_node(node_id) {
            return Err(GraphError::invalid_parameter(
                &format!("Node '{}' does not exist in the graph", node_id)
            ));
        }
        
        // Filter out the node
        let remaining_nodes: Vec<String> = self.node_ids()
            .filter(|&id| id != node_id)
            .cloned()
            .collect();
        
        // Create new nodes RecordBatch
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        
        let new_nodes = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(arrow::array::StringArray::from(remaining_nodes))],
        ).map_err(GraphError::from)?;
        
        // Filter out edges that involve the removed node
        let source_array = self.edges.column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid source column type"))?;
            
        let target_array = self.edges.column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid target column type"))?;
        
        let mut remaining_sources = Vec::new();
        let mut remaining_targets = Vec::new();
        let mut remaining_weights = Vec::new();
        
        for i in 0..self.edges.num_rows() {
            let source = source_array.value(i);
            let target = target_array.value(i);
            
            // Keep edge only if neither source nor target is the removed node
            if source != node_id && target != node_id {
                remaining_sources.push(source.to_string());
                remaining_targets.push(target.to_string());
                
                // Handle optional weight column
                if self.edges.num_columns() > 2 {
                    if let Some(weight_array) = self.edges.column(2)
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                    {
                        remaining_weights.push(weight_array.value(i));
                    }
                }
            }
        }
        
        // Create new edges RecordBatch
        let mut edge_fields = vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
        ];
        
        let mut edge_columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(arrow::array::StringArray::from(remaining_sources)),
            Arc::new(arrow::array::StringArray::from(remaining_targets)),
        ];
        
        if !remaining_weights.is_empty() {
            edge_fields.push(Field::new("weight", DataType::Float64, true));
            edge_columns.push(Arc::new(arrow::array::Float64Array::from(remaining_weights)));
        }
        
        let edges_schema = Arc::new(Schema::new(edge_fields));
        let new_edges = RecordBatch::try_new(edges_schema, edge_columns)
            .map_err(GraphError::from)?;
        
        // Rebuild graph
        self.nodes = new_nodes;
        self.edges = new_edges;
        self.indexes = GraphIndexes::build(&self.nodes, &self.edges)?;
        
        Ok(())
    }
    
    /// Add an edge to the graph
    pub fn add_edge(&mut self, source: String, target: String, weight: Option<f64>) -> Result<()> {
        // Ensure both nodes exist (add them if they don't)
        if !self.has_node(&source) {
            self.add_node(source.clone())?;
        }
        if !self.has_node(&target) {
            self.add_node(target.clone())?;
        }
        
        // Check if edge already exists
        if self.edge_weight(&source, &target).is_some() {
            return Err(GraphError::invalid_parameter(
                &format!("Edge from '{}' to '{}' already exists", source, target)
            ));
        }
        
        // Get existing edges
        let source_array = self.edges.column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid source column type"))?;
            
        let target_array = self.edges.column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid target column type"))?;
        
        let mut new_sources: Vec<String> = source_array.iter()
            .map(|s| s.unwrap_or("").to_string())
            .collect();
        let mut new_targets: Vec<String> = target_array.iter()
            .map(|t| t.unwrap_or("").to_string())
            .collect();
        
        // Add new edge
        new_sources.push(source);
        new_targets.push(target);
        
        let mut edge_fields = vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
        ];
        
        let mut edge_columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(arrow::array::StringArray::from(new_sources)),
            Arc::new(arrow::array::StringArray::from(new_targets)),
        ];
        
        // Handle weights
        if self.edges.num_columns() > 2 || weight.is_some() {
            let mut new_weights = Vec::new();
            
            // Copy existing weights
            if self.edges.num_columns() > 2 {
                if let Some(weight_array) = self.edges.column(2)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                {
                    for i in 0..weight_array.len() {
                        new_weights.push(Some(weight_array.value(i)));
                    }
                }
            } else {
                // Fill with None for existing edges if this is the first weighted edge
                for _ in 0..self.edges.num_rows() {
                    new_weights.push(None);
                }
            }
            
            // Add weight for new edge
            new_weights.push(weight);
            
            edge_fields.push(Field::new("weight", DataType::Float64, true));
            edge_columns.push(Arc::new(arrow::array::Float64Array::from(new_weights)));
        }
        
        let edges_schema = Arc::new(Schema::new(edge_fields));
        let new_edges = RecordBatch::try_new(edges_schema, edge_columns)
            .map_err(GraphError::from)?;
        
        // Rebuild graph
        self.edges = new_edges;
        self.indexes = GraphIndexes::build(&self.nodes, &self.edges)?;
        
        Ok(())
    }
    
    /// Remove an edge from the graph
    pub fn remove_edge(&mut self, source: &str, target: &str) -> Result<()> {
        // Check if edge exists
        if self.edge_weight(source, target).is_none() {
            return Err(GraphError::invalid_parameter(
                &format!("Edge from '{}' to '{}' does not exist", source, target)
            ));
        }
        
        // Filter out the edge
        let source_array = self.edges.column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid source column type"))?;
            
        let target_array = self.edges.column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| GraphError::invalid_parameter("Invalid target column type"))?;
        
        let mut remaining_sources = Vec::new();
        let mut remaining_targets = Vec::new();
        let mut remaining_weights = Vec::new();
        
        for i in 0..self.edges.num_rows() {
            let edge_source = source_array.value(i);
            let edge_target = target_array.value(i);
            
            // Keep edge only if it's not the one we want to remove
            if !(edge_source == source && edge_target == target) {
                remaining_sources.push(edge_source.to_string());
                remaining_targets.push(edge_target.to_string());
                
                // Handle optional weight column
                if self.edges.num_columns() > 2 {
                    if let Some(weight_array) = self.edges.column(2)
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                    {
                        remaining_weights.push(Some(weight_array.value(i)));
                    }
                }
            }
        }
        
        // Create new edges RecordBatch
        let mut edge_fields = vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
        ];
        
        let mut edge_columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(arrow::array::StringArray::from(remaining_sources)),
            Arc::new(arrow::array::StringArray::from(remaining_targets)),
        ];
        
        if !remaining_weights.is_empty() {
            edge_fields.push(Field::new("weight", DataType::Float64, true));
            edge_columns.push(Arc::new(arrow::array::Float64Array::from(remaining_weights)));
        }
        
        let edges_schema = Arc::new(Schema::new(edge_fields));
        let new_edges = RecordBatch::try_new(edges_schema, edge_columns)
            .map_err(GraphError::from)?;
        
        // Rebuild graph
        self.edges = new_edges;
        self.indexes = GraphIndexes::build(&self.nodes, &self.edges)?;
        
        Ok(())
    }
}