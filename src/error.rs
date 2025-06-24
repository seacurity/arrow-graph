use std::fmt;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion_common::DataFusionError),
    
    #[error("Graph construction error: {0}")]
    GraphConstruction(String),
    
    #[error("Algorithm error: {0}")]
    Algorithm(String),
    
    #[error("SQL parsing error: {0}")]
    SqlParsing(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Edge not found: source={0}, target={1}")]
    EdgeNotFound(String, String),
    
    #[error("Graph is empty")]
    EmptyGraph,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl GraphError {
    pub fn graph_construction<S: Into<String>>(msg: S) -> Self {
        GraphError::GraphConstruction(msg.into())
    }
    
    pub fn algorithm<S: Into<String>>(msg: S) -> Self {
        GraphError::Algorithm(msg.into())
    }
    
    pub fn sql_parsing<S: Into<String>>(msg: S) -> Self {
        GraphError::SqlParsing(msg.into())
    }
    
    pub fn invalid_parameter<S: Into<String>>(msg: S) -> Self {
        GraphError::InvalidParameter(msg.into())
    }
    
    pub fn node_not_found<S: Into<String>>(node_id: S) -> Self {
        GraphError::NodeNotFound(node_id.into())
    }
}