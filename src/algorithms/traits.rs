use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use crate::graph::ArrowGraph;
use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParams {
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl AlgorithmParams {
    pub fn new() -> Self {
        Self {
            params: serde_json::Map::new(),
        }
    }
    
    pub fn with_param<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.params.insert(
            key.to_string(),
            serde_json::to_value(value).unwrap(),
        );
        self
    }
    
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.params.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        Self::new()
    }
}

pub trait GraphAlgorithm {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch>;
    
    fn name(&self) -> &'static str;
    
    fn description(&self) -> &'static str {
        "Graph algorithm"
    }
}