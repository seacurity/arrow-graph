use arrow::record_batch::RecordBatch;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::Result;

pub struct BetweennessCentrality;

impl GraphAlgorithm for BetweennessCentrality {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Implement betweenness centrality calculation")
    }
    
    fn name(&self) -> &'static str {
        "betweenness_centrality"
    }
    
    fn description(&self) -> &'static str {
        "Calculate betweenness centrality for graph nodes"
    }
}

pub struct PageRank;

impl GraphAlgorithm for PageRank {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Implement PageRank algorithm")
    }
    
    fn name(&self) -> &'static str {
        "pagerank"
    }
    
    fn description(&self) -> &'static str {
        "Calculate PageRank scores for graph nodes"
    }
}