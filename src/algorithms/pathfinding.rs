use arrow::record_batch::RecordBatch;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::Result;

pub struct ShortestPath;

impl GraphAlgorithm for ShortestPath {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Implement shortest path using Dijkstra or BFS algorithm")
    }
    
    fn name(&self) -> &'static str {
        "shortest_path"
    }
    
    fn description(&self) -> &'static str {
        "Find the shortest path between two nodes"
    }
}

pub struct AllPaths;

impl GraphAlgorithm for AllPaths {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Implement all paths finding with max hops constraint")
    }
    
    fn name(&self) -> &'static str {
        "all_paths"
    }
    
    fn description(&self) -> &'static str {
        "Find all paths between two nodes with optional hop limit"
    }
}