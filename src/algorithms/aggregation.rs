use arrow::record_batch::RecordBatch;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::Result;

pub struct GraphDensity;

impl GraphAlgorithm for GraphDensity {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Calculate graph density metric")
    }
    
    fn name(&self) -> &'static str {
        "graph_density"
    }
    
    fn description(&self) -> &'static str {
        "Calculate the density of the graph"
    }
}

pub struct ClusteringCoefficient;

impl GraphAlgorithm for ClusteringCoefficient {
    fn execute(&self, graph: &ArrowGraph, params: &AlgorithmParams) -> Result<RecordBatch> {
        todo!("Calculate clustering coefficient")
    }
    
    fn name(&self) -> &'static str {
        "clustering_coefficient"
    }
    
    fn description(&self) -> &'static str {
        "Calculate clustering coefficient for the graph"
    }
}