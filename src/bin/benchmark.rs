use arrow_graph::prelude::*;
use arrow_graph::performance::{run_all_benchmarks, print_results};
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Arrow Graph v0.7.0 Performance Benchmark Suite");
    println!("==============================================");

    // Create a test graph for benchmarking
    let start = Instant::now();
    let graph = create_benchmark_graph(1000, 5000)?;
    let construction_time = start.elapsed();
    
    println!("Created test graph with {} nodes and {} edges in {:?}", 
             graph.node_count(), graph.edge_count(), construction_time);
    println!();

    // Run all benchmarks
    let results = run_all_benchmarks(&graph)?;
    
    // Print results
    print_results(&results);

    // Summary statistics
    let total_avg_time: f64 = results.iter()
        .map(|r| r.avg_duration.as_secs_f64())
        .sum();
    
    let total_throughput: f64 = results.iter()
        .map(|r| r.throughput_ops_per_sec)
        .sum();

    println!("\nSUMMARY:");
    println!("Total average benchmark time: {:.3}s", total_avg_time);
    println!("Combined throughput: {:.0} ops/s", total_throughput);
    println!("Graph construction rate: {:.0} nodes/s", 
             graph.node_count() as f64 / construction_time.as_secs_f64());

    Ok(())
}

fn create_benchmark_graph(num_nodes: usize, num_edges: usize) -> std::result::Result<ArrowGraph, Box<dyn std::error::Error>> {
    use arrow::array::{StringArray, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use rand::prelude::*;
    
    let mut rng = rand::thread_rng();
    
    // Create node schema and data
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
    ]));
    
    let node_ids: Vec<String> = (0..num_nodes).map(|i| format!("node_{}", i)).collect();
    
    let nodes_batch = RecordBatch::try_new(
        node_schema,
        vec![Arc::new(StringArray::from(node_ids.clone()))],
    )?;
    
    // Create edge schema and data
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
    ]));
    
    let mut edge_sources = Vec::new();
    let mut edge_targets = Vec::new();
    
    // Generate random edges
    for _ in 0..num_edges {
        let from_idx = rng.gen_range(0..num_nodes);
        let to_idx = rng.gen_range(0..num_nodes);
        
        if from_idx != to_idx {
            edge_sources.push(format!("node_{}", from_idx));
            edge_targets.push(format!("node_{}", to_idx));
        }
    }
    
    let edges_batch = RecordBatch::try_new(
        edge_schema,
        vec![
            Arc::new(StringArray::from(edge_sources)),
            Arc::new(StringArray::from(edge_targets)),
        ],
    )?;
    
    Ok(ArrowGraph::new(nodes_batch, edges_batch)?)
}