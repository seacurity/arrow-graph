# Arrow-Graph

A high-performance, Arrow-native graph analytics engine with SQL interface for modern data processing workflows.

## Overview

Arrow-Graph brings graph analytics to the Apache Arrow ecosystem, providing:

- **Arrow-native storage** - Zero-copy graph operations using columnar data
- **SQL-first interface** - Query graphs using familiar SQL syntax
- **Modern algorithms** - SIMD-optimized implementations of graph algorithms  
- **DataFusion integration** - Seamless integration with existing analytical workflows
- **Streaming support** - Process large graphs with bounded memory
- **ML-ready** - Built for graph neural networks and feature extraction

### Graph Model

Arrow-Graph represents graphs as Arrow RecordBatches, enabling zero-copy operations:

```
Edges Table (RecordBatch):
┌───────┬────────┬────────┐
│ src   │ dst    │ weight │
├───────┼────────┼────────┤
│ "A"   │ "B"    │ 1.0    │
│ "B"   │ "C"    │ 2.0    │
│ "C"   │ "D"    │ 1.5    │
│ "A"   │ "D"    │ 3.0    │
└───────┴────────┴────────┘

Nodes: Automatically indexed from unique src/dst values
Internal ID mapping: "A"→0, "B"→1, "C"→2, "D"→3
```

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
arrow-graph = "0.1"
```

### Basic Usage

```rust
use arrow_graph::prelude::*;
use arrow::array::{StringArray, Float64Array};
use arrow::record_batch::RecordBatch;

// Create a graph from Arrow data
let edges = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(StringArray::from(vec!["A", "B", "C"])),  // source
        Arc::new(StringArray::from(vec!["B", "C", "D"])),  // target  
        Arc::new(Float64Array::from(vec![1.0, 2.0, 1.5])), // weight
    ],
)?;

let graph = ArrowGraph::from_edges(edges)?;

// Basic graph operations
println!("Nodes: {}", graph.node_count());
println!("Edges: {}", graph.edge_count()); 
println!("Density: {:.3}", graph.density());

// Navigate the graph
let neighbors = graph.neighbors("A").unwrap();
println!("A connects to: {:?}", neighbors);
```

### SQL Interface (Coming in v0.3.0)

Built as DataFusion User-Defined Functions (UDFs) for seamless integration:

```sql
-- Find shortest paths
SELECT shortest_path('A', 'D', 'edges_table') as path;

-- Calculate PageRank
SELECT node_id, pagerank() OVER (PARTITION BY graph_id) as rank
FROM nodes_table;

-- Detect communities  
SELECT node_id, community_detection('leiden', 'edges_table') as cluster
FROM nodes_table;
```

**Current prototype** (basic graph metrics available):
```rust
// Register UDF with DataFusion
ctx.register_udf(create_udf(
    "graph_density",
    vec![DataType::Utf8], // table name
    Arc::new(DataType::Float64),
    Volatility::Stable,
    Arc::new(|args| {
        let edges = get_edges_table(args[0].as_ref())?;
        Ok(ColumnarValue::Scalar(ScalarValue::Float64(
            Some(calculate_density(&edges)?)
        )))
    }),
));
```

## Features

### Current (v0.1.0)
- ✅ Arrow RecordBatch graph construction
- ✅ Efficient adjacency list indexing
- ✅ Basic graph metrics (density, node/edge counts)
- ✅ Graph navigation (neighbors, predecessors)
- ✅ CLI tool foundation

### Coming Soon (v0.2.0)
- 🚧 Vectorized shortest path algorithms
- 🚧 PageRank with early termination
- 🚧 Connected components analysis
- 🚧 Community detection (Leiden algorithm)
- 🚧 Triangle counting and clustering

### Roadmap
- **v0.3.0**: SQL integration with DataFusion
- **v0.4.0**: Streaming graph processing 
- **v0.5.0**: ML/AI integration (GNN support)
- **v0.6.0**: Cloud-native distributed processing

## Performance

Built for modern data scales:
- **Target**: 10-100x faster than NetworkX (based on initial benchmarks with 1M edge datasets)
- **Scale**: Handle 100M+ edges on single machine
- **Memory**: Efficient columnar storage with zero-copy operations
- **SIMD**: Vectorized algorithms using Arrow compute kernels

*Comprehensive benchmarks coming in v0.2.0 release*

## Use Cases

Arrow-Graph is designed for modern graph analytics workflows:

### Social Network Analysis
- **Friend recommendations**: Find mutual connections and suggest new relationships
- **Influence measurement**: Calculate centrality metrics for key opinion leaders
- **Community detection**: Identify clusters and groups within social networks

### Fraud Detection
- **Transaction networks**: Analyze payment flows to detect suspicious patterns
- **Account linking**: Find connected accounts through shared attributes
- **Risk scoring**: Calculate graph-based features for ML fraud models

### Knowledge Graphs & Recommendation Systems
- **Product recommendations**: Graph-based collaborative filtering
- **Content discovery**: Find related articles, papers, or media through citation/reference networks
- **Semantic search**: Navigate knowledge graphs for enhanced search results

### ML/AI Feature Engineering
- **GNN preprocessing**: Prepare graph data for PyTorch Geometric or DGL
- **Graph embeddings**: Calculate node2vec, GraphSAGE features at scale
- **Pipeline integration**: Seamless integration with existing ML workflows via Arrow

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQL Layer     │    │   Algorithms     │    │   Storage       │
│                 │    │                  │    │                 │
│ DataFusion UDFs │───▶│ SIMD Optimized   │───▶│ Arrow Columnar  │
│ Graph Functions │    │ Vectorized Ops   │    │ Zero-Copy       │
│ Pattern Matching│    │ Streaming Algos  │    │ Memory Mapped   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by Apache Flink Gelly, built for the modern Arrow ecosystem.


