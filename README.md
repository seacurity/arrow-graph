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
- **Target**: 10-100x faster than NetworkX
- **Scale**: Handle 100M+ edges on single machine
- **Memory**: Efficient columnar storage with zero-copy operations
- **SIMD**: Vectorized algorithms using Arrow compute kernels

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


