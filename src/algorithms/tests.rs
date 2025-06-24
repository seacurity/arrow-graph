#[cfg(test)]
mod tests {
    use crate::algorithms::pathfinding::{ShortestPath, AllPaths};
    use crate::algorithms::centrality::PageRank;
    use crate::{GraphAlgorithm, AlgorithmParams};
    use crate::ArrowGraph;
    use arrow::record_batch::RecordBatch;
    use arrow::array::{StringArray, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_test_graph() -> ArrowGraph {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        let source_array = StringArray::from(vec!["A", "A", "B", "C", "D"]);
        let target_array = StringArray::from(vec!["B", "C", "C", "D", "E"]);
        let weight_array = Float64Array::from(vec![
            Some(1.0), 
            Some(4.0), 
            Some(2.0), 
            Some(1.0),
            Some(1.0)
        ]);

        let edges = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(source_array) as Arc<dyn arrow::array::Array>, 
                Arc::new(target_array) as Arc<dyn arrow::array::Array>, 
                Arc::new(weight_array) as Arc<dyn arrow::array::Array>
            ],
        ).unwrap();

        ArrowGraph::from_edges(edges).unwrap()
    }

    #[test]
    fn test_shortest_path_single_target() {
        let graph = create_test_graph();
        let algorithm = ShortestPath;
        
        let params = AlgorithmParams::new()
            .with_param("source", "A")
            .with_param("target", "D");
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.num_columns(), 4);
        
        // Check that we got a result with source A, target D
        let source_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let target_col = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let distance_col = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        
        assert_eq!(source_col.value(0), "A");
        assert_eq!(target_col.value(0), "D");
        assert_eq!(distance_col.value(0), 4.0); // A->B->C->D = 1+2+1 = 4, A->C->D = 4+1 = 5, so shortest is A->B->C->D = 4
    }

    #[test]
    fn test_shortest_path_all_targets() {
        let graph = create_test_graph();
        let algorithm = ShortestPath;
        
        let params = AlgorithmParams::new()
            .with_param("source", "A");
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should have paths to all reachable nodes except source
        assert!(result.num_rows() > 0);
        assert_eq!(result.num_columns(), 3);
        
        let source_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        
        // All sources should be "A"
        for i in 0..result.num_rows() {
            assert_eq!(source_col.value(i), "A");
        }
    }

    #[test]
    fn test_all_paths() {
        let graph = create_test_graph();
        let algorithm = AllPaths;
        
        let params = AlgorithmParams::new()
            .with_param("source", "A")
            .with_param("target", "D")
            .with_param("max_hops", 5);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should find multiple paths from A to D
        assert!(result.num_rows() >= 1);
        assert_eq!(result.num_columns(), 4);
        
        let source_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let target_col = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        
        for i in 0..result.num_rows() {
            assert_eq!(source_col.value(i), "A");
            assert_eq!(target_col.value(i), "D");
        }
    }

    #[test]
    fn test_shortest_path_no_path() {
        let graph = create_test_graph();
        let algorithm = ShortestPath;
        
        let params = AlgorithmParams::new()
            .with_param("source", "E")
            .with_param("target", "A");
        
        let result = algorithm.execute(&graph, &params);
        
        // Should return an error since there's no path from E to A
        assert!(result.is_err());
    }

    #[test]
    fn test_pagerank() {
        let graph = create_test_graph();
        let algorithm = PageRank;
        
        let params = AlgorithmParams::new()
            .with_param("damping_factor", 0.85)
            .with_param("max_iterations", 100)
            .with_param("tolerance", 1e-6);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should have PageRank scores for all nodes
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let score_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all nodes are present
        let mut found_nodes = std::collections::HashSet::new();
        let mut total_score = 0.0;
        
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let score = score_col.value(i);
            
            found_nodes.insert(node_id);
            total_score += score;
            
            // PageRank scores should be positive
            assert!(score > 0.0);
        }
        
        // Check that we have all expected nodes
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
        
        // PageRank scores should sum to approximately 1.0
        assert!((total_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pagerank_with_custom_params() {
        let graph = create_test_graph();
        let algorithm = PageRank;
        
        let params = AlgorithmParams::new()
            .with_param("damping_factor", 0.5)
            .with_param("max_iterations", 10)
            .with_param("tolerance", 1e-3);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert!(result.num_rows() > 0);
        
        let score_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let total_score: f64 = (0..result.num_rows()).map(|i| score_col.value(i)).sum();
        
        // Should still sum to approximately 1.0 even with different parameters
        assert!((total_score - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_pagerank_invalid_params() {
        let graph = create_test_graph();
        let algorithm = PageRank;
        
        // Test invalid damping factor
        let params = AlgorithmParams::new().with_param("damping_factor", 1.5);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        // Test invalid max_iterations
        let params = AlgorithmParams::new().with_param("max_iterations", 0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        // Test invalid tolerance
        let params = AlgorithmParams::new().with_param("tolerance", -1.0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_algorithm_names() {
        let shortest_path = ShortestPath;
        let all_paths = AllPaths;
        let pagerank = PageRank;
        
        assert_eq!(shortest_path.name(), "shortest_path");
        assert_eq!(all_paths.name(), "all_paths");
        assert_eq!(pagerank.name(), "pagerank");
    }
}