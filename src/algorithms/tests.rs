#[cfg(test)]
mod tests {
    use crate::algorithms::pathfinding::{ShortestPath, AllPaths};
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
    fn test_algorithm_names() {
        let shortest_path = ShortestPath;
        let all_paths = AllPaths;
        
        assert_eq!(shortest_path.name(), "shortest_path");
        assert_eq!(all_paths.name(), "all_paths");
    }
}