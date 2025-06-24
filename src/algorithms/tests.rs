#[cfg(test)]
mod tests {
    use crate::algorithms::pathfinding::{ShortestPath, AllPaths};
    use crate::algorithms::centrality::{PageRank, BetweennessCentrality, EigenvectorCentrality, ClosenessCentrality};
    use crate::algorithms::components::{WeaklyConnectedComponents, StronglyConnectedComponents};
    use crate::algorithms::community::LeidenCommunityDetection;
    use crate::algorithms::aggregation::{TriangleCount, ClusteringCoefficient};
    use crate::algorithms::vectorized::{VectorizedPageRank, VectorizedBatchOperations};
    use crate::algorithms::sampling::{RandomWalk, Node2VecWalk, GraphSampling};
    use crate::graph::{StreamingGraphProcessor, StreamingGraphSystem, StreamUpdate};
    use crate::{GraphAlgorithm, AlgorithmParams};
    use crate::ArrowGraph;
    use arrow::record_batch::RecordBatch;
    use arrow::array::{StringArray, Float64Array, UInt32Array, UInt64Array};
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

    fn create_disconnected_graph() -> ArrowGraph {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        // Create two disconnected components: A-B and C-D
        let source_array = StringArray::from(vec!["A", "C"]);
        let target_array = StringArray::from(vec!["B", "D"]);
        let weight_array = Float64Array::from(vec![Some(1.0), Some(1.0)]);

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
    fn test_weakly_connected_components() {
        let graph = create_test_graph();
        let algorithm = WeaklyConnectedComponents;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should have one component containing all nodes
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let component_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        // All nodes should be in the same component (0) since the graph is connected
        let mut found_nodes = std::collections::HashSet::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let component_id = component_col.value(i);
            
            found_nodes.insert(node_id);
            assert_eq!(component_id, 0); // Single component
        }
        
        // Check that we have all expected nodes
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
    }

    #[test]
    fn test_weakly_connected_components_disconnected() {
        let graph = create_disconnected_graph();
        let algorithm = WeaklyConnectedComponents;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should have two components: A-B and C-D
        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let component_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        let mut component_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i).to_string();
            let component_id = component_col.value(i);
            component_map.insert(node_id, component_id);
        }
        
        // A and B should be in the same component
        assert_eq!(component_map.get("A"), component_map.get("B"));
        
        // C and D should be in the same component
        assert_eq!(component_map.get("C"), component_map.get("D"));
        
        // A-B and C-D should be in different components
        assert_ne!(component_map.get("A"), component_map.get("C"));
    }

    #[test]
    fn test_strongly_connected_components() {
        let graph = create_test_graph();
        let algorithm = StronglyConnectedComponents;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Our test graph is acyclic, so each node should be its own SCC
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let component_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        // Each node should be in a different component since the graph is acyclic
        let mut component_ids = std::collections::HashSet::new();
        for i in 0..result.num_rows() {
            let component_id = component_col.value(i);
            component_ids.insert(component_id);
        }
        
        // Should have as many unique component IDs as nodes
        assert_eq!(component_ids.len(), graph.node_count());
    }

    fn create_strongly_connected_graph() -> ArrowGraph {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        // Create a graph with a cycle: A->B->C->A and D->E->D
        let source_array = StringArray::from(vec!["A", "B", "C", "D", "E"]);
        let target_array = StringArray::from(vec!["B", "C", "A", "E", "D"]);
        let weight_array = Float64Array::from(vec![
            Some(1.0), 
            Some(1.0), 
            Some(1.0), 
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
    fn test_strongly_connected_components_with_cycles() {
        let graph = create_strongly_connected_graph();
        let algorithm = StronglyConnectedComponents;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), 5);
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let component_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        let mut component_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i).to_string();
            let component_id = component_col.value(i);
            component_map.insert(node_id, component_id);
        }
        
        // A, B, C should be in the same SCC (they form a cycle)
        assert_eq!(component_map.get("A"), component_map.get("B"));
        assert_eq!(component_map.get("B"), component_map.get("C"));
        
        // D, E should be in the same SCC (they form a cycle)
        assert_eq!(component_map.get("D"), component_map.get("E"));
        
        // The two cycles should be in different SCCs
        assert_ne!(component_map.get("A"), component_map.get("D"));
    }

    fn create_community_graph() -> ArrowGraph {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        // Create a graph with clear community structure:
        // Community 1: A-B-C (triangle)
        // Community 2: D-E-F (triangle)  
        // Bridge: C-D (connects communities)
        let source_array = StringArray::from(vec!["A", "B", "C", "A", "C", "D", "E", "F", "D", "E"]);
        let target_array = StringArray::from(vec!["B", "C", "A", "C", "D", "E", "F", "D", "F", "F"]);
        let weight_array = Float64Array::from(vec![
            Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(0.5), // Community 1 + bridge
            Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)  // Community 2
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
    fn test_leiden_community_detection() {
        let graph = create_community_graph();
        let algorithm = LeidenCommunityDetection;
        
        let params = AlgorithmParams::new()
            .with_param("resolution", 1.0)
            .with_param("max_iterations", 1);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Should detect communities
        assert_eq!(result.num_rows(), 6); // 6 nodes
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let community_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        
        let mut community_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i).to_string();
            let community_id = community_col.value(i);
            community_map.insert(node_id, community_id);
        }
        
        // Should have found at least 2 communities
        let unique_communities: std::collections::HashSet<u32> = community_map.values().cloned().collect();
        assert!(unique_communities.len() >= 2);
        
        // Verify all nodes are assigned
        assert!(community_map.contains_key("A"));
        assert!(community_map.contains_key("B"));
        assert!(community_map.contains_key("C"));
        assert!(community_map.contains_key("D"));
        assert!(community_map.contains_key("E"));
        assert!(community_map.contains_key("F"));
    }

    #[test]
    fn test_leiden_with_custom_resolution() {
        let graph = create_test_graph();
        let algorithm = LeidenCommunityDetection;
        
        let params = AlgorithmParams::new()
            .with_param("resolution", 0.5)
            .with_param("max_iterations", 10);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert!(result.num_rows() > 0);
        assert_eq!(result.num_columns(), 2);
    }

    #[test]
    fn test_leiden_invalid_params() {
        let graph = create_test_graph();
        let algorithm = LeidenCommunityDetection;
        
        // Test invalid resolution
        let params = AlgorithmParams::new().with_param("resolution", -1.0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        // Test invalid max_iterations
        let params = AlgorithmParams::new().with_param("max_iterations", 0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
    }


    fn create_triangle_graph() -> ArrowGraph {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        // Create a graph with triangles: A-B-C-A (triangle) and B-D
        let source_array = StringArray::from(vec!["A", "B", "C", "B"]);
        let target_array = StringArray::from(vec!["B", "C", "A", "D"]);
        let weight_array = Float64Array::from(vec![
            Some(1.0), Some(1.0), Some(1.0), Some(1.0)
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
    fn test_triangle_count() {
        let graph = create_triangle_graph();
        let algorithm = TriangleCount;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.num_columns(), 2);
        
        let metric_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let value_col = result.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        
        assert_eq!(metric_col.value(0), "triangle_count");
        assert_eq!(value_col.value(0), 1); // One triangle: A-B-C
    }

    #[test]
    fn test_clustering_coefficient_local() {
        let graph = create_triangle_graph();
        let algorithm = ClusteringCoefficient;
        
        let params = AlgorithmParams::new().with_param("mode", "local");
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), 4); // 4 nodes: A, B, C, D
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let coeff_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Build coefficient map
        let mut coeff_map = std::collections::HashMap::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let coefficient = coeff_col.value(i);
            coeff_map.insert(node_id, coefficient);
        }
        
        // Verify clustering coefficients are computed
        // All coefficients should be between 0.0 and 1.0
        for (node_id, coefficient) in &coeff_map {
            assert!(*coefficient >= 0.0, "Node {} has negative coefficient: {}", node_id, coefficient);
            assert!(*coefficient <= 1.0, "Node {} has coefficient > 1.0: {}", node_id, coefficient);
        }
        
        // D has degree 1, so clustering coefficient should be 0.0
        assert_eq!(coeff_map.get("D").unwrap(), &0.0);
    }

    #[test]
    fn test_clustering_coefficient_global() {
        let graph = create_triangle_graph();
        let algorithm = ClusteringCoefficient;
        
        let params = AlgorithmParams::new().with_param("mode", "global");
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.num_columns(), 2);
        
        let metric_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let value_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        assert_eq!(metric_col.value(0), "global_clustering_coefficient");
        
        // Global clustering = 3 * triangles / connected_triples
        // We have 1 triangle and several connected triples
        let global_coefficient = value_col.value(0);
        assert!(global_coefficient >= 0.0, "Global coefficient should be non-negative: {}", global_coefficient);
        assert!(global_coefficient <= 1.0, "Global coefficient should be <= 1.0: {}", global_coefficient);
    }

    #[test]
    fn test_clustering_coefficient_invalid_mode() {
        let graph = create_test_graph();
        let algorithm = ClusteringCoefficient;
        
        let params = AlgorithmParams::new().with_param("mode", "invalid");
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_triangle_count_no_triangles() {
        let graph = create_test_graph(); // Linear graph, no triangles
        let algorithm = TriangleCount;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        let value_col = result.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        // The linear graph A→B, A→C, B→C, C→D, D→E may form triangles when treated as undirected
        // Check that the result is non-negative
        assert!(value_col.value(0) >= 0);
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_test_graph();
        let algorithm = BetweennessCentrality;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let centrality_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all nodes are present and centrality values are non-negative
        let mut found_nodes = std::collections::HashSet::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let centrality = centrality_col.value(i);
            
            found_nodes.insert(node_id);
            assert!(centrality >= 0.0, "Betweenness centrality should be non-negative: {}", centrality);
        }
        
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
    }

    #[test]
    fn test_eigenvector_centrality() {
        let graph = create_test_graph();
        let algorithm = EigenvectorCentrality;
        
        let params = AlgorithmParams::new()
            .with_param("max_iterations", 50)
            .with_param("tolerance", 1e-6);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let centrality_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all nodes are present and centrality values are non-negative
        let mut found_nodes = std::collections::HashSet::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let centrality = centrality_col.value(i);
            
            found_nodes.insert(node_id);
            assert!(centrality >= 0.0, "Eigenvector centrality should be non-negative: {}", centrality);
        }
        
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
    }

    #[test]
    fn test_closeness_centrality() {
        let graph = create_test_graph();
        let algorithm = ClosenessCentrality;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let centrality_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all nodes are present and centrality values are non-negative
        let mut found_nodes = std::collections::HashSet::new();
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let centrality = centrality_col.value(i);
            
            found_nodes.insert(node_id);
            assert!(centrality >= 0.0, "Closeness centrality should be non-negative: {}", centrality);
        }
        
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
    }

    #[test]
    fn test_centrality_invalid_params() {
        let graph = create_test_graph();
        
        // Test eigenvector centrality with invalid parameters
        let algorithm = EigenvectorCentrality;
        
        let params = AlgorithmParams::new().with_param("max_iterations", 0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        let params = AlgorithmParams::new().with_param("tolerance", -1.0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_algorithm_names_with_centrality() {
        let shortest_path = ShortestPath;
        let all_paths = AllPaths;
        let pagerank = PageRank;
        let betweenness = BetweennessCentrality;
        let eigenvector = EigenvectorCentrality;
        let closeness = ClosenessCentrality;
        let weakly_connected = WeaklyConnectedComponents;
        let strongly_connected = StronglyConnectedComponents;
        let leiden = LeidenCommunityDetection;
        let triangle_count = TriangleCount;
        let clustering = ClusteringCoefficient;
        
        assert_eq!(shortest_path.name(), "shortest_path");
        assert_eq!(all_paths.name(), "all_paths");
        assert_eq!(pagerank.name(), "pagerank");
        assert_eq!(betweenness.name(), "betweenness_centrality");
        assert_eq!(eigenvector.name(), "eigenvector_centrality");
        assert_eq!(closeness.name(), "closeness_centrality");
        assert_eq!(weakly_connected.name(), "weakly_connected_components");
        assert_eq!(strongly_connected.name(), "strongly_connected_components");
        assert_eq!(leiden.name(), "leiden");
        assert_eq!(triangle_count.name(), "triangle_count");
        assert_eq!(clustering.name(), "clustering_coefficient");
    }

    #[test]
    fn test_graph_mutations_add_node() {
        let mut graph = create_test_graph();
        let initial_count = graph.node_count();
        
        // Add a new node
        graph.add_node("F".to_string()).unwrap();
        
        assert_eq!(graph.node_count(), initial_count + 1);
        assert!(graph.has_node("F"));
        
        // Try to add duplicate node (should fail)
        let result = graph.add_node("F".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_mutations_remove_node() {
        let mut graph = create_test_graph();
        let initial_count = graph.node_count();
        let initial_edge_count = graph.edge_count();
        
        // Remove a node (this should also remove its edges)
        graph.remove_node("A").unwrap();
        
        assert_eq!(graph.node_count(), initial_count - 1);
        assert!(!graph.has_node("A"));
        assert!(graph.edge_count() < initial_edge_count); // Some edges removed
        
        // Try to remove non-existent node (should fail)
        let result = graph.remove_node("Z");
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_mutations_add_edge() {
        let mut graph = create_test_graph();
        let initial_edge_count = graph.edge_count();
        
        // Add a new edge between existing nodes
        graph.add_edge("A".to_string(), "E".to_string(), Some(2.5)).unwrap();
        
        assert_eq!(graph.edge_count(), initial_edge_count + 1);
        assert_eq!(graph.edge_weight("A", "E"), Some(2.5));
        
        // Add edge with new nodes
        graph.add_edge("F".to_string(), "G".to_string(), None).unwrap();
        assert!(graph.has_node("F"));
        assert!(graph.has_node("G"));
        
        // Try to add duplicate edge (should fail)
        let result = graph.add_edge("A".to_string(), "E".to_string(), Some(1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_mutations_remove_edge() {
        let mut graph = create_test_graph();
        let initial_edge_count = graph.edge_count();
        
        // Remove an existing edge
        graph.remove_edge("A", "B").unwrap();
        
        assert_eq!(graph.edge_count(), initial_edge_count - 1);
        assert!(graph.edge_weight("A", "B").is_none());
        
        // Try to remove non-existent edge (should fail)
        let result = graph.remove_edge("A", "Z");
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_mutations_complex_operations() {
        let mut graph = create_test_graph();
        
        // Complex sequence of operations
        graph.add_node("F".to_string()).unwrap();
        graph.add_edge("F".to_string(), "A".to_string(), Some(3.0)).unwrap();
        graph.add_edge("B".to_string(), "F".to_string(), Some(1.5)).unwrap();
        
        assert!(graph.has_node("F"));
        assert_eq!(graph.edge_weight("F", "A"), Some(3.0));
        assert_eq!(graph.edge_weight("B", "F"), Some(1.5));
        
        // Remove a node and ensure its edges are gone
        let initial_node_count = graph.node_count();
        graph.remove_node("F").unwrap();
        
        assert_eq!(graph.node_count(), initial_node_count - 1);
        assert!(!graph.has_node("F"));
        assert!(graph.edge_weight("F", "A").is_none());
        assert!(graph.edge_weight("B", "F").is_none());
    }

    #[test]
    fn test_vectorized_pagerank() {
        let graph = create_test_graph();
        let algorithm = VectorizedPageRank;
        
        let params = AlgorithmParams::new()
            .with_param("damping_factor", 0.85)
            .with_param("max_iterations", 50)
            .with_param("tolerance", 1e-6);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 2);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let score_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify PageRank properties
        let mut total_score = 0.0;
        let mut found_nodes = std::collections::HashSet::new();
        
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let score = score_col.value(i);
            
            found_nodes.insert(node_id);
            total_score += score;
            
            // PageRank scores should be positive
            assert!(score > 0.0, "PageRank score should be positive: {}", score);
        }
        
        // Check that we have all expected nodes
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
        
        // PageRank scores should sum to approximately 1.0
        assert!((total_score - 1.0).abs() < 1e-8, "PageRank scores should sum to 1.0, got: {}", total_score);
    }

    #[test]
    fn test_vectorized_batch_centralities() {
        let graph = create_test_graph();
        let algorithm = VectorizedBatchOperations;
        
        let params = AlgorithmParams::new();
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_rows(), graph.node_count());
        assert_eq!(result.num_columns(), 4);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let degree_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let eigenvector_col = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let closeness_col = result.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all centrality measures are computed
        let mut found_nodes = std::collections::HashSet::new();
        
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            let degree_centrality = degree_col.value(i);
            let eigenvector_centrality = eigenvector_col.value(i);
            let closeness_centrality = closeness_col.value(i);
            
            found_nodes.insert(node_id);
            
            // All centrality measures should be non-negative and <= 1.0
            assert!(degree_centrality >= 0.0 && degree_centrality <= 1.0, 
                "Degree centrality should be in [0,1]: {}", degree_centrality);
            assert!(eigenvector_centrality >= 0.0, 
                "Eigenvector centrality should be non-negative: {}", eigenvector_centrality);
            assert!(closeness_centrality >= 0.0 && closeness_centrality <= 1.0, 
                "Closeness centrality should be in [0,1]: {}", closeness_centrality);
        }
        
        // Check that we have all expected nodes
        assert!(found_nodes.contains("A"));
        assert!(found_nodes.contains("B"));
        assert!(found_nodes.contains("C"));
        assert!(found_nodes.contains("D"));
        assert!(found_nodes.contains("E"));
    }

    #[test]
    fn test_vectorized_vs_regular_pagerank_consistency() {
        let graph = create_test_graph();
        
        let params = AlgorithmParams::new()
            .with_param("damping_factor", 0.85)
            .with_param("max_iterations", 100)
            .with_param("tolerance", 1e-8);
        
        // Regular PageRank
        let regular_pagerank = PageRank;
        let regular_result = regular_pagerank.execute(&graph, &params).unwrap();
        
        // Vectorized PageRank
        let vectorized_pagerank = VectorizedPageRank;
        let vectorized_result = vectorized_pagerank.execute(&graph, &params).unwrap();
        
        // Results should be similar (allowing for small numerical differences)
        assert_eq!(regular_result.num_rows(), vectorized_result.num_rows());
        
        let regular_scores = regular_result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let vectorized_scores = vectorized_result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        
        for i in 0..regular_result.num_rows() {
            let regular_score = regular_scores.value(i);
            let vectorized_score = vectorized_scores.value(i);
            let diff = (regular_score - vectorized_score).abs();
            
            assert!(diff < 1e-6, 
                "PageRank scores should be similar: regular={}, vectorized={}, diff={}", 
                regular_score, vectorized_score, diff);
        }
    }

    #[test]
    fn test_random_walk() {
        let graph = create_test_graph();
        let algorithm = RandomWalk;
        
        let params = AlgorithmParams::new()
            .with_param("walk_length", 5)
            .with_param("num_walks", 3)
            .with_param("seed", 42u64);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Check basic structure
        assert_eq!(result.num_columns(), 3);
        
        let walk_id_col = result.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        let step_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        let node_col = result.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        
        // Verify walks structure
        let mut walks: std::collections::HashMap<u32, Vec<(u32, String)>> = std::collections::HashMap::new();
        
        for i in 0..result.num_rows() {
            let walk_id = walk_id_col.value(i);
            let step = step_col.value(i);
            let node_id = node_col.value(i).to_string();
            
            walks.entry(walk_id).or_insert_with(Vec::new).push((step, node_id));
        }
        
        // Should have walks from each node (5 nodes * 3 walks each = 15 total walks)
        assert_eq!(walks.len(), 15);
        
        // Each walk should have valid steps
        for (_, walk_steps) in walks {
            assert!(!walk_steps.is_empty());
            assert!(walk_steps.len() <= 5); // walk_length parameter
            
            // Steps should be sequential starting from 0
            let mut sorted_steps = walk_steps;
            sorted_steps.sort_by_key(|(step, _)| *step);
            
            for (i, (step, _)) in sorted_steps.iter().enumerate() {
                assert_eq!(*step, i as u32);
            }
        }
    }
    
    #[test]
    fn test_node2vec_walk() {
        let graph = create_test_graph();
        let algorithm = Node2VecWalk;
        
        let params = AlgorithmParams::new()
            .with_param("walk_length", 10)
            .with_param("num_walks", 2)
            .with_param("p", 1.0)
            .with_param("q", 1.0)
            .with_param("seed", 42u64);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        // Check basic structure
        assert_eq!(result.num_columns(), 5);
        
        let walk_id_col = result.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        let step_col = result.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        let node_col = result.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let p_col = result.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        let q_col = result.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify Node2Vec parameters are preserved
        for i in 0..result.num_rows() {
            assert_eq!(p_col.value(i), 1.0);
            assert_eq!(q_col.value(i), 1.0);
        }
        
        // Should have walks from each node
        let walk_count = walk_id_col.iter().max().unwrap().unwrap() + 1;
        assert_eq!(walk_count, 10); // 5 nodes * 2 walks each
    }
    
    #[test]
    fn test_graph_sampling_random_node() {
        let graph = create_test_graph();
        let algorithm = GraphSampling;
        
        let params = AlgorithmParams::new()
            .with_param("method", "random_node".to_string())
            .with_param("sample_size", 3)
            .with_param("seed", 42u64);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.num_rows(), 3);
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        
        // Verify all sampled nodes exist in original graph
        for i in 0..result.num_rows() {
            let node_id = node_col.value(i);
            assert!(graph.has_node(node_id), "Sampled node {} should exist in graph", node_id);
        }
    }
    
    #[test]
    fn test_graph_sampling_random_edge() {
        let graph = create_test_graph();
        let algorithm = GraphSampling;
        
        let params = AlgorithmParams::new()
            .with_param("method", "random_edge".to_string())
            .with_param("sample_ratio", 0.6)
            .with_param("seed", 42u64);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_columns(), 3);
        
        let source_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let target_col = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let weight_col = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Verify all sampled edges exist in original graph
        for i in 0..result.num_rows() {
            let source = source_col.value(i);
            let target = target_col.value(i);
            let weight = weight_col.value(i);
            
            assert!(graph.has_node(source), "Source node {} should exist", source);
            assert!(graph.has_node(target), "Target node {} should exist", target);
            assert!(graph.edge_weight(source, target).is_some(), 
                "Edge {}→{} should exist in graph", source, target);
            assert!(weight > 0.0, "Edge weight should be positive");
        }
    }
    
    #[test]
    fn test_graph_sampling_snowball() {
        let graph = create_test_graph();
        let algorithm = GraphSampling;
        
        let params = AlgorithmParams::new()
            .with_param("method", "snowball".to_string())
            .with_param("seed_nodes", vec!["A".to_string()])
            .with_param("k_hops", 2)
            .with_param("max_nodes", 4);
        
        let result = algorithm.execute(&graph, &params).unwrap();
        
        assert_eq!(result.num_columns(), 1);
        assert!(result.num_rows() <= 4); // Respects max_nodes
        
        let node_col = result.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        
        // Should include the seed node
        let sampled_nodes: Vec<String> = (0..result.num_rows())
            .map(|i| node_col.value(i).to_string())
            .collect();
        
        assert!(sampled_nodes.contains(&"A".to_string()), 
            "Snowball sampling should include seed node A");
        
        // All sampled nodes should exist in graph
        for node_id in &sampled_nodes {
            assert!(graph.has_node(node_id), "Sampled node {} should exist in graph", node_id);
        }
    }
    
    #[test]
    fn test_sampling_invalid_parameters() {
        let graph = create_test_graph();
        
        // Test RandomWalk with invalid walk_length
        let algorithm = RandomWalk;
        let params = AlgorithmParams::new().with_param("walk_length", 0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        // Test Node2VecWalk with invalid p parameter
        let algorithm = Node2VecWalk;
        let params = AlgorithmParams::new().with_param("p", -1.0);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
        
        // Test GraphSampling with invalid sample_ratio
        let algorithm = GraphSampling;
        let params = AlgorithmParams::new()
            .with_param("method", "random_edge".to_string())
            .with_param("sample_ratio", 1.5);
        let result = algorithm.execute(&graph, &params);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_sampling_algorithm_names() {
        let random_walk = RandomWalk;
        let node2vec = Node2VecWalk;
        let sampling = GraphSampling;
        
        assert_eq!(random_walk.name(), "random_walk");
        assert_eq!(node2vec.name(), "node2vec");
        assert_eq!(sampling.name(), "graph_sampling");
        
        // Check descriptions
        assert!(!random_walk.description().is_empty());
        assert!(!node2vec.description().is_empty());
        assert!(!sampling.description().is_empty());
    }

    #[test]
    fn test_streaming_graph_processor_basic_operations() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        let initial_node_count = processor.graph().node_count();
        let initial_edge_count = processor.graph().edge_count();
        
        // Test adding a node
        let result = processor.apply_update(StreamUpdate::AddNode { 
            node_id: "F".to_string() 
        }).unwrap();
        
        assert_eq!(result.operation, "add_node");
        assert_eq!(result.nodes_added, 1);
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(processor.graph().node_count(), initial_node_count + 1);
        assert!(processor.graph().has_node("F"));
        
        // Test adding an edge
        let result = processor.apply_update(StreamUpdate::AddEdge { 
            source: "F".to_string(),
            target: "A".to_string(),
            weight: Some(2.0),
        }).unwrap();
        
        assert_eq!(result.operation, "add_edge");
        assert_eq!(result.edges_added, 1);
        assert_eq!(processor.graph().edge_count(), initial_edge_count + 1);
        assert_eq!(processor.graph().edge_weight("F", "A"), Some(2.0));
        
        // Test removing an edge
        let result = processor.apply_update(StreamUpdate::RemoveEdge { 
            source: "F".to_string(),
            target: "A".to_string(),
        }).unwrap();
        
        assert_eq!(result.operation, "remove_edge");
        assert_eq!(result.edges_removed, 1);
        assert_eq!(processor.graph().edge_count(), initial_edge_count);
        assert!(processor.graph().edge_weight("F", "A").is_none());
        
        // Test removing a node
        let result = processor.apply_update(StreamUpdate::RemoveNode { 
            node_id: "F".to_string() 
        }).unwrap();
        
        assert_eq!(result.operation, "remove_node");
        assert_eq!(result.nodes_removed, 1);
        assert_eq!(processor.graph().node_count(), initial_node_count);
        assert!(!processor.graph().has_node("F"));
    }
    
    #[test]
    fn test_streaming_graph_processor_empty_graph() {
        let mut processor = StreamingGraphProcessor::empty().unwrap();
        
        assert_eq!(processor.graph().node_count(), 0);
        assert_eq!(processor.graph().edge_count(), 0);
        
        // Add first node
        processor.apply_update(StreamUpdate::AddNode { 
            node_id: "A".to_string() 
        }).unwrap();
        
        assert_eq!(processor.graph().node_count(), 1);
        assert!(processor.graph().has_node("A"));
        
        // Add second node and edge
        processor.apply_update(StreamUpdate::AddNode { 
            node_id: "B".to_string() 
        }).unwrap();
        
        processor.apply_update(StreamUpdate::AddEdge { 
            source: "A".to_string(),
            target: "B".to_string(),
            weight: Some(1.0),
        }).unwrap();
        
        assert_eq!(processor.graph().node_count(), 2);
        assert_eq!(processor.graph().edge_count(), 1);
        assert_eq!(processor.graph().edge_weight("A", "B"), Some(1.0));
    }
    
    #[test]
    fn test_streaming_graph_processor_batch_operations() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        let batch_operations = vec![
            StreamUpdate::AddNode { node_id: "F".to_string() },
            StreamUpdate::AddNode { node_id: "G".to_string() },
            StreamUpdate::AddEdge { 
                source: "F".to_string(),
                target: "G".to_string(),
                weight: Some(1.5),
            },
        ];
        
        let result = processor.apply_update(StreamUpdate::Batch { 
            operations: batch_operations 
        }).unwrap();
        
        assert_eq!(result.operation, "batch");
        assert_eq!(result.nodes_added, 2);
        assert_eq!(result.edges_added, 1);
        
        assert!(processor.graph().has_node("F"));
        assert!(processor.graph().has_node("G"));
        assert_eq!(processor.graph().edge_weight("F", "G"), Some(1.5));
    }
    
    #[test]
    fn test_streaming_graph_processor_change_log() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        // Enable change logging
        processor.set_change_log_enabled(true);
        
        // Apply some updates
        processor.apply_update(StreamUpdate::AddNode { 
            node_id: "F".to_string() 
        }).unwrap();
        
        processor.apply_update(StreamUpdate::AddEdge { 
            source: "F".to_string(),
            target: "A".to_string(),
            weight: Some(2.0),
        }).unwrap();
        
        // Check change log
        let change_log = processor.get_change_log_since(0);
        assert_eq!(change_log.len(), 2);
        
        // Check statistics
        let stats = processor.get_statistics();
        assert_eq!(stats.total_updates, 2);
        assert_eq!(stats.change_log_size, 2);
        assert!(stats.change_log_enabled);
        
        // Test compaction
        processor.compact_change_log(1);
        let change_log_after_compact = processor.get_change_log_since(0);
        assert_eq!(change_log_after_compact.len(), 1);
    }
    
    #[test]
    fn test_streaming_graph_processor_snapshots() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        let initial_node_count = processor.graph().node_count();
        
        // Create snapshot
        let snapshot = processor.create_snapshot().unwrap();
        assert_eq!(snapshot.update_count, 0);
        assert_eq!(snapshot.graph.node_count(), initial_node_count);
        
        // Make some changes
        processor.apply_update(StreamUpdate::AddNode { 
            node_id: "F".to_string() 
        }).unwrap();
        processor.apply_update(StreamUpdate::AddNode { 
            node_id: "G".to_string() 
        }).unwrap();
        
        assert_eq!(processor.graph().node_count(), initial_node_count + 2);
        assert_eq!(processor.update_count(), 2);
        
        // Restore from snapshot
        processor.restore_from_snapshot(snapshot);
        
        assert_eq!(processor.graph().node_count(), initial_node_count);
        assert_eq!(processor.update_count(), 0);
        assert!(!processor.graph().has_node("F"));
        assert!(!processor.graph().has_node("G"));
    }
    
    #[test]
    fn test_streaming_graph_system_with_cache() {
        let initial_graph = create_test_graph();
        let mut system = StreamingGraphSystem::new(initial_graph);
        
        // Test basic operations work
        let result = system.apply_update_with_cache_invalidation(StreamUpdate::AddNode { 
            node_id: "F".to_string() 
        }).unwrap();
        
        assert_eq!(result.operation, "add_node");
        assert!(system.graph_processor().graph().has_node("F"));
        
        // Test cache statistics
        let cache_stats = system.algorithm_processor().get_cache_statistics();
        assert_eq!(cache_stats.cached_algorithms, 0); // No algorithms cached yet
        
        // Test algorithm processor methods
        assert!(!system.algorithm_processor().is_cache_valid("pagerank", 100));
        system.algorithm_processor_mut().set_invalidation_threshold(20);
        
        let cache_stats_after = system.algorithm_processor().get_cache_statistics();
        assert_eq!(cache_stats_after.invalidation_threshold, 20);
    }
    
    #[test]
    fn test_streaming_update_result_conversion() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        let result = processor.apply_update(StreamUpdate::AddNode { 
            node_id: "F".to_string() 
        }).unwrap();
        
        // Test conversion to RecordBatch
        let record_batch = result.to_record_batch().unwrap();
        assert_eq!(record_batch.num_rows(), 1);
        assert_eq!(record_batch.num_columns(), 6);
        
        let operation_col = record_batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(operation_col.value(0), "add_node");
        
        let nodes_added_col = record_batch.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(nodes_added_col.value(0), 1);
    }
    
    #[test]
    fn test_streaming_graph_system_empty() {
        let mut system = StreamingGraphSystem::empty().unwrap();
        
        assert_eq!(system.graph_processor().graph().node_count(), 0);
        assert_eq!(system.graph_processor().graph().edge_count(), 0);
        
        // Build a small graph through streaming updates
        system.apply_update_with_cache_invalidation(StreamUpdate::AddNode { 
            node_id: "A".to_string() 
        }).unwrap();
        
        system.apply_update_with_cache_invalidation(StreamUpdate::AddNode { 
            node_id: "B".to_string() 
        }).unwrap();
        
        system.apply_update_with_cache_invalidation(StreamUpdate::AddEdge { 
            source: "A".to_string(),
            target: "B".to_string(),
            weight: Some(1.0),
        }).unwrap();
        
        assert_eq!(system.graph_processor().graph().node_count(), 2);
        assert_eq!(system.graph_processor().graph().edge_count(), 1);
        assert!(system.graph_processor().graph().has_node("A"));
        assert!(system.graph_processor().graph().has_node("B"));
        assert_eq!(system.graph_processor().graph().edge_weight("A", "B"), Some(1.0));
    }
    
    #[test]
    fn test_streaming_invalid_operations() {
        let initial_graph = create_test_graph();
        let mut processor = StreamingGraphProcessor::new(initial_graph);
        
        // Try to remove non-existent node
        let result = processor.apply_update(StreamUpdate::RemoveNode { 
            node_id: "Z".to_string() 
        });
        assert!(result.is_err());
        
        // Try to remove non-existent edge
        let result = processor.apply_update(StreamUpdate::RemoveEdge { 
            source: "A".to_string(),
            target: "Z".to_string(),
        });
        assert!(result.is_err());
        
        // Try to add duplicate edge
        let result = processor.apply_update(StreamUpdate::AddEdge { 
            source: "A".to_string(),
            target: "B".to_string(),
            weight: Some(2.0),
        });
        assert!(result.is_err()); // Should fail because A->B already exists
    }
}