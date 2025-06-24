#[cfg(test)]
mod tests {
    use crate::ArrowGraph;
    use arrow::record_batch::RecordBatch;
    use arrow::array::{StringArray, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_test_nodes() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, true),
        ]));

        let id_array = StringArray::from(vec!["A", "B", "C", "D"]);
        let label_array = StringArray::from(vec![
            Some("Node A"), 
            Some("Node B"), 
            Some("Node C"), 
            Some("Node D")
        ]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array) as Arc<dyn arrow::array::Array>, 
                Arc::new(label_array) as Arc<dyn arrow::array::Array>
            ],
        ).unwrap()
    }

    fn create_test_edges() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, true),
        ]));

        let source_array = StringArray::from(vec!["A", "A", "B", "C"]);
        let target_array = StringArray::from(vec!["B", "C", "C", "D"]);
        let weight_array = Float64Array::from(vec![
            Some(1.0), 
            Some(2.0), 
            Some(1.5), 
            Some(0.5)
        ]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(source_array) as Arc<dyn arrow::array::Array>, 
                Arc::new(target_array) as Arc<dyn arrow::array::Array>, 
                Arc::new(weight_array) as Arc<dyn arrow::array::Array>
            ],
        ).unwrap()
    }

    #[test]
    fn test_graph_creation() {
        let nodes = create_test_nodes();
        let edges = create_test_edges();
        
        let graph = ArrowGraph::new(nodes, edges).unwrap();
        
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
        
        // Test adjacency
        assert!(graph.has_node("A"));
        assert!(graph.has_node("B"));
        assert!(graph.has_node("C"));
        assert!(graph.has_node("D"));
        assert!(!graph.has_node("E"));
        
        // Test neighbors
        let a_neighbors = graph.neighbors("A").unwrap();
        assert_eq!(a_neighbors.len(), 2);
        assert!(a_neighbors.contains(&"B".to_string()));
        assert!(a_neighbors.contains(&"C".to_string()));
        
        // Test edge weights
        assert_eq!(graph.edge_weight("A", "B"), Some(1.0));
        assert_eq!(graph.edge_weight("A", "C"), Some(2.0));
        assert_eq!(graph.edge_weight("A", "D"), None);
    }

    #[test]
    fn test_graph_from_edges_only() {
        let edges = create_test_edges();
        let graph = ArrowGraph::from_edges(edges).unwrap();
        
        // Should infer 4 nodes from edges
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
        
        assert!(graph.has_node("A"));
        assert!(graph.has_node("B"));
        assert!(graph.has_node("C"));
        assert!(graph.has_node("D"));
    }

    #[test]
    fn test_graph_density() {
        let nodes = create_test_nodes();
        let edges = create_test_edges();
        let graph = ArrowGraph::new(nodes, edges).unwrap();
        
        // 4 nodes, 4 edges, max possible = 4 * 3 = 12
        // density = 4/12 = 1/3 ≈ 0.333
        let density = graph.density();
        assert!((density - 0.3333333333333333).abs() < 0.0001);
    }

    #[test]
    fn test_predecessors() {
        let nodes = create_test_nodes();
        let edges = create_test_edges();
        let graph = ArrowGraph::new(nodes, edges).unwrap();
        
        // C has predecessors A and B
        let c_predecessors = graph.predecessors("C").unwrap();
        assert_eq!(c_predecessors.len(), 2);
        assert!(c_predecessors.contains(&"A".to_string()));
        assert!(c_predecessors.contains(&"B".to_string()));
        
        // A has no predecessors
        let a_predecessors = graph.predecessors("A").unwrap();
        assert_eq!(a_predecessors.len(), 0);
    }
}