use datafusion::error::Result as DataFusionResult;
use std::collections::HashMap;

/// Multi-graph processor for handling multiple graphs in a single query
/// Supports queries like: SELECT * FROM graph1.edges UNION SELECT * FROM graph2.edges
pub struct MultiGraphProcessor;

/// Represents a named graph with its metadata
#[derive(Debug, Clone, PartialEq)]
pub struct NamedGraph {
    pub name: String,
    pub edges: Vec<MultiGraphEdge>,
    pub properties: HashMap<String, String>,
    pub creation_time: Option<i64>,
}

/// Represents an edge in a multi-graph context
#[derive(Debug, Clone, PartialEq)]
pub struct MultiGraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub graph_name: String, // Which graph this edge belongs to
    pub edge_type: Option<String>, // Optional edge type/label
    pub properties: HashMap<String, String>,
}

/// Configuration for multi-graph queries
#[derive(Debug, Clone)]
pub struct MultiGraphConfig {
    pub include_graph_metadata: bool,
    pub merge_duplicate_edges: bool,
    pub edge_weight_combination: EdgeWeightCombination,
}

/// How to combine edge weights when merging graphs
#[derive(Debug, Clone)]
pub enum EdgeWeightCombination {
    Sum,
    Average,
    Maximum,
    Minimum,
    KeepFirst,
    KeepLast,
}

impl Default for MultiGraphConfig {
    fn default() -> Self {
        Self {
            include_graph_metadata: true,
            merge_duplicate_edges: false,
            edge_weight_combination: EdgeWeightCombination::Average,
        }
    }
}

impl MultiGraphProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Create a new named graph
    pub fn create_graph(
        &self,
        name: &str,
        edges: Vec<MultiGraphEdge>,
        properties: Option<HashMap<String, String>>,
    ) -> DataFusionResult<NamedGraph> {
        let graph = NamedGraph {
            name: name.to_string(),
            edges,
            properties: properties.unwrap_or_default(),
            creation_time: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64),
        };

        Ok(graph)
    }

    /// Merge multiple graphs into a single graph
    pub fn merge_graphs(
        &self,
        graphs: &[NamedGraph],
        config: &MultiGraphConfig,
    ) -> DataFusionResult<NamedGraph> {
        let mut merged_edges = Vec::new();
        let mut merged_properties = HashMap::new();

        // Collect all edges from all graphs
        for graph in graphs {
            merged_edges.extend(graph.edges.clone());
            
            // Merge properties with graph name prefix
            for (key, value) in &graph.properties {
                let prefixed_key = format!("{}_{}", graph.name, key);
                merged_properties.insert(prefixed_key, value.clone());
            }
        }

        // Handle duplicate edge merging if requested
        if config.merge_duplicate_edges {
            merged_edges = self.merge_duplicate_edges(merged_edges, &config.edge_weight_combination)?;
        }

        let merged_graph_name = graphs
            .iter()
            .map(|g| g.name.as_str())
            .collect::<Vec<_>>()
            .join("_merged_");

        Ok(NamedGraph {
            name: merged_graph_name,
            edges: merged_edges,
            properties: merged_properties,
            creation_time: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64),
        })
    }

    /// Merge duplicate edges according to the specified combination strategy
    fn merge_duplicate_edges(
        &self,
        edges: Vec<MultiGraphEdge>,
        combination: &EdgeWeightCombination,
    ) -> DataFusionResult<Vec<MultiGraphEdge>> {
        let mut edge_groups: HashMap<(String, String), Vec<MultiGraphEdge>> = HashMap::new();

        // Group edges by (source, target) pair
        for edge in edges {
            let key = (edge.source.clone(), edge.target.clone());
            edge_groups.entry(key).or_insert_with(Vec::new).push(edge);
        }

        let mut merged_edges = Vec::new();

        for ((source, target), group) in edge_groups {
            if group.len() == 1 {
                // No duplicates, keep as is
                merged_edges.push(group.into_iter().next().unwrap());
            } else {
                // Merge duplicates
                let combined_weight = match combination {
                    EdgeWeightCombination::Sum => group.iter().map(|e| e.weight).sum(),
                    EdgeWeightCombination::Average => {
                        group.iter().map(|e| e.weight).sum::<f64>() / group.len() as f64
                    }
                    EdgeWeightCombination::Maximum => {
                        group.iter().map(|e| e.weight).fold(f64::NEG_INFINITY, f64::max)
                    }
                    EdgeWeightCombination::Minimum => {
                        group.iter().map(|e| e.weight).fold(f64::INFINITY, f64::min)
                    }
                    EdgeWeightCombination::KeepFirst => group.first().unwrap().weight,
                    EdgeWeightCombination::KeepLast => group.last().unwrap().weight,
                };

                // Combine graph names
                let combined_graph_names = group
                    .iter()
                    .map(|e| e.graph_name.as_str())
                    .collect::<Vec<_>>()
                    .join(",");

                // Merge properties
                let mut combined_properties = HashMap::new();
                for edge in &group {
                    for (key, value) in &edge.properties {
                        let prefixed_key = format!("{}_{}", edge.graph_name, key);
                        combined_properties.insert(prefixed_key, value.clone());
                    }
                }

                let merged_edge = MultiGraphEdge {
                    source,
                    target,
                    weight: combined_weight,
                    graph_name: combined_graph_names,
                    edge_type: group.first().unwrap().edge_type.clone(),
                    properties: combined_properties,
                };

                merged_edges.push(merged_edge);
            }
        }

        Ok(merged_edges)
    }

    /// Query edges across multiple graphs
    pub fn query_multi_graph(
        &self,
        graphs: &[NamedGraph],
        graph_names: Option<&[String]>,
        edge_filter: Option<Box<dyn Fn(&MultiGraphEdge) -> bool>>,
    ) -> DataFusionResult<Vec<MultiGraphEdge>> {
        let mut result_edges = Vec::new();

        for graph in graphs {
            // Filter by graph names if specified
            if let Some(names) = graph_names {
                if !names.contains(&graph.name) {
                    continue;
                }
            }

            // Apply edge filter if provided
            let filtered_edges = if let Some(ref filter) = edge_filter {
                graph.edges.iter().filter(|e| filter(e)).cloned().collect()
            } else {
                graph.edges.clone()
            };

            result_edges.extend(filtered_edges);
        }

        Ok(result_edges)
    }

    /// Find cross-graph connections (edges that span multiple graphs)
    pub fn find_cross_graph_connections(
        &self,
        graphs: &[NamedGraph],
    ) -> DataFusionResult<Vec<(String, String, Vec<String>)>> {
        let mut node_to_graphs: HashMap<String, Vec<String>> = HashMap::new();

        // Map each node to the graphs it appears in
        for graph in graphs {
            for edge in &graph.edges {
                node_to_graphs
                    .entry(edge.source.clone())
                    .or_insert_with(Vec::new)
                    .push(graph.name.clone());
                node_to_graphs
                    .entry(edge.target.clone())
                    .or_insert_with(Vec::new)
                    .push(graph.name.clone());
            }
        }

        // Find nodes that appear in multiple graphs
        let mut cross_graph_connections = Vec::new();
        for (node, graph_list) in node_to_graphs {
            if graph_list.len() > 1 {
                // Remove duplicates and sort
                let mut unique_graphs = graph_list;
                unique_graphs.sort();
                unique_graphs.dedup();
                
                if unique_graphs.len() > 1 {
                    cross_graph_connections.push((
                        "cross_graph_node".to_string(),
                        node,
                        unique_graphs,
                    ));
                }
            }
        }

        Ok(cross_graph_connections)
    }

    /// Calculate graph similarity between two graphs
    pub fn calculate_graph_similarity(
        &self,
        graph1: &NamedGraph,
        graph2: &NamedGraph,
    ) -> DataFusionResult<f64> {
        // Simple Jaccard similarity based on shared edges
        let edges1: std::collections::HashSet<(String, String)> = graph1
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect();

        let edges2: std::collections::HashSet<(String, String)> = graph2
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect();

        let intersection_size = edges1.intersection(&edges2).count();
        let union_size = edges1.union(&edges2).count();

        if union_size == 0 {
            Ok(0.0)
        } else {
            Ok(intersection_size as f64 / union_size as f64)
        }
    }

    /// Get graph statistics for multiple graphs
    pub fn get_multi_graph_statistics(
        &self,
        graphs: &[NamedGraph],
    ) -> DataFusionResult<HashMap<String, GraphStatistics>> {
        let mut stats = HashMap::new();

        for graph in graphs {
            let graph_stats = self.calculate_graph_statistics(graph)?;
            stats.insert(graph.name.clone(), graph_stats);
        }

        Ok(stats)
    }

    /// Calculate statistics for a single graph
    fn calculate_graph_statistics(&self, graph: &NamedGraph) -> DataFusionResult<GraphStatistics> {
        let edge_count = graph.edges.len();
        
        let mut nodes = std::collections::HashSet::new();
        let mut total_weight = 0.0;
        
        for edge in &graph.edges {
            nodes.insert(edge.source.clone());
            nodes.insert(edge.target.clone());
            total_weight += edge.weight;
        }

        let node_count = nodes.len();
        let average_weight = if edge_count > 0 {
            total_weight / edge_count as f64
        } else {
            0.0
        };

        let density = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };

        Ok(GraphStatistics {
            node_count,
            edge_count,
            average_edge_weight: average_weight,
            density,
            total_weight,
        })
    }
}

/// Statistics for a graph
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_edge_weight: f64,
    pub density: f64,
    pub total_weight: f64,
}

impl Default for MultiGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common multi-graph operations
pub struct MultiGraphQueries;

impl MultiGraphQueries {
    /// Find the largest graph by edge count
    pub fn find_largest_graph(graphs: &[NamedGraph]) -> Option<&NamedGraph> {
        graphs.iter().max_by_key(|g| g.edges.len())
    }

    /// Find graphs containing a specific node
    pub fn find_graphs_with_node<'a>(graphs: &'a [NamedGraph], node_id: &str) -> Vec<&'a NamedGraph> {
        graphs
            .iter()
            .filter(|graph| {
                graph.edges.iter().any(|edge| 
                    edge.source == node_id || edge.target == node_id
                )
            })
            .collect()
    }

    /// Get all unique nodes across multiple graphs
    pub fn get_all_unique_nodes(graphs: &[NamedGraph]) -> Vec<String> {
        let mut nodes = std::collections::HashSet::new();
        
        for graph in graphs {
            for edge in &graph.edges {
                nodes.insert(edge.source.clone());
                nodes.insert(edge.target.clone());
            }
        }

        nodes.into_iter().collect()
    }

    /// Find overlapping nodes between graphs
    pub fn find_overlapping_nodes(graph1: &NamedGraph, graph2: &NamedGraph) -> Vec<String> {
        let nodes1: std::collections::HashSet<String> = graph1
            .edges
            .iter()
            .flat_map(|e| vec![e.source.clone(), e.target.clone()])
            .collect();

        let nodes2: std::collections::HashSet<String> = graph2
            .edges
            .iter()
            .flat_map(|e| vec![e.source.clone(), e.target.clone()])
            .collect();

        nodes1.intersection(&nodes2).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph(name: &str, edges: Vec<(String, String, f64)>) -> NamedGraph {
        let multi_edges = edges
            .into_iter()
            .map(|(source, target, weight)| MultiGraphEdge {
                source,
                target,
                weight,
                graph_name: name.to_string(),
                edge_type: None,
                properties: HashMap::new(),
            })
            .collect();

        NamedGraph {
            name: name.to_string(),
            edges: multi_edges,
            properties: HashMap::new(),
            creation_time: Some(1704067200),
        }
    }

    #[test]
    fn test_create_graph() {
        let processor = MultiGraphProcessor::new();
        
        let edges = vec![
            MultiGraphEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 1.0,
                graph_name: "test_graph".to_string(),
                edge_type: None,
                properties: HashMap::new(),
            }
        ];

        let graph = processor.create_graph("test_graph", edges, None).unwrap();
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_merge_graphs() {
        let processor = MultiGraphProcessor::new();
        
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("C".to_string(), "D".to_string(), 3.0),
            ("D".to_string(), "A".to_string(), 4.0),
        ]);

        let config = MultiGraphConfig::default();
        let merged = processor.merge_graphs(&[graph1, graph2], &config).unwrap();
        
        assert_eq!(merged.edges.len(), 4);
        assert!(merged.name.contains("merged"));
    }

    #[test]
    fn test_merge_duplicate_edges() {
        let processor = MultiGraphProcessor::new();
        
        let edges = vec![
            MultiGraphEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 1.0,
                graph_name: "graph1".to_string(),
                edge_type: None,
                properties: HashMap::new(),
            },
            MultiGraphEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 3.0,
                graph_name: "graph2".to_string(),
                edge_type: None,
                properties: HashMap::new(),
            },
        ];

        let merged = processor.merge_duplicate_edges(edges, &EdgeWeightCombination::Average).unwrap();
        
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].weight, 2.0); // Average of 1.0 and 3.0
        assert!(merged[0].graph_name.contains("graph1") && merged[0].graph_name.contains("graph2"));
    }

    #[test]
    fn test_query_multi_graph() {
        let processor = MultiGraphProcessor::new();
        
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("C".to_string(), "D".to_string(), 3.0),
            ("D".to_string(), "A".to_string(), 4.0),
        ]);

        // Query specific graphs
        let result = processor.query_multi_graph(
            &[graph1.clone(), graph2.clone()],
            Some(&vec!["graph1".to_string()]),
            None,
        ).unwrap();
        
        assert_eq!(result.len(), 2); // Only edges from graph1

        // Query with edge filter
        let result_filtered = processor.query_multi_graph(
            &[graph1, graph2],
            None,
            Some(Box::new(|edge| edge.weight > 2.0)),
        ).unwrap();
        
        assert_eq!(result_filtered.len(), 2); // Edges with weight > 2.0
    }

    #[test]
    fn test_find_cross_graph_connections() {
        let processor = MultiGraphProcessor::new();
        
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("C".to_string(), "D".to_string(), 3.0), // C appears in both graphs
            ("D".to_string(), "E".to_string(), 4.0),
        ]);

        let connections = processor.find_cross_graph_connections(&[graph1, graph2]).unwrap();
        
        assert!(!connections.is_empty());
        // Should find node C as it appears in both graphs
        assert!(connections.iter().any(|(_, node, _)| node == "C"));
    }

    #[test]
    fn test_calculate_graph_similarity() {
        let processor = MultiGraphProcessor::new();
        
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("A".to_string(), "B".to_string(), 1.0), // Same edge
            ("C".to_string(), "D".to_string(), 3.0), // Different edge
        ]);

        let similarity = processor.calculate_graph_similarity(&graph1, &graph2).unwrap();
        
        // Jaccard similarity: 1 shared edge / 3 total unique edges = 1/3 ≈ 0.33
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_get_multi_graph_statistics() {
        let processor = MultiGraphProcessor::new();
        
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("C".to_string(), "D".to_string(), 3.0),
        ]);

        let stats = processor.get_multi_graph_statistics(&[graph1, graph2]).unwrap();
        
        assert_eq!(stats.len(), 2);
        assert!(stats.contains_key("graph1"));
        assert!(stats.contains_key("graph2"));
        
        let graph1_stats = &stats["graph1"];
        assert_eq!(graph1_stats.edge_count, 2);
        assert_eq!(graph1_stats.node_count, 3); // A, B, C
    }

    #[test]
    fn test_multi_graph_queries() {
        let graph1 = create_test_graph("graph1", vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
        ]);

        let graph2 = create_test_graph("graph2", vec![
            ("C".to_string(), "D".to_string(), 3.0),
            ("D".to_string(), "E".to_string(), 4.0),
        ]);

        let graphs = vec![graph1, graph2];

        // Test largest graph
        let largest = MultiGraphQueries::find_largest_graph(&graphs);
        assert!(largest.is_some());
        assert_eq!(largest.unwrap().edges.len(), 2);

        // Test find graphs with node
        let graphs_with_c = MultiGraphQueries::find_graphs_with_node(&graphs, "C");
        assert_eq!(graphs_with_c.len(), 2); // C appears in both graphs

        // Test all unique nodes
        let all_nodes = MultiGraphQueries::get_all_unique_nodes(&graphs);
        assert!(all_nodes.len() >= 5); // A, B, C, D, E

        // Test overlapping nodes
        let overlapping = MultiGraphQueries::find_overlapping_nodes(&graphs[0], &graphs[1]);
        assert!(overlapping.contains(&"C".to_string()));
    }

    #[test]
    fn test_edge_weight_combinations() {
        let processor = MultiGraphProcessor::new();
        
        let edges = vec![
            MultiGraphEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 2.0,
                graph_name: "graph1".to_string(),
                edge_type: None,
                properties: HashMap::new(),
            },
            MultiGraphEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 4.0,
                graph_name: "graph2".to_string(),
                edge_type: None,
                properties: HashMap::new(),
            },
        ];

        // Test different combination strategies
        let sum_result = processor.merge_duplicate_edges(edges.clone(), &EdgeWeightCombination::Sum).unwrap();
        assert_eq!(sum_result[0].weight, 6.0);

        let max_result = processor.merge_duplicate_edges(edges.clone(), &EdgeWeightCombination::Maximum).unwrap();
        assert_eq!(max_result[0].weight, 4.0);

        let min_result = processor.merge_duplicate_edges(edges.clone(), &EdgeWeightCombination::Minimum).unwrap();
        assert_eq!(min_result[0].weight, 2.0);

        let first_result = processor.merge_duplicate_edges(edges.clone(), &EdgeWeightCombination::KeepFirst).unwrap();
        assert_eq!(first_result[0].weight, 2.0);

        let last_result = processor.merge_duplicate_edges(edges, &EdgeWeightCombination::KeepLast).unwrap();
        assert_eq!(last_result[0].weight, 4.0);
    }
}