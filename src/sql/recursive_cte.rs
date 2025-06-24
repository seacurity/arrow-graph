use datafusion::error::Result as DataFusionResult;
use std::collections::{HashMap, HashSet, VecDeque};

/// Recursive CTE (Common Table Expression) processor for graph path queries
/// Supports patterns like: WITH RECURSIVE paths AS (...) SELECT ... FROM paths
pub struct RecursiveCteProcessor;

/// Represents a path in the graph
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPath {
    pub nodes: Vec<String>,
    pub total_weight: f64,
    pub depth: usize,
}

/// Configuration for recursive path queries
#[derive(Debug, Clone)]
pub struct RecursiveQueryConfig {
    pub max_depth: usize,
    pub include_cycles: bool,
    pub weight_threshold: Option<f64>,
    pub max_paths: Option<usize>,
}

impl Default for RecursiveQueryConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            include_cycles: false,
            weight_threshold: None,
            max_paths: Some(1000),
        }
    }
}

impl RecursiveCteProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Process a recursive path query
    /// Example: Find all paths from node A to node B with max depth 5
    pub fn find_recursive_paths(
        &self,
        edges: &[(String, String, f64)], // (source, target, weight)
        start_node: &str,
        end_node: Option<&str>,
        config: &RecursiveQueryConfig,
    ) -> DataFusionResult<Vec<GraphPath>> {
        let mut paths = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited_in_path = HashSet::new();

        // Build adjacency map for efficient lookup
        let mut adjacency_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for (source, target, weight) in edges {
            adjacency_map
                .entry(source.clone())
                .or_insert_with(Vec::new)
                .push((target.clone(), *weight));
        }

        // Initialize with starting node
        let initial_path = GraphPath {
            nodes: vec![start_node.to_string()],
            total_weight: 0.0,
            depth: 0,
        };
        queue.push_back((initial_path, visited_in_path.clone()));

        while let Some((current_path, mut path_visited)) = queue.pop_front() {
            // Check if we've reached the target (if specified)
            if let Some(target) = end_node {
                if current_path.nodes.last() == Some(&target.to_string()) {
                    paths.push(current_path.clone());
                    if let Some(max_paths) = config.max_paths {
                        if paths.len() >= max_paths {
                            break;
                        }
                    }
                    continue;
                }
            } else {
                // If no specific target, include all paths up to max depth
                paths.push(current_path.clone());
                if let Some(max_paths) = config.max_paths {
                    if paths.len() >= max_paths {
                        break;
                    }
                }
            }

            // Check depth limit
            if current_path.depth >= config.max_depth {
                continue;
            }

            // Get current node
            let current_node = current_path.nodes.last().unwrap();

            // Explore neighbors
            if let Some(neighbors) = adjacency_map.get(current_node) {
                for (neighbor, weight) in neighbors {
                    // Check for cycles
                    if !config.include_cycles && current_path.nodes.contains(neighbor) {
                        continue;
                    }

                    // Check weight threshold
                    if let Some(threshold) = config.weight_threshold {
                        if current_path.total_weight + weight > threshold {
                            continue;
                        }
                    }

                    // Create new path
                    let mut new_nodes = current_path.nodes.clone();
                    new_nodes.push(neighbor.clone());

                    let new_path = GraphPath {
                        nodes: new_nodes,
                        total_weight: current_path.total_weight + weight,
                        depth: current_path.depth + 1,
                    };

                    // Update visited set for this path
                    let mut new_visited = path_visited.clone();
                    new_visited.insert(neighbor.clone());

                    queue.push_back((new_path, new_visited));
                }
            }
        }

        Ok(paths)
    }

    /// Find shortest paths using recursive approach (for CTE queries)
    pub fn find_shortest_paths(
        &self,
        edges: &[(String, String, f64)],
        start_node: &str,
        end_node: &str,
        max_depth: usize,
    ) -> DataFusionResult<Vec<GraphPath>> {
        let config = RecursiveQueryConfig {
            max_depth,
            include_cycles: false,
            weight_threshold: None,
            max_paths: Some(10), // Limit to top 10 shortest paths
        };

        let mut all_paths = self.find_recursive_paths(edges, start_node, Some(end_node), &config)?;

        // Sort by total weight to get shortest paths first
        all_paths.sort_by(|a, b| a.total_weight.partial_cmp(&b.total_weight).unwrap_or(std::cmp::Ordering::Equal));

        Ok(all_paths)
    }

    /// Find all reachable nodes from a starting node (for CTE queries)
    pub fn find_reachable_nodes(
        &self,
        edges: &[(String, String, f64)],
        start_node: &str,
        max_depth: usize,
    ) -> DataFusionResult<Vec<String>> {
        let config = RecursiveQueryConfig {
            max_depth,
            include_cycles: false,
            weight_threshold: None,
            max_paths: None, // No limit on paths for reachability
        };

        let paths = self.find_recursive_paths(edges, start_node, None, &config)?;

        // Collect all unique nodes reachable from start
        let mut reachable: HashSet<String> = HashSet::new();
        for path in paths {
            for node in path.nodes {
                reachable.insert(node);
            }
        }

        // Remove the starting node itself
        reachable.remove(start_node);

        Ok(reachable.into_iter().collect())
    }

    /// Process a WITH RECURSIVE query (simplified parser)
    /// Example: "WITH RECURSIVE paths(source, target, depth) AS (
    ///            SELECT source, target, 1 FROM edges WHERE source = 'A'
    ///            UNION ALL
    ///            SELECT e.source, e.target, p.depth + 1 
    ///            FROM edges e JOIN paths p ON e.source = p.target 
    ///            WHERE p.depth < 5
    ///          ) SELECT * FROM paths"
    pub fn parse_and_execute_recursive_cte(
        &self,
        _cte_query: &str,
        edges: &[(String, String, f64)],
    ) -> DataFusionResult<Vec<GraphPath>> {
        // This is a simplified implementation
        // A full implementation would parse the actual SQL CTE syntax
        
        // For now, return paths from a default starting node
        let config = RecursiveQueryConfig::default();
        
        // Use first edge's source as starting point if available
        if let Some((start_node, _, _)) = edges.first() {
            self.find_recursive_paths(edges, start_node, None, &config)
        } else {
            Ok(Vec::new())
        }
    }
}

impl Default for RecursiveCteProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common recursive graph queries
pub struct RecursiveGraphQueries;

impl RecursiveGraphQueries {
    /// Standard "find all paths" CTE query
    pub fn all_paths_query(
        processor: &RecursiveCteProcessor,
        edges: &[(String, String, f64)],
        start: &str,
        end: &str,
        max_depth: usize,
    ) -> DataFusionResult<Vec<GraphPath>> {
        let config = RecursiveQueryConfig {
            max_depth,
            include_cycles: false,
            weight_threshold: None,
            max_paths: Some(100),
        };
        
        processor.find_recursive_paths(edges, start, Some(end), &config)
    }

    /// Standard "find connected component" CTE query
    pub fn connected_component_query(
        processor: &RecursiveCteProcessor,
        edges: &[(String, String, f64)],
        start: &str,
    ) -> DataFusionResult<Vec<String>> {
        processor.find_reachable_nodes(edges, start, 50) // Max depth 50 for components
    }

    /// Standard "shortest path" CTE query
    pub fn shortest_path_query(
        processor: &RecursiveCteProcessor,
        edges: &[(String, String, f64)],
        start: &str,
        end: &str,
    ) -> DataFusionResult<Option<GraphPath>> {
        let paths = processor.find_shortest_paths(edges, start, end, 20)?;
        Ok(paths.into_iter().next())
    }

    /// Standard "paths within distance" CTE query
    pub fn paths_within_distance_query(
        processor: &RecursiveCteProcessor,
        edges: &[(String, String, f64)],
        start: &str,
        max_distance: f64,
    ) -> DataFusionResult<Vec<GraphPath>> {
        let config = RecursiveQueryConfig {
            max_depth: 20,
            include_cycles: false,
            weight_threshold: Some(max_distance),
            max_paths: Some(1000),
        };
        
        processor.find_recursive_paths(edges, start, None, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_edges() -> Vec<(String, String, f64)> {
        vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 2.0),
            ("C".to_string(), "D".to_string(), 1.0),
            ("A".to_string(), "C".to_string(), 4.0),
            ("B".to_string(), "D".to_string(), 3.0),
        ]
    }

    #[test]
    fn test_find_recursive_paths() {
        let processor = RecursiveCteProcessor::new();
        let edges = create_test_edges();
        let config = RecursiveQueryConfig::default();

        let paths = processor.find_recursive_paths(&edges, "A", Some("D"), &config).unwrap();
        
        assert!(!paths.is_empty());
        
        // Should find multiple paths from A to D
        let path_endpoints: Vec<_> = paths.iter()
            .map(|p| (p.nodes.first().unwrap(), p.nodes.last().unwrap()))
            .collect();
        
        for (start, end) in path_endpoints {
            assert_eq!(start, "A");
            assert_eq!(end, "D");
        }
    }

    #[test]
    fn test_find_shortest_paths() {
        let processor = RecursiveCteProcessor::new();
        let edges = create_test_edges();

        let paths = processor.find_shortest_paths(&edges, "A", "D", 5).unwrap();
        
        assert!(!paths.is_empty());
        
        // Paths should be sorted by weight
        for i in 1..paths.len() {
            assert!(paths[i-1].total_weight <= paths[i].total_weight);
        }
    }

    #[test]
    fn test_find_reachable_nodes() {
        let processor = RecursiveCteProcessor::new();
        let edges = create_test_edges();

        let reachable = processor.find_reachable_nodes(&edges, "A", 5).unwrap();
        
        assert!(reachable.contains(&"B".to_string()));
        assert!(reachable.contains(&"C".to_string()));
        assert!(reachable.contains(&"D".to_string()));
        assert!(!reachable.contains(&"A".to_string())); // Shouldn't include start node itself
    }

    #[test]
    fn test_recursive_query_utilities() {
        let processor = RecursiveCteProcessor::new();
        let edges = create_test_edges();

        // Test all paths query
        let all_paths = RecursiveGraphQueries::all_paths_query(&processor, &edges, "A", "D", 5).unwrap();
        assert!(!all_paths.is_empty());

        // Test connected component query
        let component = RecursiveGraphQueries::connected_component_query(&processor, &edges, "A").unwrap();
        assert!(component.len() >= 3); // Should reach B, C, D

        // Test shortest path query
        let shortest = RecursiveGraphQueries::shortest_path_query(&processor, &edges, "A", "D").unwrap();
        assert!(shortest.is_some());

        // Test paths within distance query
        let within_distance = RecursiveGraphQueries::paths_within_distance_query(&processor, &edges, "A", 5.0).unwrap();
        assert!(!within_distance.is_empty());
    }

    #[test]
    fn test_cycle_detection() {
        let processor = RecursiveCteProcessor::new();
        
        // Create edges with a cycle
        let edges = vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 1.0),
            ("C".to_string(), "A".to_string(), 1.0), // Creates cycle
        ];

        let config = RecursiveQueryConfig {
            max_depth: 5,
            include_cycles: false,
            weight_threshold: None,
            max_paths: Some(100),
        };

        let paths = processor.find_recursive_paths(&edges, "A", None, &config).unwrap();
        
        // Without cycles, should not have infinite paths
        assert!(paths.len() < 100);
        
        // All paths should have unique nodes (no cycles)
        for path in paths {
            let mut unique_nodes = HashSet::new();
            for node in &path.nodes {
                assert!(unique_nodes.insert(node.clone()), "Found cycle in path: {:?}", path.nodes);
            }
        }
    }

    #[test]
    fn test_max_depth_limit() {
        let processor = RecursiveCteProcessor::new();
        let edges = create_test_edges();
        
        let config = RecursiveQueryConfig {
            max_depth: 2,
            include_cycles: false,
            weight_threshold: None,
            max_paths: None,
        };

        let paths = processor.find_recursive_paths(&edges, "A", None, &config).unwrap();
        
        // All paths should respect max depth
        for path in paths {
            assert!(path.depth <= 2, "Path exceeded max depth: {:?}", path);
        }
    }
}