use datafusion::error::Result as DataFusionResult;
use std::collections::HashMap;

/// Temporal graph query processor for time-based edge filtering
/// Supports queries like: SELECT * FROM edges WHERE edge_timestamp BETWEEN '2024-01-01' AND '2024-12-31'
pub struct TemporalGraphProcessor;

/// Represents a temporal edge with timestamp information
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub timestamp: i64, // Unix timestamp
    pub properties: HashMap<String, String>,
}

/// Time window configuration for temporal queries
#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub start_time: i64,  // Unix timestamp
    pub end_time: i64,    // Unix timestamp
    pub window_size: Option<i64>, // Optional sliding window size in seconds
}

/// Configuration for temporal graph queries
#[derive(Debug, Clone)]
pub struct TemporalQueryConfig {
    pub time_window: TimeWindow,
    pub include_edge_properties: bool,
    pub aggregate_by_time_bucket: Option<i64>, // Bucket size in seconds for temporal aggregation
}

impl TemporalGraphProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Filter edges by time window
    /// Example: Find all edges that occurred between two timestamps
    pub fn filter_edges_by_time(
        &self,
        edges: &[TemporalEdge],
        time_window: &TimeWindow,
    ) -> DataFusionResult<Vec<TemporalEdge>> {
        let filtered_edges = edges
            .iter()
            .filter(|edge| {
                edge.timestamp >= time_window.start_time && edge.timestamp <= time_window.end_time
            })
            .cloned()
            .collect();

        Ok(filtered_edges)
    }

    /// Get graph snapshot at a specific timestamp
    /// Returns all edges that were active at the given time
    pub fn get_graph_snapshot(
        &self,
        edges: &[TemporalEdge],
        snapshot_time: i64,
    ) -> DataFusionResult<Vec<TemporalEdge>> {
        let snapshot_edges = edges
            .iter()
            .filter(|edge| edge.timestamp <= snapshot_time)
            .cloned()
            .collect();

        Ok(snapshot_edges)
    }

    /// Find temporal paths - paths that respect time ordering
    /// Each edge in the path must have a timestamp >= previous edge
    pub fn find_temporal_paths(
        &self,
        edges: &[TemporalEdge],
        start_node: &str,
        end_node: &str,
        time_window: &TimeWindow,
    ) -> DataFusionResult<Vec<Vec<TemporalEdge>>> {
        // Filter edges within time window first
        let window_edges = self.filter_edges_by_time(edges, time_window)?;
        
        // Build adjacency map with temporal ordering
        let mut adjacency_map: HashMap<String, Vec<&TemporalEdge>> = HashMap::new();
        for edge in &window_edges {
            adjacency_map
                .entry(edge.source.clone())
                .or_insert_with(Vec::new)
                .push(edge);
        }

        // Sort edges by timestamp for each node
        for edges_list in adjacency_map.values_mut() {
            edges_list.sort_by_key(|edge| edge.timestamp);
        }

        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        
        self.dfs_temporal_paths(
            &adjacency_map,
            start_node,
            end_node,
            &mut current_path,
            &mut paths,
            time_window.start_time,
        );

        Ok(paths)
    }

    /// Depth-first search for temporal paths
    fn dfs_temporal_paths(
        &self,
        adjacency_map: &HashMap<String, Vec<&TemporalEdge>>,
        current_node: &str,
        target_node: &str,
        current_path: &mut Vec<TemporalEdge>,
        all_paths: &mut Vec<Vec<TemporalEdge>>,
        min_timestamp: i64,
    ) {
        if current_node == target_node {
            all_paths.push(current_path.clone());
            return;
        }

        if let Some(edges) = adjacency_map.get(current_node) {
            for edge in edges {
                // Ensure temporal ordering: each edge must happen after the previous one
                if edge.timestamp >= min_timestamp {
                    current_path.push((*edge).clone());
                    
                    self.dfs_temporal_paths(
                        adjacency_map,
                        &edge.target,
                        target_node,
                        current_path,
                        all_paths,
                        edge.timestamp,
                    );
                    
                    current_path.pop();
                }
            }
        }
    }

    /// Aggregate edges by time buckets
    /// Example: Count edges per hour, day, week, etc.
    pub fn aggregate_by_time_bucket(
        &self,
        edges: &[TemporalEdge],
        bucket_size_seconds: i64,
    ) -> DataFusionResult<HashMap<i64, usize>> {
        let mut buckets: HashMap<i64, usize> = HashMap::new();

        for edge in edges {
            let bucket = (edge.timestamp / bucket_size_seconds) * bucket_size_seconds;
            *buckets.entry(bucket).or_insert(0) += 1;
        }

        Ok(buckets)
    }

    /// Find active nodes at a specific time
    /// Returns nodes that had at least one edge within the time window before the timestamp
    pub fn get_active_nodes_at_time(
        &self,
        edges: &[TemporalEdge],
        timestamp: i64,
        lookback_window: i64, // How far back to look for activity
    ) -> DataFusionResult<Vec<String>> {
        let window_start = timestamp - lookback_window;
        let time_window = TimeWindow {
            start_time: window_start,
            end_time: timestamp,
            window_size: Some(lookback_window),
        };

        let active_edges = self.filter_edges_by_time(edges, &time_window)?;
        
        let mut active_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for edge in active_edges {
            active_nodes.insert(edge.source);
            active_nodes.insert(edge.target);
        }

        Ok(active_nodes.into_iter().collect())
    }

    /// Calculate temporal centrality - how central a node is within a time window
    pub fn calculate_temporal_centrality(
        &self,
        edges: &[TemporalEdge],
        node_id: &str,
        time_window: &TimeWindow,
    ) -> DataFusionResult<f64> {
        let window_edges = self.filter_edges_by_time(edges, time_window)?;
        
        let mut degree = 0;
        let total_edges = window_edges.len();

        for edge in &window_edges {
            if edge.source == node_id || edge.target == node_id {
                degree += 1;
            }
        }

        if total_edges == 0 {
            Ok(0.0)
        } else {
            Ok(degree as f64 / total_edges as f64)
        }
    }

    /// Process a temporal SQL query (simplified parser)
    /// Example: "SELECT source, target FROM edges WHERE edge_timestamp BETWEEN 1704067200 AND 1735689600"
    pub fn parse_and_execute_temporal_query(
        &self,
        _query: &str,
        edges: &[TemporalEdge],
        config: &TemporalQueryConfig,
    ) -> DataFusionResult<Vec<TemporalEdge>> {
        // This is a simplified implementation
        // A full implementation would parse the actual SQL syntax
        
        self.filter_edges_by_time(edges, &config.time_window)
    }
}

impl Default for TemporalGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common temporal graph operations
pub struct TemporalGraphQueries;

impl TemporalGraphQueries {
    /// Standard "edges in last N days" query
    pub fn edges_in_last_n_days(
        processor: &TemporalGraphProcessor,
        edges: &[TemporalEdge],
        days: i64,
        current_time: i64,
    ) -> DataFusionResult<Vec<TemporalEdge>> {
        let seconds_per_day = 86400;
        let window_start = current_time - (days * seconds_per_day);
        
        let time_window = TimeWindow {
            start_time: window_start,
            end_time: current_time,
            window_size: Some(days * seconds_per_day),
        };

        processor.filter_edges_by_time(edges, &time_window)
    }

    /// Standard "temporal shortest path" query
    pub fn temporal_shortest_path(
        processor: &TemporalGraphProcessor,
        edges: &[TemporalEdge],
        start: &str,
        end: &str,
        time_window: &TimeWindow,
    ) -> DataFusionResult<Option<Vec<TemporalEdge>>> {
        let paths = processor.find_temporal_paths(edges, start, end, time_window)?;
        
        // Return the path with minimum total time duration
        let shortest_path = paths
            .into_iter()
            .min_by_key(|path| {
                if path.is_empty() {
                    0
                } else {
                    path.last().unwrap().timestamp - path.first().unwrap().timestamp
                }
            });

        Ok(shortest_path)
    }

    /// Standard "nodes active in period" query
    pub fn most_active_nodes_in_period(
        processor: &TemporalGraphProcessor,
        edges: &[TemporalEdge],
        time_window: &TimeWindow,
        top_n: usize,
    ) -> DataFusionResult<Vec<(String, usize)>> {
        let window_edges = processor.filter_edges_by_time(edges, time_window)?;
        
        let mut node_activity: HashMap<String, usize> = HashMap::new();
        
        for edge in window_edges {
            *node_activity.entry(edge.source).or_insert(0) += 1;
            *node_activity.entry(edge.target).or_insert(0) += 1;
        }

        let mut activity_vec: Vec<(String, usize)> = node_activity.into_iter().collect();
        activity_vec.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by activity descending
        activity_vec.truncate(top_n);

        Ok(activity_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_temporal_edges() -> Vec<TemporalEdge> {
        vec![
            TemporalEdge {
                source: "A".to_string(),
                target: "B".to_string(),
                weight: 1.0,
                timestamp: 1704067200, // 2024-01-01 00:00:00
                properties: HashMap::new(),
            },
            TemporalEdge {
                source: "B".to_string(),
                target: "C".to_string(),
                weight: 2.0,
                timestamp: 1704153600, // 2024-01-02 00:00:00
                properties: HashMap::new(),
            },
            TemporalEdge {
                source: "C".to_string(),
                target: "D".to_string(),
                weight: 1.0,
                timestamp: 1704240000, // 2024-01-03 00:00:00
                properties: HashMap::new(),
            },
            TemporalEdge {
                source: "A".to_string(),
                target: "C".to_string(),
                weight: 4.0,
                timestamp: 1704326400, // 2024-01-04 00:00:00
                properties: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_filter_edges_by_time() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        let time_window = TimeWindow {
            start_time: 1704067200, // 2024-01-01
            end_time: 1704240000,   // 2024-01-03
            window_size: None,
        };

        let filtered = processor.filter_edges_by_time(&edges, &time_window).unwrap();
        assert_eq!(filtered.len(), 3); // Should include first 3 edges
        
        // Last edge should be filtered out
        assert!(!filtered.iter().any(|e| e.timestamp == 1704326400));
    }

    #[test]
    fn test_get_graph_snapshot() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        let snapshot = processor.get_graph_snapshot(&edges, 1704153600).unwrap();
        assert_eq!(snapshot.len(), 2); // Should include first 2 edges
        
        let snapshot_later = processor.get_graph_snapshot(&edges, 1704326400).unwrap();
        assert_eq!(snapshot_later.len(), 4); // Should include all edges
    }

    #[test]
    fn test_find_temporal_paths() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        let time_window = TimeWindow {
            start_time: 1704067200,
            end_time: 1704326400,
            window_size: None,
        };

        let paths = processor.find_temporal_paths(&edges, "A", "D", &time_window).unwrap();
        assert!(!paths.is_empty());
        
        // Verify temporal ordering in paths
        for path in &paths {
            for i in 1..path.len() {
                assert!(path[i].timestamp >= path[i-1].timestamp);
            }
        }
    }

    #[test]
    fn test_aggregate_by_time_bucket() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        // Group by day (86400 seconds)
        let buckets = processor.aggregate_by_time_bucket(&edges, 86400).unwrap();
        
        assert!(buckets.len() >= 3); // Should have at least 3 different days
        
        // Each bucket should have at least 1 edge
        for count in buckets.values() {
            assert!(*count >= 1);
        }
    }

    #[test]
    fn test_get_active_nodes_at_time() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        let active_nodes = processor.get_active_nodes_at_time(
            &edges,
            1704240000, // 2024-01-03
            172800,     // 2 days lookback
        ).unwrap();
        
        // Should include nodes A, B, C, D
        assert!(active_nodes.contains(&"A".to_string()));
        assert!(active_nodes.contains(&"B".to_string()));
        assert!(active_nodes.contains(&"C".to_string()));
    }

    #[test]
    fn test_calculate_temporal_centrality() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        let time_window = TimeWindow {
            start_time: 1704067200,
            end_time: 1704326400,
            window_size: None,
        };

        let centrality_a = processor.calculate_temporal_centrality(&edges, "A", &time_window).unwrap();
        let centrality_d = processor.calculate_temporal_centrality(&edges, "D", &time_window).unwrap();
        
        // A should have higher centrality than D (appears in more edges)
        assert!(centrality_a > centrality_d);
    }

    #[test]
    fn test_temporal_query_utilities() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        // Test edges in last N days
        let recent_edges = TemporalGraphQueries::edges_in_last_n_days(
            &processor,
            &edges,
            5, // last 5 days
            1704326400, // from 2024-01-04
        ).unwrap();
        assert_eq!(recent_edges.len(), 4); // All edges should be included
        
        // Test temporal shortest path
        let time_window = TimeWindow {
            start_time: 1704067200,
            end_time: 1704326400,
            window_size: None,
        };
        
        let shortest_path = TemporalGraphQueries::temporal_shortest_path(
            &processor,
            &edges,
            "A",
            "D",
            &time_window,
        ).unwrap();
        assert!(shortest_path.is_some());
        
        // Test most active nodes
        let active_nodes = TemporalGraphQueries::most_active_nodes_in_period(
            &processor,
            &edges,
            &time_window,
            3, // top 3
        ).unwrap();
        assert!(active_nodes.len() <= 3);
        assert!(!active_nodes.is_empty());
    }

    #[test]
    fn test_empty_time_window() {
        let processor = TemporalGraphProcessor::new();
        let edges = create_test_temporal_edges();
        
        // Time window that excludes all edges
        let empty_window = TimeWindow {
            start_time: 1500000000, // Much earlier
            end_time: 1600000000,   // Still before any edges
            window_size: None,
        };

        let filtered = processor.filter_edges_by_time(&edges, &empty_window).unwrap();
        assert!(filtered.is_empty());
        
        let paths = processor.find_temporal_paths(&edges, "A", "D", &empty_window).unwrap();
        assert!(paths.is_empty());
    }
}