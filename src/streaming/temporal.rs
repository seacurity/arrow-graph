use crate::error::Result;
use crate::streaming::incremental::{IncrementalGraphProcessor, UpdateResult};
use arrow::array::Array;
use std::collections::{HashMap, VecDeque};

/// Time-based sliding window for graph evolution analysis
/// Maintains snapshots of graph state over time and provides temporal analytics
#[derive(Debug, Clone)]
pub struct SlidingWindowProcessor {
    window_size: std::time::Duration,
    max_snapshots: usize,
    snapshots: VecDeque<GraphSnapshot>,
    current_metrics: HashMap<String, f64>,
}

/// A snapshot of graph state at a specific point in time
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub timestamp: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub largest_component_size: usize,
    pub component_count: usize,
    pub clustering_coefficient: f64,
    pub custom_metrics: HashMap<String, f64>,
}

/// Configuration for temporal analysis
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    pub window_duration: std::time::Duration,
    pub snapshot_interval: std::time::Duration,
    pub max_snapshots: usize,
    pub track_custom_metrics: Vec<String>,
}

/// Temporal analytics results
#[derive(Debug, Clone)]
pub struct TemporalAnalytics {
    pub trend_analysis: HashMap<String, TrendInfo>,
    pub volatility_metrics: HashMap<String, f64>,
    pub change_points: Vec<ChangePoint>,
    pub evolution_patterns: Vec<EvolutionPattern>,
}

/// Information about a metric's trend over time
#[derive(Debug, Clone)]
pub struct TrendInfo {
    pub metric_name: String,
    pub direction: TrendDirection,
    pub slope: f64,
    pub correlation: f64,
    pub volatility: f64,
    pub recent_change: f64,
}

/// Direction of trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Detected change point in graph evolution
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub timestamp: u64,
    pub metric_name: String,
    pub before_value: f64,
    pub after_value: f64,
    pub change_magnitude: f64,
    pub confidence: f64,
}

/// Detected evolution pattern
#[derive(Debug, Clone)]
pub struct EvolutionPattern {
    pub pattern_type: PatternType,
    pub start_time: u64,
    pub end_time: u64,
    pub affected_metrics: Vec<String>,
    pub strength: f64,
}

/// Types of evolution patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Growth,          // Steady growth in graph size
    Decay,           // Steady decline in activity
    Burst,           // Sudden spike in activity
    Oscillation,     // Periodic changes
    PhaseTransition, // Fundamental change in structure
    Stabilization,   // Convergence to steady state
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_duration: std::time::Duration::from_secs(3600), // 1 hour
            snapshot_interval: std::time::Duration::from_secs(60), // 1 minute
            max_snapshots: 1000,
            track_custom_metrics: vec![
                "pagerank_entropy".to_string(),
                "edge_density_variance".to_string(),
                "community_modularity".to_string(),
            ],
        }
    }
}

impl SlidingWindowProcessor {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            window_size: config.window_duration,
            max_snapshots: config.max_snapshots,
            snapshots: VecDeque::new(),
            current_metrics: HashMap::new(),
        }
    }

    /// Default sliding window processor
    pub fn default() -> Self {
        Self::new(TemporalConfig::default())
    }

    /// Initialize with current graph state
    pub fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        let snapshot = self.create_snapshot(processor)?;
        self.snapshots.push_back(snapshot);
        Ok(())
    }

    /// Update window with new graph state
    pub fn update(&mut self, processor: &IncrementalGraphProcessor, _changes: &UpdateResult) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create new snapshot
        let snapshot = self.create_snapshot(processor)?;
        self.snapshots.push_back(snapshot);

        // Remove old snapshots outside window
        self.cleanup_old_snapshots(current_time);

        // Limit total snapshots
        while self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }

        Ok(())
    }

    /// Create a snapshot of current graph state
    fn create_snapshot(&self, processor: &IncrementalGraphProcessor) -> Result<GraphSnapshot> {
        let graph = processor.graph();
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        let density = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };

        let avg_degree = if node_count > 0 {
            (2 * edge_count) as f64 / node_count as f64
        } else {
            0.0
        };

        // Analyze components
        let components = self.analyze_components(processor)?;
        let largest_component_size = components.iter().map(|c| c.len()).max().unwrap_or(0);
        let component_count = components.len();

        // Estimate clustering coefficient
        let clustering_coefficient = self.estimate_clustering_coefficient(processor)?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(GraphSnapshot {
            timestamp,
            node_count,
            edge_count,
            density,
            avg_degree,
            largest_component_size,
            component_count,
            clustering_coefficient,
            custom_metrics: HashMap::new(), // TODO: Implement custom metrics
        })
    }

    /// Analyze connected components
    fn analyze_components(&self, processor: &IncrementalGraphProcessor) -> Result<Vec<Vec<String>>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        let nodes_batch = &graph.nodes;

        let mut components = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Get all nodes
        let mut nodes = Vec::new();
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }

        // Build adjacency list
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;

            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();

                adjacency.entry(source.clone()).or_insert_with(Vec::new).push(target.clone());
                adjacency.entry(target).or_insert_with(Vec::new).push(source);
            }
        }

        // Find components using DFS
        for node in nodes {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                let mut stack = vec![node.clone()];

                while let Some(current) = stack.pop() {
                    if visited.insert(current.clone()) {
                        component.push(current.clone());

                        if let Some(neighbors) = adjacency.get(&current) {
                            for neighbor in neighbors {
                                if !visited.contains(neighbor) {
                                    stack.push(neighbor.clone());
                                }
                            }
                        }
                    }
                }

                if !component.is_empty() {
                    components.push(component);
                }
            }
        }

        Ok(components)
    }

    /// Estimate clustering coefficient
    fn estimate_clustering_coefficient(&self, _processor: &IncrementalGraphProcessor) -> Result<f64> {
        // Simplified implementation - in practice would calculate actual clustering
        Ok(0.3) // Placeholder value
    }

    /// Remove snapshots outside the time window
    fn cleanup_old_snapshots(&mut self, current_time: u64) {
        let window_start = current_time.saturating_sub(self.window_size.as_secs());

        while let Some(front) = self.snapshots.front() {
            if front.timestamp < window_start {
                self.snapshots.pop_front();
            } else {
                break;
            }
        }
    }

    /// Perform temporal analysis on the sliding window
    pub fn analyze_temporal_patterns(&self) -> Result<TemporalAnalytics> {
        if self.snapshots.len() < 2 {
            return Ok(TemporalAnalytics {
                trend_analysis: HashMap::new(),
                volatility_metrics: HashMap::new(),
                change_points: Vec::new(),
                evolution_patterns: Vec::new(),
            });
        }

        let trend_analysis = self.analyze_trends()?;
        let volatility_metrics = self.calculate_volatility()?;
        let change_points = self.detect_change_points()?;
        let evolution_patterns = self.detect_evolution_patterns()?;

        Ok(TemporalAnalytics {
            trend_analysis,
            volatility_metrics,
            change_points,
            evolution_patterns,
        })
    }

    /// Analyze trends for each metric
    fn analyze_trends(&self) -> Result<HashMap<String, TrendInfo>> {
        let mut trends = HashMap::new();

        let metrics = ["node_count", "edge_count", "density", "avg_degree", "component_count"];

        for metric in &metrics {
            let values: Vec<f64> = self.snapshots.iter()
                .map(|s| self.get_metric_value(s, metric))
                .collect();

            if values.len() >= 2 {
                let trend_info = self.calculate_trend_info(metric, &values);
                trends.insert(metric.to_string(), trend_info);
            }
        }

        Ok(trends)
    }

    /// Get metric value from snapshot
    fn get_metric_value(&self, snapshot: &GraphSnapshot, metric: &str) -> f64 {
        match metric {
            "node_count" => snapshot.node_count as f64,
            "edge_count" => snapshot.edge_count as f64,
            "density" => snapshot.density,
            "avg_degree" => snapshot.avg_degree,
            "component_count" => snapshot.component_count as f64,
            "clustering_coefficient" => snapshot.clustering_coefficient,
            _ => snapshot.custom_metrics.get(metric).copied().unwrap_or(0.0),
        }
    }

    /// Calculate trend information for a metric
    fn calculate_trend_info(&self, metric_name: &str, values: &[f64]) -> TrendInfo {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        // Calculate linear regression slope
        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = x_values.iter().zip(values.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter()
            .map(|x| (x - x_mean).powi(2))
            .sum();

        let slope = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Calculate correlation coefficient
        let x_std = (x_values.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>() / n).sqrt();
        let y_std = (values.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n).sqrt();

        let correlation = if x_std > 1e-10 && y_std > 1e-10 {
            numerator / (n * x_std * y_std)
        } else {
            0.0
        };

        // Calculate volatility (standard deviation)
        let volatility = y_std;

        // Determine trend direction
        let direction = if slope.abs() < 0.01 {
            if volatility > y_mean * 0.1 {
                TrendDirection::Volatile
            } else {
                TrendDirection::Stable
            }
        } else if slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        // Recent change (last vs first value)
        let recent_change = if values.len() > 1 {
            let first = values[0];
            let last = values[values.len() - 1];
            if first.abs() > 1e-10 {
                (last - first) / first
            } else {
                0.0
            }
        } else {
            0.0
        };

        TrendInfo {
            metric_name: metric_name.to_string(),
            direction,
            slope,
            correlation,
            volatility,
            recent_change,
        }
    }

    /// Calculate volatility metrics
    fn calculate_volatility(&self) -> Result<HashMap<String, f64>> {
        let mut volatility = HashMap::new();

        let metrics = ["node_count", "edge_count", "density", "avg_degree"];

        for metric in &metrics {
            let values: Vec<f64> = self.snapshots.iter()
                .map(|s| self.get_metric_value(s, metric))
                .collect();

            if values.len() > 1 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let vol = variance.sqrt();

                volatility.insert(metric.to_string(), vol);
            }
        }

        Ok(volatility)
    }

    /// Detect change points in time series
    fn detect_change_points(&self) -> Result<Vec<ChangePoint>> {
        let mut change_points = Vec::new();

        let metrics = ["node_count", "edge_count", "density", "avg_degree"];

        for metric in &metrics {
            let values: Vec<f64> = self.snapshots.iter()
                .map(|s| self.get_metric_value(s, metric))
                .collect();

            if values.len() < 10 {
                continue; // Need enough data for change point detection
            }

            // Simple change point detection using moving averages
            let window_size = values.len() / 4; // Quarter of the data
            if window_size < 2 {
                continue;
            }

            for i in window_size..(values.len() - window_size) {
                let before_mean = values[(i - window_size)..i].iter().sum::<f64>() / window_size as f64;
                let after_mean = values[i..(i + window_size)].iter().sum::<f64>() / window_size as f64;

                let change_magnitude = (after_mean - before_mean).abs();
                let relative_change = if before_mean.abs() > 1e-10 {
                    change_magnitude / before_mean.abs()
                } else {
                    0.0
                };

                // Threshold for significant change
                if relative_change > 0.2 {
                    let confidence = (relative_change - 0.2) / 0.8; // Scale to 0-1
                    change_points.push(ChangePoint {
                        timestamp: self.snapshots[i].timestamp,
                        metric_name: metric.to_string(),
                        before_value: before_mean,
                        after_value: after_mean,
                        change_magnitude,
                        confidence: confidence.min(1.0),
                    });
                }
            }
        }

        Ok(change_points)
    }

    /// Detect evolution patterns
    fn detect_evolution_patterns(&self) -> Result<Vec<EvolutionPattern>> {
        let mut patterns = Vec::new();

        if self.snapshots.len() < 5 {
            return Ok(patterns);
        }

        // Detect growth patterns
        let node_values: Vec<f64> = self.snapshots.iter()
            .map(|s| s.node_count as f64)
            .collect();

        if self.is_growth_pattern(&node_values) {
            patterns.push(EvolutionPattern {
                pattern_type: PatternType::Growth,
                start_time: self.snapshots[0].timestamp,
                end_time: self.snapshots[self.snapshots.len() - 1].timestamp,
                affected_metrics: vec!["node_count".to_string(), "edge_count".to_string()],
                strength: self.calculate_pattern_strength(&node_values),
            });
        }

        // Detect burst patterns
        let edge_values: Vec<f64> = self.snapshots.iter()
            .map(|s| s.edge_count as f64)
            .collect();

        if let Some((start_idx, end_idx)) = self.detect_burst_pattern(&edge_values) {
            patterns.push(EvolutionPattern {
                pattern_type: PatternType::Burst,
                start_time: self.snapshots[start_idx].timestamp,
                end_time: self.snapshots[end_idx].timestamp,
                affected_metrics: vec!["edge_count".to_string(), "density".to_string()],
                strength: self.calculate_burst_strength(&edge_values, start_idx, end_idx),
            });
        }

        Ok(patterns)
    }

    /// Check if values show a growth pattern
    fn is_growth_pattern(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }

        let increasing_count = values.windows(2)
            .filter(|w| w[1] > w[0])
            .count();

        increasing_count as f64 / (values.len() - 1) as f64 > 0.7 // 70% increasing
    }

    /// Detect burst pattern (sudden spike followed by normalization)
    fn detect_burst_pattern(&self, values: &[f64]) -> Option<(usize, usize)> {
        if values.len() < 5 {
            return None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();

        let threshold = mean + 2.0 * std;

        for i in 1..(values.len() - 1) {
            if values[i] > threshold && values[i - 1] < threshold {
                // Found potential start of burst
                for j in (i + 1)..values.len() {
                    if values[j] < threshold {
                        return Some((i, j));
                    }
                }
            }
        }

        None
    }

    /// Calculate pattern strength
    fn calculate_pattern_strength(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let first = values[0];
        let last = values[values.len() - 1];

        if first.abs() > 1e-10 {
            ((last - first) / first).abs().min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate burst strength
    fn calculate_burst_strength(&self, values: &[f64], start_idx: usize, end_idx: usize) -> f64 {
        if start_idx >= values.len() || end_idx >= values.len() || start_idx >= end_idx {
            return 0.0;
        }

        let baseline = values[0..start_idx].iter().sum::<f64>() / start_idx.max(1) as f64;
        let peak = values[start_idx..=end_idx].iter().fold(0.0f64, |a, b| a.max(*b));

        if baseline > 1e-10 {
            ((peak - baseline) / baseline).min(2.0) / 2.0 // Normalize to 0-1
        } else {
            0.0
        }
    }

    /// Get current window statistics
    pub fn window_stats(&self) -> WindowStats {
        let duration = if let (Some(first), Some(last)) = (self.snapshots.front(), self.snapshots.back()) {
            last.timestamp - first.timestamp
        } else {
            0
        };

        WindowStats {
            snapshot_count: self.snapshots.len(),
            window_duration_seconds: duration,
            oldest_timestamp: self.snapshots.front().map(|s| s.timestamp).unwrap_or(0),
            newest_timestamp: self.snapshots.back().map(|s| s.timestamp).unwrap_or(0),
        }
    }

    /// Get recent snapshots
    pub fn recent_snapshots(&self, count: usize) -> Vec<&GraphSnapshot> {
        self.snapshots.iter().rev().take(count).collect()
    }

    /// Get metric evolution over time
    pub fn metric_evolution(&self, metric_name: &str) -> Vec<(u64, f64)> {
        self.snapshots.iter()
            .map(|s| (s.timestamp, self.get_metric_value(s, metric_name)))
            .collect()
    }
}

/// Statistics about the sliding window
#[derive(Debug, Clone)]
pub struct WindowStats {
    pub snapshot_count: usize,
    pub window_duration_seconds: u64,
    pub oldest_timestamp: u64,
    pub newest_timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ArrowGraph;
    use crate::streaming::incremental::IncrementalGraphProcessor;
    use arrow::array::{StringArray, Float64Array};
    use arrow::record_batch::RecordBatch;
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    fn create_test_graph() -> Result<ArrowGraph> {
        let nodes_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
        ]));
        let node_ids = StringArray::from(vec!["A", "B", "C"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["A", "B"]);
        let targets = StringArray::from(vec!["B", "C"]);
        let weights = Float64Array::from(vec![1.0, 1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_sliding_window_initialization() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();

        let mut window = SlidingWindowProcessor::default();
        window.initialize(&processor).unwrap();

        assert_eq!(window.snapshots.len(), 1);

        let snapshot = &window.snapshots[0];
        assert_eq!(snapshot.node_count, 3);
        assert_eq!(snapshot.edge_count, 2);
        assert!(snapshot.density > 0.0);
    }

    #[test]
    fn test_sliding_window_update() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();

        let mut window = SlidingWindowProcessor::default();
        window.initialize(&processor).unwrap();

        // Add more edges
        processor.add_edge("A".to_string(), "C".to_string(), 1.0).unwrap();

        let update_result = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 1,
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };

        window.update(&processor, &update_result).unwrap();

        assert_eq!(window.snapshots.len(), 2);

        let latest = &window.snapshots[1];
        assert_eq!(latest.edge_count, 2); // Should still be 2 (not incrementally updated)
    }

    #[test]
    fn test_trend_analysis() {
        let mut window = SlidingWindowProcessor::default();

        // Create mock snapshots with increasing node counts
        for i in 0..5 {
            let snapshot = GraphSnapshot {
                timestamp: i,
                node_count: (i + 1) as usize,
                edge_count: i as usize,
                density: (i as f64) * 0.1,
                avg_degree: (i as f64) * 0.5,
                largest_component_size: (i + 1) as usize,
                component_count: 1,
                clustering_coefficient: 0.3,
                custom_metrics: HashMap::new(),
            };
            window.snapshots.push_back(snapshot);
        }

        let analytics = window.analyze_temporal_patterns().unwrap();

        assert!(!analytics.trend_analysis.is_empty());

        let node_trend = analytics.trend_analysis.get("node_count").unwrap();
        assert_eq!(node_trend.direction, TrendDirection::Increasing);
        assert!(node_trend.slope > 0.0);
    }

    #[test]
    fn test_change_point_detection() {
        let mut window = SlidingWindowProcessor::default();

        // Create snapshots with a sudden change
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        for (i, &value) in values.iter().enumerate() {
            let snapshot = GraphSnapshot {
                timestamp: i as u64,
                node_count: value as usize,
                edge_count: 0,
                density: 0.0,
                avg_degree: 0.0,
                largest_component_size: 1,
                component_count: 1,
                clustering_coefficient: 0.0,
                custom_metrics: HashMap::new(),
            };
            window.snapshots.push_back(snapshot);
        }

        let analytics = window.analyze_temporal_patterns().unwrap();

        assert!(!analytics.change_points.is_empty());

        let change_point = &analytics.change_points[0];
        assert_eq!(change_point.metric_name, "node_count");
        assert!(change_point.change_magnitude > 0.0);
    }

    #[test]
    fn test_evolution_pattern_detection() {
        let mut window = SlidingWindowProcessor::default();

        // Create growth pattern
        for i in 0..10 {
            let snapshot = GraphSnapshot {
                timestamp: i,
                node_count: (i + 1) as usize,
                edge_count: (i * 2) as usize,
                density: 0.1,
                avg_degree: 1.0,
                largest_component_size: (i + 1) as usize,
                component_count: 1,
                clustering_coefficient: 0.3,
                custom_metrics: HashMap::new(),
            };
            window.snapshots.push_back(snapshot);
        }

        let analytics = window.analyze_temporal_patterns().unwrap();

        assert!(!analytics.evolution_patterns.is_empty());

        let growth_pattern = analytics.evolution_patterns.iter()
            .find(|p| p.pattern_type == PatternType::Growth);
        assert!(growth_pattern.is_some());

        let pattern = growth_pattern.unwrap();
        assert!(pattern.strength > 0.0);
        assert!(pattern.affected_metrics.contains(&"node_count".to_string()));
    }

    #[test]
    fn test_metric_evolution() {
        let mut window = SlidingWindowProcessor::default();

        for i in 0..5 {
            let snapshot = GraphSnapshot {
                timestamp: i * 1000, // Different timestamps
                node_count: ((i + 1) * 2) as usize,
                edge_count: i as usize,
                density: 0.1,
                avg_degree: 1.0,
                largest_component_size: 1,
                component_count: 1,
                clustering_coefficient: 0.3,
                custom_metrics: HashMap::new(),
            };
            window.snapshots.push_back(snapshot);
        }

        let evolution = window.metric_evolution("node_count");
        assert_eq!(evolution.len(), 5);

        assert_eq!(evolution[0], (0, 2.0));
        assert_eq!(evolution[1], (1000, 4.0));
        assert_eq!(evolution[4], (4000, 10.0));
    }

    #[test]
    fn test_window_stats() {
        let mut window = SlidingWindowProcessor::default();

        for i in 0..3 {
            let snapshot = GraphSnapshot {
                timestamp: i * 1000,
                node_count: 1,
                edge_count: 0,
                density: 0.0,
                avg_degree: 0.0,
                largest_component_size: 1,
                component_count: 1,
                clustering_coefficient: 0.0,
                custom_metrics: HashMap::new(),
            };
            window.snapshots.push_back(snapshot);
        }

        let stats = window.window_stats();
        assert_eq!(stats.snapshot_count, 3);
        assert_eq!(stats.window_duration_seconds, 2000);
        assert_eq!(stats.oldest_timestamp, 0);
        assert_eq!(stats.newest_timestamp, 2000);
    }
}