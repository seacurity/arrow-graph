use crate::error::Result;
use crate::streaming::incremental::{IncrementalGraphProcessor, UpdateResult};
use arrow::array::Array;
use std::collections::{HashMap, HashSet, VecDeque};

/// Graph change detection algorithms for anomaly detection and community drift
/// These algorithms monitor graph evolution and detect significant structural changes
#[derive(Debug, Clone)]
pub struct GraphChangeDetector {
    // Community tracking
    historical_communities: VecDeque<HashMap<String, String>>, // Rolling window of community assignments
    community_stability_threshold: f64, // Threshold for detecting community drift
    max_history_length: usize,
    
    // Anomaly detection
    baseline_metrics: GraphMetrics,
    anomaly_thresholds: AnomalyThresholds,
    anomaly_history: VecDeque<AnomalyEvent>,
    
    // Change tracking
    significant_changes: Vec<GraphChange>,
    #[allow(dead_code)]
    change_sensitivity: f64,
}

/// Key graph metrics for anomaly detection
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub average_degree: f64,
    pub clustering_coefficient: f64,
    pub largest_component_size: usize,
    pub component_count: usize,
}

/// Thresholds for detecting anomalous changes
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    pub density_change: f64,        // Threshold for density change
    pub degree_change: f64,         // Threshold for average degree change
    pub clustering_change: f64,     // Threshold for clustering coefficient change
    pub component_change: f64,      // Threshold for component structure change
    pub edge_burst_rate: f64,       // Threshold for edge addition rate
    pub node_burst_rate: f64,       // Threshold for node addition rate
}

/// Detected anomaly event
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    pub timestamp: u64,
    pub anomaly_type: AnomalyType,
    pub severity: f64,  // 0.0 to 1.0
    pub description: String,
    pub affected_nodes: Vec<String>,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    DensitySpike,       // Sudden increase in edge density
    DensityDrop,        // Sudden decrease in edge density
    DegreeBurst,        // Unusual degree distribution changes
    CommunityShift,     // Significant community structure changes
    ComponentMerge,     // Connected components merging
    ComponentSplit,     // Connected components splitting
    EdgeBurst,          // Rapid edge additions
    NodeBurst,          // Rapid node additions
    IsolatedNodes,      // Nodes becoming isolated
    HubFormation,       // New high-degree nodes appearing
}

/// Detected significant graph change
#[derive(Debug, Clone)]
pub struct GraphChange {
    pub change_type: ChangeType,
    pub magnitude: f64,
    pub affected_nodes: Vec<String>,
    pub timestamp: u64,
}

/// Types of significant structural changes
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    CommunityMerge,
    CommunitySplit,
    CommunityDrift,
    TopologyChange,
    ScaleChange,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            density_change: 0.1,      // 10% change in density
            degree_change: 0.15,      // 15% change in average degree
            clustering_change: 0.2,   // 20% change in clustering
            component_change: 0.05,   // 5% change in component structure
            edge_burst_rate: 10.0,    // 10+ edges per update
            node_burst_rate: 5.0,     // 5+ nodes per update
        }
    }
}

impl GraphChangeDetector {
    pub fn new(max_history: usize, change_sensitivity: f64) -> Self {
        Self {
            historical_communities: VecDeque::new(),
            community_stability_threshold: 0.8, // 80% stability required
            max_history_length: max_history,
            baseline_metrics: GraphMetrics {
                node_count: 0,
                edge_count: 0,
                density: 0.0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                largest_component_size: 0,
                component_count: 0,
            },
            anomaly_thresholds: AnomalyThresholds::default(),
            anomaly_history: VecDeque::new(),
            significant_changes: Vec::new(),
            change_sensitivity,
        }
    }

    /// Default change detector
    pub fn default() -> Self {
        Self::new(10, 0.1) // Keep 10 snapshots, 10% change sensitivity
    }

    /// Initialize the detector with current graph state
    pub fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        self.baseline_metrics = self.compute_graph_metrics(processor)?;
        
        // Initialize community detection
        let communities = self.detect_communities(processor)?;
        self.historical_communities.push_back(communities);
        
        Ok(())
    }

    /// Update detector with new graph changes
    pub fn update(&mut self, processor: &IncrementalGraphProcessor, changes: &UpdateResult) -> Result<Vec<AnomalyEvent>> {
        let mut detected_anomalies = Vec::new();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Compute current metrics
        let current_metrics = self.compute_graph_metrics(processor)?;

        // Detect anomalies based on metric changes
        detected_anomalies.extend(self.detect_metric_anomalies(&current_metrics, current_time)?);

        // Detect burst anomalies
        detected_anomalies.extend(self.detect_burst_anomalies(changes, current_time)?);

        // Detect community changes
        let current_communities = self.detect_communities(processor)?;
        detected_anomalies.extend(self.detect_community_anomalies(&current_communities, current_time)?);

        // Update history
        self.update_history(current_communities, &detected_anomalies);
        self.baseline_metrics = current_metrics;

        // Store significant changes
        if !detected_anomalies.is_empty() {
            let change = GraphChange {
                change_type: self.classify_change(&detected_anomalies),
                magnitude: self.calculate_change_magnitude(&detected_anomalies),
                affected_nodes: self.extract_affected_nodes(&detected_anomalies),
                timestamp: current_time,
            };
            self.significant_changes.push(change);
        }

        Ok(detected_anomalies)
    }

    /// Compute key graph metrics
    fn compute_graph_metrics(&self, processor: &IncrementalGraphProcessor) -> Result<GraphMetrics> {
        let graph = processor.graph();
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        
        let density = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };

        let average_degree = if node_count > 0 {
            (2 * edge_count) as f64 / node_count as f64
        } else {
            0.0
        };

        // Simplified clustering coefficient (would need more complex calculation)
        let clustering_coefficient = self.estimate_clustering_coefficient(processor)?;

        // Component analysis
        let components = self.analyze_components(processor)?;
        let largest_component_size = components.iter().map(|c| c.len()).max().unwrap_or(0);
        let component_count = components.len();

        Ok(GraphMetrics {
            node_count,
            edge_count,
            density,
            average_degree,
            clustering_coefficient,
            largest_component_size,
            component_count,
        })
    }

    /// Estimate clustering coefficient (simplified)
    fn estimate_clustering_coefficient(&self, _processor: &IncrementalGraphProcessor) -> Result<f64> {
        // Simplified implementation - in practice would calculate actual clustering
        Ok(0.3) // Placeholder value
    }

    /// Analyze connected components
    fn analyze_components(&self, processor: &IncrementalGraphProcessor) -> Result<Vec<Vec<String>>> {
        let graph = processor.graph();
        let edges_batch = &graph.edges;
        let nodes_batch = &graph.nodes;

        let mut components = Vec::new();
        let mut visited = HashSet::new();
        
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

    /// Detect communities using simplified algorithm
    fn detect_communities(&self, processor: &IncrementalGraphProcessor) -> Result<HashMap<String, String>> {
        // Simplified community detection - in practice would use more sophisticated algorithms
        let components = self.analyze_components(processor)?;
        let mut communities = HashMap::new();
        
        for (i, component) in components.iter().enumerate() {
            let community_id = format!("community_{}", i);
            for node in component {
                communities.insert(node.clone(), community_id.clone());
            }
        }
        
        Ok(communities)
    }

    /// Detect anomalies based on metric changes
    fn detect_metric_anomalies(&self, current: &GraphMetrics, timestamp: u64) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();
        let baseline = &self.baseline_metrics;

        // Density anomalies
        if baseline.density > 0.0 {
            let density_change = (current.density - baseline.density).abs() / baseline.density;
            if density_change > self.anomaly_thresholds.density_change {
                let anomaly_type = if current.density > baseline.density {
                    AnomalyType::DensitySpike
                } else {
                    AnomalyType::DensityDrop
                };
                
                anomalies.push(AnomalyEvent {
                    timestamp,
                    anomaly_type,
                    severity: (density_change / self.anomaly_thresholds.density_change).min(1.0),
                    description: format!("Density changed by {:.2}%", density_change * 100.0),
                    affected_nodes: Vec::new(),
                });
            }
        }

        // Degree anomalies
        if baseline.average_degree > 0.0 {
            let degree_change = (current.average_degree - baseline.average_degree).abs() / baseline.average_degree;
            if degree_change > self.anomaly_thresholds.degree_change {
                anomalies.push(AnomalyEvent {
                    timestamp,
                    anomaly_type: AnomalyType::DegreeBurst,
                    severity: (degree_change / self.anomaly_thresholds.degree_change).min(1.0),
                    description: format!("Average degree changed by {:.2}%", degree_change * 100.0),
                    affected_nodes: Vec::new(),
                });
            }
        }

        // Component anomalies
        if baseline.component_count > 0 {
            let component_change = (current.component_count as f64 - baseline.component_count as f64).abs() / baseline.component_count as f64;
            if component_change > self.anomaly_thresholds.component_change {
                let anomaly_type = if current.component_count < baseline.component_count {
                    AnomalyType::ComponentMerge
                } else {
                    AnomalyType::ComponentSplit
                };
                
                anomalies.push(AnomalyEvent {
                    timestamp,
                    anomaly_type,
                    severity: (component_change / self.anomaly_thresholds.component_change).min(1.0),
                    description: format!("Component count changed from {} to {}", baseline.component_count, current.component_count),
                    affected_nodes: Vec::new(),
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect burst anomalies
    fn detect_burst_anomalies(&self, changes: &UpdateResult, timestamp: u64) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();

        // Edge burst detection
        if changes.edges_added as f64 > self.anomaly_thresholds.edge_burst_rate {
            anomalies.push(AnomalyEvent {
                timestamp,
                anomaly_type: AnomalyType::EdgeBurst,
                severity: (changes.edges_added as f64 / self.anomaly_thresholds.edge_burst_rate).min(1.0),
                description: format!("Burst of {} edges added", changes.edges_added),
                affected_nodes: Vec::new(),
            });
        }

        // Node burst detection
        if changes.vertices_added as f64 > self.anomaly_thresholds.node_burst_rate {
            anomalies.push(AnomalyEvent {
                timestamp,
                anomaly_type: AnomalyType::NodeBurst,
                severity: (changes.vertices_added as f64 / self.anomaly_thresholds.node_burst_rate).min(1.0),
                description: format!("Burst of {} nodes added", changes.vertices_added),
                affected_nodes: Vec::new(),
            });
        }

        Ok(anomalies)
    }

    /// Detect community-based anomalies
    fn detect_community_anomalies(&self, current_communities: &HashMap<String, String>, timestamp: u64) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();

        if let Some(previous_communities) = self.historical_communities.back() {
            let stability = self.calculate_community_stability(previous_communities, current_communities);
            
            if stability < self.community_stability_threshold {
                anomalies.push(AnomalyEvent {
                    timestamp,
                    anomaly_type: AnomalyType::CommunityShift,
                    severity: 1.0 - stability,
                    description: format!("Community stability: {:.2}%", stability * 100.0),
                    affected_nodes: self.find_community_changes(previous_communities, current_communities),
                });
            }
        }

        Ok(anomalies)
    }

    /// Calculate community stability between two snapshots
    fn calculate_community_stability(&self, previous: &HashMap<String, String>, current: &HashMap<String, String>) -> f64 {
        let mut stable_assignments = 0;
        let mut total_nodes = 0;

        for (node, prev_community) in previous {
            if let Some(curr_community) = current.get(node) {
                if prev_community == curr_community {
                    stable_assignments += 1;
                }
                total_nodes += 1;
            }
        }

        if total_nodes > 0 {
            stable_assignments as f64 / total_nodes as f64
        } else {
            1.0
        }
    }

    /// Find nodes that changed communities
    fn find_community_changes(&self, previous: &HashMap<String, String>, current: &HashMap<String, String>) -> Vec<String> {
        let mut changed_nodes = Vec::new();

        for (node, prev_community) in previous {
            if let Some(curr_community) = current.get(node) {
                if prev_community != curr_community {
                    changed_nodes.push(node.clone());
                }
            }
        }

        changed_nodes
    }

    /// Update history with new data
    fn update_history(&mut self, communities: HashMap<String, String>, anomalies: &[AnomalyEvent]) {
        // Update community history
        self.historical_communities.push_back(communities);
        if self.historical_communities.len() > self.max_history_length {
            self.historical_communities.pop_front();
        }

        // Update anomaly history
        for anomaly in anomalies {
            self.anomaly_history.push_back(anomaly.clone());
        }
        if self.anomaly_history.len() > self.max_history_length * 5 {
            self.anomaly_history.pop_front();
        }
    }

    /// Classify the type of change based on detected anomalies
    fn classify_change(&self, anomalies: &[AnomalyEvent]) -> ChangeType {
        if anomalies.iter().any(|a| a.anomaly_type == AnomalyType::CommunityShift) {
            ChangeType::CommunityDrift
        } else if anomalies.iter().any(|a| matches!(a.anomaly_type, AnomalyType::ComponentMerge | AnomalyType::ComponentSplit)) {
            ChangeType::TopologyChange
        } else if anomalies.iter().any(|a| matches!(a.anomaly_type, AnomalyType::EdgeBurst | AnomalyType::NodeBurst)) {
            ChangeType::ScaleChange
        } else {
            ChangeType::TopologyChange
        }
    }

    /// Calculate the magnitude of change
    fn calculate_change_magnitude(&self, anomalies: &[AnomalyEvent]) -> f64 {
        if anomalies.is_empty() {
            0.0
        } else {
            anomalies.iter().map(|a| a.severity).sum::<f64>() / anomalies.len() as f64
        }
    }

    /// Extract all affected nodes from anomalies
    fn extract_affected_nodes(&self, anomalies: &[AnomalyEvent]) -> Vec<String> {
        let mut nodes = HashSet::new();
        for anomaly in anomalies {
            for node in &anomaly.affected_nodes {
                nodes.insert(node.clone());
            }
        }
        nodes.into_iter().collect()
    }

    /// Get recent anomalies
    pub fn recent_anomalies(&self, count: usize) -> Vec<&AnomalyEvent> {
        self.anomaly_history.iter().rev().take(count).collect()
    }

    /// Get significant changes
    pub fn significant_changes(&self) -> &[GraphChange] {
        &self.significant_changes
    }

    /// Check if the graph is in an anomalous state
    pub fn is_anomalous_state(&self) -> bool {
        // Check if recent anomalies exceed threshold
        let recent_severity: f64 = self.anomaly_history.iter()
            .rev()
            .take(3) // Last 3 anomalies
            .map(|a| a.severity)
            .sum();
        
        recent_severity > 2.0 // Threshold for anomalous state
    }

    /// Get current community stability
    pub fn current_community_stability(&self) -> Option<f64> {
        if self.historical_communities.len() >= 2 {
            let current = self.historical_communities.back()?;
            let previous = self.historical_communities.get(self.historical_communities.len() - 2)?;
            Some(self.calculate_community_stability(previous, current))
        } else {
            None
        }
    }
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
        let node_ids = StringArray::from(vec!["A", "B", "C", "D"]);
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
    fn test_change_detector_initialization() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut detector = GraphChangeDetector::default();
        detector.initialize(&processor).unwrap();
        
        assert_eq!(detector.baseline_metrics.node_count, 4);
        assert_eq!(detector.baseline_metrics.edge_count, 2);
        assert!(detector.baseline_metrics.density > 0.0);
        assert_eq!(detector.historical_communities.len(), 1);
    }

    #[test]
    fn test_metric_computation() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let detector = GraphChangeDetector::default();
        let metrics = detector.compute_graph_metrics(&processor).unwrap();
        
        assert_eq!(metrics.node_count, 4);
        assert_eq!(metrics.edge_count, 2);
        assert!(metrics.density > 0.0);
        assert!(metrics.average_degree > 0.0);
    }

    #[test]
    fn test_component_analysis() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let detector = GraphChangeDetector::default();
        let components = detector.analyze_components(&processor).unwrap();
        
        assert_eq!(components.len(), 2); // {A,B,C} and {D}
        
        // Find the larger component
        let larger_component = components.iter().max_by_key(|c| c.len()).unwrap();
        assert_eq!(larger_component.len(), 3);
    }

    #[test]
    fn test_burst_anomaly_detection() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let detector = GraphChangeDetector::default();
        
        // Simulate edge burst
        let edge_burst_changes = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 15, // Above threshold of 10
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };
        
        let anomalies = detector.detect_burst_anomalies(&edge_burst_changes, 12345).unwrap();
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::EdgeBurst);
        assert!(anomalies[0].severity > 0.0);
    }

    #[test]
    fn test_community_stability() {
        let detector = GraphChangeDetector::default();
        
        let mut community1 = HashMap::new();
        community1.insert("A".to_string(), "comm1".to_string());
        community1.insert("B".to_string(), "comm1".to_string());
        community1.insert("C".to_string(), "comm2".to_string());
        
        let mut community2 = HashMap::new();
        community2.insert("A".to_string(), "comm1".to_string());
        community2.insert("B".to_string(), "comm2".to_string()); // Changed
        community2.insert("C".to_string(), "comm2".to_string());
        
        let stability = detector.calculate_community_stability(&community1, &community2);
        assert!((stability - 0.666667).abs() < 0.001); // 2/3 stability
    }

    #[test]
    fn test_full_anomaly_detection() {
        let graph = create_test_graph().unwrap();
        let mut processor = IncrementalGraphProcessor::new(graph).unwrap();
        processor.set_batch_size(1);
        
        let mut detector = GraphChangeDetector::default();
        detector.initialize(&processor).unwrap();
        
        // Add many edges to trigger anomaly
        for i in 0..12 {
            processor.add_edge(format!("node_{}", i), format!("node_{}", i + 1), 1.0).unwrap();
        }
        
        let changes = UpdateResult {
            vertices_added: 24, // Added nodes
            vertices_removed: 0,
            edges_added: 12,
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };
        
        let anomalies = detector.update(&processor, &changes).unwrap();
        assert!(!anomalies.is_empty());
        
        // Should detect burst anomalies
        assert!(anomalies.iter().any(|a| matches!(a.anomaly_type, AnomalyType::EdgeBurst | AnomalyType::NodeBurst)));
    }

    #[test]
    fn test_anomalous_state_detection() {
        let mut detector = GraphChangeDetector::default();
        
        // Add high-severity anomalies
        for i in 0..3 {
            detector.anomaly_history.push_back(AnomalyEvent {
                timestamp: i,
                anomaly_type: AnomalyType::DensitySpike,
                severity: 0.8,
                description: "Test anomaly".to_string(),
                affected_nodes: vec![],
            });
        }
        
        assert!(detector.is_anomalous_state());
        
        // Clear history
        detector.anomaly_history.clear();
        assert!(!detector.is_anomalous_state());
    }

    #[test]
    fn test_change_classification() {
        let detector = GraphChangeDetector::default();
        
        let community_anomaly = AnomalyEvent {
            timestamp: 0,
            anomaly_type: AnomalyType::CommunityShift,
            severity: 0.5,
            description: "Test".to_string(),
            affected_nodes: vec![],
        };
        
        let change_type = detector.classify_change(&[community_anomaly]);
        assert_eq!(change_type, ChangeType::CommunityDrift);
        
        let burst_anomaly = AnomalyEvent {
            timestamp: 0,
            anomaly_type: AnomalyType::EdgeBurst,
            severity: 0.7,
            description: "Test".to_string(),
            affected_nodes: vec![],
        };
        
        let change_type = detector.classify_change(&[burst_anomaly]);
        assert_eq!(change_type, ChangeType::ScaleChange);
    }
}