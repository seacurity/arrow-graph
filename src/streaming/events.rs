use crate::error::Result;
use crate::streaming::incremental::{IncrementalGraphProcessor, UpdateResult};
use crate::streaming::algorithms::{StreamingAlgorithm, StreamingPageRank, StreamingConnectedComponents};
use crate::streaming::detection::{GraphChangeDetector, AnomalyEvent, AnomalyType};
use std::collections::HashMap;

/// Event-driven graph processing system that triggers algorithms based on graph changes
/// This system monitors graph updates and automatically executes relevant algorithms
#[derive(Debug)]
pub struct EventDrivenProcessor {
    // Core components
    processor: IncrementalGraphProcessor,
    change_detector: GraphChangeDetector,
    
    // Streaming algorithms
    pagerank: Option<StreamingPageRank>,
    components: Option<StreamingConnectedComponents>,
    
    // Event system
    event_rules: Vec<EventRule>,
    event_history: Vec<ProcessedEvent>,
    max_history: usize,
    
    // Performance tracking
    algorithm_stats: HashMap<String, AlgorithmStats>,
}

/// Rule that defines when and how to trigger algorithms based on events
#[derive(Debug, Clone)]
pub struct EventRule {
    pub name: String,
    pub trigger: EventTrigger,
    pub action: EventAction,
    pub priority: u8, // 0 = highest priority
    pub enabled: bool,
    pub cooldown_seconds: u64, // Minimum time between triggers
    pub last_triggered: Option<u64>,
}

/// Conditions that trigger algorithm execution
#[derive(Debug, Clone)]
pub enum EventTrigger {
    // Graph structure changes
    VertexCountChange(ChangeThreshold),
    EdgeCountChange(ChangeThreshold),
    DensityChange(ChangeThreshold),
    ComponentCountChange(usize), // Trigger when component count changes by N
    
    // Anomaly detection
    AnomalyDetected(Vec<AnomalyType>),
    AnomalySeverity(f64), // Trigger when anomaly severity exceeds threshold
    
    // Time-based
    Periodic(u64), // Trigger every N seconds
    
    // Complex conditions
    Custom(CustomTrigger),
}

/// Threshold configuration for numeric changes
#[derive(Debug, Clone)]
pub struct ChangeThreshold {
    pub absolute: Option<f64>,
    pub relative: Option<f64>, // Percentage change
}

/// Custom trigger function
#[derive(Debug, Clone)]
pub struct CustomTrigger {
    pub name: String,
    pub condition: TriggerCondition,
}

/// Different types of trigger conditions
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    HighDegreeNodes(usize, f64), // (min_degree, min_fraction)
    RapidGrowth(f64),            // Growth rate threshold
    StructuralChange(f64),       // Structural similarity threshold
    CommunityInstability(f64),   // Community stability threshold
}

/// Actions to take when events are triggered
#[derive(Debug, Clone)]
pub enum EventAction {
    // Algorithm execution
    RunPageRank,
    RunConnectedComponents,
    RunAllAlgorithms,
    
    // Analysis actions
    PerformAnomalyAnalysis,
    UpdateBaseline,
    
    // Notification actions
    LogEvent,
    TriggerAlert(AlertLevel),
    
    // Combined actions
    Multiple(Vec<EventAction>),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

/// Processed event record
#[derive(Debug, Clone)]
pub struct ProcessedEvent {
    pub timestamp: u64,
    pub rule_name: String,
    pub trigger_type: String,
    pub action_taken: String,
    pub execution_time_ms: u64,
    pub success: bool,
    pub details: HashMap<String, String>,
}

/// Performance statistics for algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmStats {
    pub total_executions: u64,
    pub total_time_ms: u64,
    pub average_time_ms: f64,
    pub last_execution: Option<u64>,
    pub success_rate: f64,
}

impl EventDrivenProcessor {
    /// Create new event-driven processor
    pub fn new(graph: crate::graph::ArrowGraph) -> Result<Self> {
        let processor = IncrementalGraphProcessor::new(graph)?;
        let change_detector = GraphChangeDetector::default();

        Ok(Self {
            processor,
            change_detector,
            pagerank: None,
            components: None,
            event_rules: Vec::new(),
            event_history: Vec::new(),
            max_history: 1000,
            algorithm_stats: HashMap::new(),
        })
    }

    /// Initialize the processor with default algorithms
    pub fn initialize(&mut self) -> Result<()> {
        // Initialize change detector
        self.change_detector.initialize(&self.processor)?;

        // Initialize algorithms
        let mut pagerank = StreamingPageRank::default();
        pagerank.initialize(&self.processor)?;
        self.pagerank = Some(pagerank);

        let mut components = StreamingConnectedComponents::new();
        components.initialize(&self.processor)?;
        self.components = Some(components);

        // Set up default event rules
        self.setup_default_rules();

        Ok(())
    }

    /// Setup default event rules
    fn setup_default_rules(&mut self) {
        // Rule 1: Run PageRank on significant edge changes
        self.add_rule(EventRule {
            name: "pagerank_on_edge_changes".to_string(),
            trigger: EventTrigger::EdgeCountChange(ChangeThreshold {
                absolute: Some(10.0),
                relative: Some(0.05), // 5% change
            }),
            action: EventAction::RunPageRank,
            priority: 1,
            enabled: true,
            cooldown_seconds: 30,
            last_triggered: None,
        });

        // Rule 2: Run connected components on vertex changes
        self.add_rule(EventRule {
            name: "components_on_vertex_changes".to_string(),
            trigger: EventTrigger::VertexCountChange(ChangeThreshold {
                absolute: Some(5.0),
                relative: Some(0.03), // 3% change
            }),
            action: EventAction::RunConnectedComponents,
            priority: 1,
            enabled: true,
            cooldown_seconds: 20,
            last_triggered: None,
        });

        // Rule 3: Alert on critical anomalies
        self.add_rule(EventRule {
            name: "critical_anomaly_alert".to_string(),
            trigger: EventTrigger::AnomalySeverity(0.8),
            action: EventAction::TriggerAlert(AlertLevel::Critical),
            priority: 0, // Highest priority
            enabled: true,
            cooldown_seconds: 60,
            last_triggered: None,
        });

        // Rule 4: Periodic baseline update
        self.add_rule(EventRule {
            name: "periodic_baseline_update".to_string(),
            trigger: EventTrigger::Periodic(300), // Every 5 minutes
            action: EventAction::UpdateBaseline,
            priority: 2,
            enabled: true,
            cooldown_seconds: 0,
            last_triggered: None,
        });

        // Rule 5: Full analysis on component structure changes
        self.add_rule(EventRule {
            name: "full_analysis_on_component_changes".to_string(),
            trigger: EventTrigger::ComponentCountChange(1),
            action: EventAction::Multiple(vec![
                EventAction::RunAllAlgorithms,
                EventAction::PerformAnomalyAnalysis,
                EventAction::LogEvent,
            ]),
            priority: 1,
            enabled: true,
            cooldown_seconds: 45,
            last_triggered: None,
        });
    }

    /// Add a new event rule
    pub fn add_rule(&mut self, rule: EventRule) {
        self.event_rules.push(rule);
        // Sort by priority (lower number = higher priority)
        self.event_rules.sort_by_key(|r| r.priority);
    }

    /// Update the graph and process events
    pub fn update(&mut self, changes: UpdateResult) -> Result<Vec<ProcessedEvent>> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut triggered_events = Vec::new();

        // Detect anomalies first
        let anomalies = self.change_detector.update(&self.processor, &changes)?;

        // Create a list of rule indices that need to be triggered
        let mut rules_to_trigger = Vec::new();
        
        for (index, rule) in self.event_rules.iter().enumerate() {
            if !rule.enabled {
                continue;
            }

            // Check cooldown
            if let Some(last_triggered) = rule.last_triggered {
                if current_time < last_triggered + rule.cooldown_seconds {
                    continue;
                }
            }

            // Check if rule should trigger
            if self.should_trigger_rule(rule, &changes, &anomalies, current_time)? {
                rules_to_trigger.push(index);
            }
        }

        // Execute triggered rules
        for rule_index in rules_to_trigger {
            let rule_name = self.event_rules[rule_index].name.clone();
            let rule_action = self.event_rules[rule_index].action.clone();
            let rule_trigger = self.event_rules[rule_index].trigger.clone();
            
            let start_time = std::time::Instant::now();
            
            // Execute the action
            let success = self.execute_action(&rule_action, &changes, &anomalies)?;
            
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            // Record the event
            let event = ProcessedEvent {
                timestamp: current_time,
                rule_name,
                trigger_type: format!("{:?}", rule_trigger),
                action_taken: format!("{:?}", rule_action),
                execution_time_ms: execution_time,
                success,
                details: self.create_event_details(&changes, &anomalies),
            };

            triggered_events.push(event.clone());
            self.record_event(event);

            // Update rule state
            self.event_rules[rule_index].last_triggered = Some(current_time);

            // Update algorithm stats if applicable
            self.update_algorithm_stats(&rule_action, execution_time, success);
        }

        Ok(triggered_events)
    }

    /// Check if a rule should trigger
    fn should_trigger_rule(
        &self,
        rule: &EventRule,
        changes: &UpdateResult,
        anomalies: &[AnomalyEvent],
        current_time: u64,
    ) -> Result<bool> {
        match &rule.trigger {
            EventTrigger::VertexCountChange(threshold) => {
                self.check_change_threshold(changes.vertices_added as f64, threshold)
            }

            EventTrigger::EdgeCountChange(threshold) => {
                self.check_change_threshold(changes.edges_added as f64, threshold)
            }

            EventTrigger::DensityChange(threshold) => {
                // This would require tracking previous density
                // For now, use edge count as a proxy
                self.check_change_threshold(changes.edges_added as f64, threshold)
            }

            EventTrigger::ComponentCountChange(min_change) => {
                Ok(changes.affected_components.len() >= *min_change)
            }

            EventTrigger::AnomalyDetected(types) => {
                Ok(anomalies.iter().any(|a| types.contains(&a.anomaly_type)))
            }

            EventTrigger::AnomalySeverity(threshold) => {
                Ok(anomalies.iter().any(|a| a.severity >= *threshold))
            }

            EventTrigger::Periodic(interval) => {
                if let Some(last_triggered) = rule.last_triggered {
                    Ok(current_time >= last_triggered + interval)
                } else {
                    Ok(true) // First time
                }
            }

            EventTrigger::Custom(custom) => {
                self.evaluate_custom_trigger(custom, changes, anomalies)
            }
        }
    }

    /// Check if a numeric change meets the threshold
    fn check_change_threshold(&self, change: f64, threshold: &ChangeThreshold) -> Result<bool> {
        let mut triggered = false;

        if let Some(absolute) = threshold.absolute {
            triggered |= change >= absolute;
        }

        if let Some(relative) = threshold.relative {
            // For relative threshold, we'd need the previous value
            // For now, just use absolute as a fallback
            triggered |= change >= relative * 100.0; // Simplified
        }

        Ok(triggered)
    }

    /// Evaluate custom trigger conditions
    fn evaluate_custom_trigger(
        &self,
        custom: &CustomTrigger,
        _changes: &UpdateResult,
        anomalies: &[AnomalyEvent],
    ) -> Result<bool> {
        match &custom.condition {
            TriggerCondition::HighDegreeNodes(_min_degree, _min_fraction) => {
                // Would need degree distribution analysis
                Ok(false) // Placeholder
            }

            TriggerCondition::RapidGrowth(threshold) => {
                // Check for rapid growth anomalies
                Ok(anomalies.iter().any(|a| {
                    matches!(a.anomaly_type, AnomalyType::EdgeBurst | AnomalyType::NodeBurst)
                        && a.severity >= *threshold
                }))
            }

            TriggerCondition::StructuralChange(threshold) => {
                // Check for structural anomalies
                Ok(anomalies.iter().any(|a| {
                    matches!(
                        a.anomaly_type,
                        AnomalyType::ComponentMerge | AnomalyType::ComponentSplit | AnomalyType::CommunityShift
                    ) && a.severity >= *threshold
                }))
            }

            TriggerCondition::CommunityInstability(threshold) => {
                if let Some(stability) = self.change_detector.current_community_stability() {
                    Ok(stability < *threshold)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Execute an action
    fn execute_action(
        &mut self,
        action: &EventAction,
        changes: &UpdateResult,
        anomalies: &[AnomalyEvent],
    ) -> Result<bool> {
        match action {
            EventAction::RunPageRank => {
                if let Some(ref mut pagerank) = self.pagerank {
                    pagerank.update(&self.processor, changes)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }

            EventAction::RunConnectedComponents => {
                if let Some(ref mut components) = self.components {
                    components.update(&self.processor, changes)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }

            EventAction::RunAllAlgorithms => {
                let mut success = true;
                
                if let Some(ref mut pagerank) = self.pagerank {
                    success &= pagerank.update(&self.processor, changes).is_ok();
                }
                
                if let Some(ref mut components) = self.components {
                    success &= components.update(&self.processor, changes).is_ok();
                }
                
                Ok(success)
            }

            EventAction::PerformAnomalyAnalysis => {
                // Anomaly analysis is already done in update()
                Ok(true)
            }

            EventAction::UpdateBaseline => {
                // Would update baseline metrics in change detector
                Ok(true)
            }

            EventAction::LogEvent => {
                log::info!("Event triggered: {} anomalies detected", anomalies.len());
                Ok(true)
            }

            EventAction::TriggerAlert(level) => {
                match level {
                    AlertLevel::Info => log::info!("Graph event alert: {} anomalies", anomalies.len()),
                    AlertLevel::Warning => log::warn!("Graph event warning: {} anomalies", anomalies.len()),
                    AlertLevel::Critical => log::error!("Critical graph event: {} anomalies", anomalies.len()),
                }
                Ok(true)
            }

            EventAction::Multiple(actions) => {
                let mut all_success = true;
                for sub_action in actions {
                    let success = self.execute_action(sub_action, changes, anomalies)?;
                    all_success &= success;
                }
                Ok(all_success)
            }
        }
    }

    /// Create event details
    fn create_event_details(
        &self,
        changes: &UpdateResult,
        anomalies: &[AnomalyEvent],
    ) -> HashMap<String, String> {
        let mut details = HashMap::new();

        details.insert("vertices_added".to_string(), changes.vertices_added.to_string());
        details.insert("vertices_removed".to_string(), changes.vertices_removed.to_string());
        details.insert("edges_added".to_string(), changes.edges_added.to_string());
        details.insert("edges_removed".to_string(), changes.edges_removed.to_string());
        details.insert("anomalies_count".to_string(), anomalies.len().to_string());

        if !anomalies.is_empty() {
            let max_severity = anomalies.iter()
                .map(|a| a.severity)
                .fold(0.0f64, |a, b| a.max(b));
            details.insert("max_anomaly_severity".to_string(), max_severity.to_string());
        }

        details
    }

    /// Record an event in history
    fn record_event(&mut self, event: ProcessedEvent) {
        self.event_history.push(event);

        // Limit history size
        while self.event_history.len() > self.max_history {
            self.event_history.remove(0);
        }
    }

    /// Update algorithm performance statistics
    fn update_algorithm_stats(&mut self, action: &EventAction, execution_time: u64, success: bool) {
        let algorithm_name = match action {
            EventAction::RunPageRank => "pagerank",
            EventAction::RunConnectedComponents => "connected_components",
            EventAction::RunAllAlgorithms => "all_algorithms",
            _ => return,
        };

        let stats = self.algorithm_stats.entry(algorithm_name.to_string()).or_insert(AlgorithmStats {
            total_executions: 0,
            total_time_ms: 0,
            average_time_ms: 0.0,
            last_execution: None,
            success_rate: 0.0,
        });

        stats.total_executions += 1;
        stats.total_time_ms += execution_time;
        stats.average_time_ms = stats.total_time_ms as f64 / stats.total_executions as f64;
        stats.last_execution = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );

        // Update success rate (simplified)
        if success {
            stats.success_rate = (stats.success_rate * (stats.total_executions - 1) as f64 + 1.0)
                / stats.total_executions as f64;
        } else {
            stats.success_rate = (stats.success_rate * (stats.total_executions - 1) as f64)
                / stats.total_executions as f64;
        }
    }

    /// Get processor reference
    pub fn processor(&self) -> &IncrementalGraphProcessor {
        &self.processor
    }

    /// Get mutable processor reference
    pub fn processor_mut(&mut self) -> &mut IncrementalGraphProcessor {
        &mut self.processor
    }

    /// Get PageRank results
    pub fn pagerank_results(&self) -> Option<&HashMap<String, f64>> {
        self.pagerank.as_ref().map(|pr| pr.get_result())
    }

    /// Get connected components results
    pub fn components_results(&self) -> Option<&HashMap<String, String>> {
        self.components.as_ref().map(|cc| cc.get_result())
    }

    /// Get recent events
    pub fn recent_events(&self, count: usize) -> Vec<&ProcessedEvent> {
        self.event_history.iter().rev().take(count).collect()
    }

    /// Get algorithm statistics
    pub fn algorithm_stats(&self) -> &HashMap<String, AlgorithmStats> {
        &self.algorithm_stats
    }

    /// Get event rules
    pub fn event_rules(&self) -> &[EventRule] {
        &self.event_rules
    }

    /// Enable/disable a rule
    pub fn set_rule_enabled(&mut self, rule_name: &str, enabled: bool) {
        if let Some(rule) = self.event_rules.iter_mut().find(|r| r.name == rule_name) {
            rule.enabled = enabled;
        }
    }

    /// Get recent anomalies
    pub fn recent_anomalies(&self, count: usize) -> Vec<&AnomalyEvent> {
        self.change_detector.recent_anomalies(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ArrowGraph;
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
        let sources = StringArray::from(vec!["A"]);
        let targets = StringArray::from(vec!["B"]);
        let weights = Float64Array::from(vec![1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_event_driven_processor_initialization() {
        let graph = create_test_graph().unwrap();
        let mut processor = EventDrivenProcessor::new(graph).unwrap();
        
        processor.initialize().unwrap();
        
        assert!(processor.pagerank.is_some());
        assert!(processor.components.is_some());
        assert!(!processor.event_rules.is_empty());
    }

    #[test]
    fn test_event_rule_creation() {
        let rule = EventRule {
            name: "test_rule".to_string(),
            trigger: EventTrigger::EdgeCountChange(ChangeThreshold {
                absolute: Some(5.0),
                relative: None,
            }),
            action: EventAction::RunPageRank,
            priority: 1,
            enabled: true,
            cooldown_seconds: 30,
            last_triggered: None,
        };

        assert_eq!(rule.name, "test_rule");
        assert!(rule.enabled);
        assert_eq!(rule.cooldown_seconds, 30);
    }

    #[test]
    fn test_change_threshold_evaluation() {
        let graph = create_test_graph().unwrap();
        let processor = EventDrivenProcessor::new(graph).unwrap();

        let threshold = ChangeThreshold {
            absolute: Some(10.0),
            relative: Some(0.1),
        };

        assert!(processor.check_change_threshold(15.0, &threshold).unwrap());
        assert!(!processor.check_change_threshold(5.0, &threshold).unwrap());
    }

    #[test]
    fn test_event_processing() {
        let graph = create_test_graph().unwrap();
        let mut processor = EventDrivenProcessor::new(graph).unwrap();
        processor.initialize().unwrap();

        // Simulate significant changes
        let changes = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 15, // Above threshold
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };

        let events = processor.update(changes).unwrap();
        
        // Should trigger some events due to edge changes
        assert!(!events.is_empty());
        
        // Check that algorithms were updated
        assert!(processor.pagerank_results().is_some());
        assert!(processor.components_results().is_some());
    }

    #[test]
    fn test_rule_cooldown() {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let graph = create_test_graph().unwrap();
        let mut processor = EventDrivenProcessor::new(graph).unwrap();
        processor.initialize().unwrap();
        
        // Clear default rules to test only our cooldown rule
        processor.event_rules.clear();

        // Add a rule with cooldown
        let rule = EventRule {
            name: "test_cooldown".to_string(),
            trigger: EventTrigger::EdgeCountChange(ChangeThreshold {
                absolute: Some(10.0), // Higher threshold
                relative: None,
            }),
            action: EventAction::LogEvent,
            priority: 1,
            enabled: true,
            cooldown_seconds: 60,
            last_triggered: Some(current_time), // Just triggered
        };
        
        processor.add_rule(rule);

        let changes = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 15, // Above threshold
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };

        // Should not trigger any events due to cooldown
        let events = processor.update(changes.clone()).unwrap();
        assert!(events.is_empty());

        // Update the rule to have old last_triggered time
        processor.event_rules[0].last_triggered = Some(current_time - 120); // 2 minutes ago
        
        // Now should trigger
        let events = processor.update(changes).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_algorithm_stats_tracking() {
        let graph = create_test_graph().unwrap();
        let mut processor = EventDrivenProcessor::new(graph).unwrap();
        processor.initialize().unwrap();

        // Execute some actions
        processor.update_algorithm_stats(&EventAction::RunPageRank, 100, true);
        processor.update_algorithm_stats(&EventAction::RunPageRank, 200, true);

        let stats = processor.algorithm_stats();
        let pagerank_stats = stats.get("pagerank").unwrap();

        assert_eq!(pagerank_stats.total_executions, 2);
        assert_eq!(pagerank_stats.total_time_ms, 300);
        assert_eq!(pagerank_stats.average_time_ms, 150.0);
        assert_eq!(pagerank_stats.success_rate, 1.0);
    }

    #[test]
    fn test_custom_trigger_evaluation() {
        let graph = create_test_graph().unwrap();
        let processor = EventDrivenProcessor::new(graph).unwrap();

        let custom_trigger = CustomTrigger {
            name: "rapid_growth".to_string(),
            condition: TriggerCondition::RapidGrowth(0.5),
        };

        // Create anomaly that should trigger
        let anomalies = vec![AnomalyEvent {
            timestamp: 0,
            anomaly_type: AnomalyType::EdgeBurst,
            severity: 0.8,
            description: "Test burst".to_string(),
            affected_nodes: vec![],
        }];

        let changes = UpdateResult {
            vertices_added: 0,
            vertices_removed: 0,
            edges_added: 0,
            edges_removed: 0,
            affected_components: vec![],
            recomputation_needed: false,
        };

        assert!(processor.evaluate_custom_trigger(&custom_trigger, &changes, &anomalies).unwrap());
    }

    #[test]
    fn test_event_history_management() {
        let graph = create_test_graph().unwrap();
        let mut processor = EventDrivenProcessor::new(graph).unwrap();
        processor.max_history = 3; // Small limit for testing

        // Add more events than the limit
        for i in 0..5 {
            let event = ProcessedEvent {
                timestamp: i,
                rule_name: format!("rule_{}", i),
                trigger_type: "test".to_string(),
                action_taken: "test".to_string(),
                execution_time_ms: 100,
                success: true,
                details: HashMap::new(),
            };
            processor.record_event(event);
        }

        // Should only keep the last 3 events
        assert_eq!(processor.event_history.len(), 3);
        assert_eq!(processor.event_history[0].rule_name, "rule_2");
        assert_eq!(processor.event_history[2].rule_name, "rule_4");
    }
}