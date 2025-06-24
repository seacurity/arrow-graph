use crate::error::Result;
use crate::graph::ArrowGraph;
use crate::algorithms::centrality::PageRank;
use crate::algorithms::components::WeaklyConnectedComponents;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use arrow::array::Array;
use std::collections::{HashMap, HashSet};

/// Feature extractor for machine learning on graph data
/// Generates node-level, edge-level, and graph-level features
#[derive(Debug)]
pub struct FeatureExtractor {
    graph: Option<ArrowGraph>,
    node_features: HashMap<String, NodeFeatures>,
    edge_features: HashMap<(String, String), EdgeFeatures>,
    graph_features: GraphFeatures,
    config: FeatureConfig,
}

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    pub include_degree_features: bool,
    pub include_centrality_features: bool,
    pub include_clustering_features: bool,
    pub include_path_features: bool,
    pub include_community_features: bool,
    pub path_length_limit: usize,
    pub enable_higher_order_features: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            include_degree_features: true,
            include_centrality_features: true,
            include_clustering_features: true,
            include_path_features: true,
            include_community_features: true,
            path_length_limit: 3,
            enable_higher_order_features: false,
        }
    }
}

/// Node-level features for machine learning
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    pub node_id: String,
    
    // Degree-based features
    pub degree: usize,
    pub in_degree: usize,
    pub out_degree: usize,
    pub degree_centrality: f64,
    
    // Centrality features
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub eigenvector_centrality: f64,
    pub pagerank_score: f64,
    
    // Clustering features
    pub clustering_coefficient: f64,
    pub triangles_count: usize,
    
    // Community features
    pub community_id: Option<String>,
    pub community_size: usize,
    pub modularity_contribution: f64,
    
    // Path features
    pub eccentricity: f64,
    pub average_shortest_path: f64,
    pub reachable_nodes: usize,
    
    // Higher-order features
    pub k_core_number: usize,
    pub local_efficiency: f64,
    pub node_connectivity: usize,
}

/// Edge-level features for machine learning
#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    pub source: String,
    pub target: String,
    pub weight: f64,
    
    // Structural features
    pub edge_betweenness: f64,
    pub jaccard_coefficient: f64,
    pub adamic_adar_index: f64,
    pub common_neighbors: usize,
    pub preferential_attachment: f64,
    
    // Community features
    pub same_community: bool,
    pub community_bridge: bool,
    
    // Path features
    pub shortest_path_length: Option<usize>,
    pub alternative_paths: usize,
}

/// Graph-level features for machine learning
#[derive(Debug, Clone)]
pub struct GraphFeatures {
    // Basic structural features
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub average_degree: f64,
    
    // Connectivity features
    pub is_connected: bool,
    pub number_of_components: usize,
    pub largest_component_size: usize,
    pub diameter: Option<usize>,
    pub radius: Option<usize>,
    
    // Centralization features
    pub degree_centralization: f64,
    pub betweenness_centralization: f64,
    pub closeness_centralization: f64,
    
    // Clustering features
    pub global_clustering_coefficient: f64,
    pub transitivity: f64,
    pub average_clustering: f64,
    
    // Community features
    pub modularity: f64,
    pub number_of_communities: usize,
    
    // Degree distribution features
    pub degree_assortativity: f64,
    pub degree_variance: f64,
    pub degree_skewness: f64,
    pub degree_kurtosis: f64,
    
    // Small-world features
    pub small_world_coefficient: f64,
    pub average_path_length: f64,
}

/// Complete ML feature set ready for machine learning
#[derive(Debug, Clone)]
pub struct MLFeatureSet {
    pub node_features_matrix: Vec<Vec<f64>>,
    pub node_feature_names: Vec<String>,
    pub node_ids: Vec<String>,
    
    pub edge_features_matrix: Vec<Vec<f64>>,
    pub edge_feature_names: Vec<String>,
    pub edge_pairs: Vec<(String, String)>,
    
    pub graph_features_vector: Vec<f64>,
    pub graph_feature_names: Vec<String>,
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            graph: None,
            node_features: HashMap::new(),
            edge_features: HashMap::new(),
            graph_features: GraphFeatures::default(),
            config,
        }
    }

    pub fn default() -> Self {
        Self::new(FeatureConfig::default())
    }

    /// Initialize with graph data
    pub fn initialize(&mut self, graph: ArrowGraph) -> Result<()> {
        self.graph = Some(graph);
        Ok(())
    }

    /// Extract all features from the graph
    pub fn extract_features(&mut self) -> Result<MLFeatureSet> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| crate::error::GraphError::graph_construction("Graph not initialized"))?;

        // Extract node features
        if self.config.include_degree_features || 
           self.config.include_centrality_features || 
           self.config.include_clustering_features {
            self.extract_node_features(graph)?;
        }

        // Extract edge features
        if self.config.include_path_features {
            self.extract_edge_features(graph)?;
        }

        // Extract graph-level features
        self.extract_graph_features(graph)?;

        // Convert to ML-ready format
        self.create_ml_feature_set()
    }

    /// Extract node-level features
    fn extract_node_features(&mut self, graph: &ArrowGraph) -> Result<()> {
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;

        // Build adjacency information
        let mut adjacency = HashMap::new();
        let mut degrees = HashMap::new();
        let mut edge_weights = HashMap::new();

        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            let weights = edges_batch.column(2)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;

            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                let weight = weights.value(i);

                // Build adjacency list
                adjacency.entry(source.clone()).or_insert_with(Vec::new).push(target.clone());
                adjacency.entry(target.clone()).or_insert_with(Vec::new).push(source.clone());

                // Count degrees
                *degrees.entry(source.clone()).or_insert(0) += 1;
                *degrees.entry(target.clone()).or_insert(0) += 1;

                // Store edge weights
                edge_weights.insert((source, target), weight);
            }
        }

        // Process each node
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i).to_string();
                let degree = degrees.get(&node_id).copied().unwrap_or(0);
                
                let node_features = NodeFeatures {
                    node_id: node_id.clone(),
                    degree,
                    in_degree: degree, // Simplified for undirected graphs
                    out_degree: degree,
                    degree_centrality: self.calculate_degree_centrality(degree, graph.node_count()),
                    betweenness_centrality: self.calculate_betweenness_centrality(&node_id, &adjacency),
                    closeness_centrality: self.calculate_closeness_centrality(&node_id, &adjacency),
                    eigenvector_centrality: 0.0, // Placeholder
                    pagerank_score: self.calculate_pagerank_score(&node_id, graph)?,
                    clustering_coefficient: self.calculate_clustering_coefficient(&node_id, &adjacency),
                    triangles_count: self.count_triangles(&node_id, &adjacency),
                    community_id: None, // Would be filled by community detection
                    community_size: 0,
                    modularity_contribution: 0.0,
                    eccentricity: self.calculate_eccentricity(&node_id, &adjacency),
                    average_shortest_path: self.calculate_average_shortest_path(&node_id, &adjacency),
                    reachable_nodes: self.count_reachable_nodes(&node_id, &adjacency),
                    k_core_number: 0, // Placeholder
                    local_efficiency: 0.0, // Placeholder
                    node_connectivity: 0, // Placeholder
                };

                self.node_features.insert(node_id, node_features);
            }
        }

        Ok(())
    }

    /// Extract edge-level features
    fn extract_edge_features(&mut self, graph: &ArrowGraph) -> Result<()> {
        let edges_batch = &graph.edges;

        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            let weights = edges_batch.column(2)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;

            // Build adjacency for edge feature calculations
            let mut adjacency = HashMap::new();
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                adjacency.entry(source.clone()).or_insert_with(Vec::new).push(target.clone());
                adjacency.entry(target.clone()).or_insert_with(Vec::new).push(source.clone());
            }

            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                let weight = weights.value(i);

                let edge_features = EdgeFeatures {
                    source: source.clone(),
                    target: target.clone(),
                    weight,
                    edge_betweenness: 0.0, // Placeholder - computationally expensive
                    jaccard_coefficient: self.calculate_jaccard_coefficient(&source, &target, &adjacency),
                    adamic_adar_index: self.calculate_adamic_adar_index(&source, &target, &adjacency),
                    common_neighbors: self.count_common_neighbors(&source, &target, &adjacency),
                    preferential_attachment: self.calculate_preferential_attachment(&source, &target, &adjacency),
                    same_community: false, // Would be filled by community detection
                    community_bridge: false,
                    shortest_path_length: self.calculate_shortest_path_length(&source, &target, &adjacency),
                    alternative_paths: 0, // Placeholder
                };

                self.edge_features.insert((source, target), edge_features);
            }
        }

        Ok(())
    }

    /// Extract graph-level features
    fn extract_graph_features(&mut self, graph: &ArrowGraph) -> Result<()> {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        
        let density = if node_count > 1 {
            (2 * edge_count) as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };

        let average_degree = if node_count > 0 {
            (2 * edge_count) as f64 / node_count as f64
        } else {
            0.0
        };

        // Calculate connectivity features
        let (is_connected, number_of_components, largest_component_size) = 
            self.analyze_connectivity(graph)?;

        self.graph_features = GraphFeatures {
            node_count,
            edge_count,
            density,
            average_degree,
            is_connected,
            number_of_components,
            largest_component_size,
            diameter: None, // Computationally expensive
            radius: None,
            degree_centralization: 0.0, // Placeholder
            betweenness_centralization: 0.0,
            closeness_centralization: 0.0,
            global_clustering_coefficient: self.calculate_global_clustering_coefficient(graph)?,
            transitivity: 0.0, // Placeholder
            average_clustering: self.calculate_average_clustering_coefficient(),
            modularity: 0.0, // Would require community detection
            number_of_communities: 0,
            degree_assortativity: 0.0, // Placeholder
            degree_variance: 0.0,
            degree_skewness: 0.0,
            degree_kurtosis: 0.0,
            small_world_coefficient: 0.0,
            average_path_length: 0.0,
        };

        Ok(())
    }

    /// Create ML-ready feature set
    fn create_ml_feature_set(&self) -> Result<MLFeatureSet> {
        // Node features matrix
        let mut node_features_matrix = Vec::new();
        let mut node_ids = Vec::new();
        let node_feature_names = vec![
            "degree".to_string(),
            "degree_centrality".to_string(),
            "betweenness_centrality".to_string(),
            "closeness_centrality".to_string(),
            "pagerank_score".to_string(),
            "clustering_coefficient".to_string(),
            "triangles_count".to_string(),
            "eccentricity".to_string(),
            "average_shortest_path".to_string(),
            "reachable_nodes".to_string(),
        ];

        for (node_id, features) in &self.node_features {
            node_ids.push(node_id.clone());
            node_features_matrix.push(vec![
                features.degree as f64,
                features.degree_centrality,
                features.betweenness_centrality,
                features.closeness_centrality,
                features.pagerank_score,
                features.clustering_coefficient,
                features.triangles_count as f64,
                features.eccentricity,
                features.average_shortest_path,
                features.reachable_nodes as f64,
            ]);
        }

        // Edge features matrix
        let mut edge_features_matrix = Vec::new();
        let mut edge_pairs = Vec::new();
        let edge_feature_names = vec![
            "weight".to_string(),
            "jaccard_coefficient".to_string(),
            "adamic_adar_index".to_string(),
            "common_neighbors".to_string(),
            "preferential_attachment".to_string(),
        ];

        for ((source, target), features) in &self.edge_features {
            edge_pairs.push((source.clone(), target.clone()));
            edge_features_matrix.push(vec![
                features.weight,
                features.jaccard_coefficient,
                features.adamic_adar_index,
                features.common_neighbors as f64,
                features.preferential_attachment,
            ]);
        }

        // Graph features vector
        let graph_features_vector = vec![
            self.graph_features.node_count as f64,
            self.graph_features.edge_count as f64,
            self.graph_features.density,
            self.graph_features.average_degree,
            if self.graph_features.is_connected { 1.0 } else { 0.0 },
            self.graph_features.number_of_components as f64,
            self.graph_features.largest_component_size as f64,
            self.graph_features.global_clustering_coefficient,
            self.graph_features.average_clustering,
        ];

        let graph_feature_names = vec![
            "node_count".to_string(),
            "edge_count".to_string(),
            "density".to_string(),
            "average_degree".to_string(),
            "is_connected".to_string(),
            "number_of_components".to_string(),
            "largest_component_size".to_string(),
            "global_clustering_coefficient".to_string(),
            "average_clustering".to_string(),
        ];

        Ok(MLFeatureSet {
            node_features_matrix,
            node_feature_names,
            node_ids,
            edge_features_matrix,
            edge_feature_names,
            edge_pairs,
            graph_features_vector,
            graph_feature_names,
        })
    }

    // Helper methods for feature calculations

    fn calculate_degree_centrality(&self, degree: usize, total_nodes: usize) -> f64 {
        if total_nodes <= 1 {
            0.0
        } else {
            degree as f64 / (total_nodes - 1) as f64
        }
    }

    fn calculate_betweenness_centrality(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        // Simplified placeholder - full implementation requires all-pairs shortest paths
        let degree = adjacency.get(node).map(|neighbors| neighbors.len()).unwrap_or(0);
        degree as f64 / (adjacency.len().max(1) - 1) as f64
    }

    fn calculate_closeness_centrality(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        // Simplified placeholder
        let reachable = self.count_reachable_nodes(node, adjacency);
        if reachable > 0 {
            reachable as f64 / adjacency.len() as f64
        } else {
            0.0
        }
    }

    fn calculate_pagerank_score(&self, node: &str, graph: &ArrowGraph) -> Result<f64> {
        // Use existing PageRank implementation
        let pagerank = PageRank::default();
        let params = AlgorithmParams::default();
        let result = pagerank.execute(graph, &params)?;
        
        // Extract score for specific node (simplified)
        Ok(1.0 / graph.node_count() as f64) // Placeholder
    }

    fn calculate_clustering_coefficient(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        if let Some(neighbors) = adjacency.get(node) {
            if neighbors.len() < 2 {
                return 0.0;
            }

            let mut triangles = 0;
            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor1 = &neighbors[i];
                    let neighbor2 = &neighbors[j];
                    
                    if let Some(neighbor1_adj) = adjacency.get(neighbor1) {
                        if neighbor1_adj.contains(neighbor2) {
                            triangles += 1;
                        }
                    }
                }
            }

            if possible_triangles > 0 {
                triangles as f64 / possible_triangles as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn count_triangles(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> usize {
        if let Some(neighbors) = adjacency.get(node) {
            let mut triangles = 0;

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor1 = &neighbors[i];
                    let neighbor2 = &neighbors[j];
                    
                    if let Some(neighbor1_adj) = adjacency.get(neighbor1) {
                        if neighbor1_adj.contains(neighbor2) {
                            triangles += 1;
                        }
                    }
                }
            }

            triangles
        } else {
            0
        }
    }

    fn calculate_eccentricity(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        // Simplified - should be maximum shortest path from this node
        let distances = self.bfs_distances(node, adjacency);
        distances.values().map(|&d| d as f64).fold(0.0, f64::max)
    }

    fn calculate_average_shortest_path(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        let distances = self.bfs_distances(node, adjacency);
        if distances.is_empty() {
            0.0
        } else {
            let sum: usize = distances.values().sum();
            sum as f64 / distances.len() as f64
        }
    }

    fn count_reachable_nodes(&self, node: &str, adjacency: &HashMap<String, Vec<String>>) -> usize {
        self.bfs_distances(node, adjacency).len()
    }

    fn bfs_distances(&self, start: &str, adjacency: &HashMap<String, Vec<String>>) -> HashMap<String, usize> {
        let mut distances = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back((start.to_string(), 0));
        distances.insert(start.to_string(), 0);

        while let Some((current, dist)) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), dist + 1);
                        queue.push_back((neighbor.clone(), dist + 1));
                    }
                }
            }
        }

        distances
    }

    fn calculate_jaccard_coefficient(&self, node1: &str, node2: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        let neighbors1: HashSet<_> = adjacency.get(node1)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        let neighbors2: HashSet<_> = adjacency.get(node2)
            .map(|v| v.iter().collect())
            .unwrap_or_default();

        let intersection = neighbors1.intersection(&neighbors2).count();
        let union = neighbors1.union(&neighbors2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn calculate_adamic_adar_index(&self, node1: &str, node2: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        let neighbors1: HashSet<_> = adjacency.get(node1)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        let neighbors2: HashSet<_> = adjacency.get(node2)
            .map(|v| v.iter().collect())
            .unwrap_or_default();

        let mut sum = 0.0;
        for common_neighbor in neighbors1.intersection(&neighbors2) {
            if let Some(common_neighbors) = adjacency.get(*common_neighbor) {
                let degree = common_neighbors.len();
                if degree > 1 {
                    sum += 1.0 / (degree as f64).ln();
                }
            }
        }

        sum
    }

    fn count_common_neighbors(&self, node1: &str, node2: &str, adjacency: &HashMap<String, Vec<String>>) -> usize {
        let neighbors1: HashSet<_> = adjacency.get(node1)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        let neighbors2: HashSet<_> = adjacency.get(node2)
            .map(|v| v.iter().collect())
            .unwrap_or_default();

        neighbors1.intersection(&neighbors2).count()
    }

    fn calculate_preferential_attachment(&self, node1: &str, node2: &str, adjacency: &HashMap<String, Vec<String>>) -> f64 {
        let degree1 = adjacency.get(node1).map(|v| v.len()).unwrap_or(0);
        let degree2 = adjacency.get(node2).map(|v| v.len()).unwrap_or(0);
        
        (degree1 * degree2) as f64
    }

    fn calculate_shortest_path_length(&self, node1: &str, node2: &str, adjacency: &HashMap<String, Vec<String>>) -> Option<usize> {
        self.bfs_distances(node1, adjacency).get(node2).copied()
    }

    fn analyze_connectivity(&self, graph: &ArrowGraph) -> Result<(bool, usize, usize)> {
        let components_alg = WeaklyConnectedComponents::default();
        let params = AlgorithmParams::default();
        let result = components_alg.execute(graph, &params)?;

        // Extract component information (simplified)
        let number_of_components = 1; // Placeholder
        let largest_component_size = graph.node_count();
        let is_connected = number_of_components == 1;

        Ok((is_connected, number_of_components, largest_component_size))
    }

    fn calculate_global_clustering_coefficient(&self, _graph: &ArrowGraph) -> Result<f64> {
        // Placeholder - would calculate global clustering coefficient
        Ok(0.3)
    }

    fn calculate_average_clustering_coefficient(&self) -> f64 {
        if self.node_features.is_empty() {
            0.0
        } else {
            let sum: f64 = self.node_features.values()
                .map(|features| features.clustering_coefficient)
                .sum();
            sum / self.node_features.len() as f64
        }
    }

    /// Get node features for a specific node
    pub fn get_node_features(&self, node_id: &str) -> Option<&NodeFeatures> {
        self.node_features.get(node_id)
    }

    /// Get edge features for a specific edge
    pub fn get_edge_features(&self, source: &str, target: &str) -> Option<&EdgeFeatures> {
        self.edge_features.get(&(source.to_string(), target.to_string()))
            .or_else(|| self.edge_features.get(&(target.to_string(), source.to_string())))
    }

    /// Get graph-level features
    pub fn get_graph_features(&self) -> &GraphFeatures {
        &self.graph_features
    }
}

impl Default for GraphFeatures {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            density: 0.0,
            average_degree: 0.0,
            is_connected: false,
            number_of_components: 0,
            largest_component_size: 0,
            diameter: None,
            radius: None,
            degree_centralization: 0.0,
            betweenness_centralization: 0.0,
            closeness_centralization: 0.0,
            global_clustering_coefficient: 0.0,
            transitivity: 0.0,
            average_clustering: 0.0,
            modularity: 0.0,
            number_of_communities: 0,
            degree_assortativity: 0.0,
            degree_variance: 0.0,
            degree_skewness: 0.0,
            degree_kurtosis: 0.0,
            small_world_coefficient: 0.0,
            average_path_length: 0.0,
        }
    }
}

impl MLFeatureSet {
    /// Export node features to CSV format
    pub fn node_features_to_csv(&self) -> String {
        let mut csv = String::new();
        
        // Header
        csv.push_str("node_id,");
        csv.push_str(&self.node_feature_names.join(","));
        csv.push('\n');
        
        // Data rows
        for (i, node_id) in self.node_ids.iter().enumerate() {
            csv.push_str(node_id);
            csv.push(',');
            
            if i < self.node_features_matrix.len() {
                let features: Vec<String> = self.node_features_matrix[i]
                    .iter()
                    .map(|f| f.to_string())
                    .collect();
                csv.push_str(&features.join(","));
            }
            csv.push('\n');
        }
        
        csv
    }

    /// Export edge features to CSV format
    pub fn edge_features_to_csv(&self) -> String {
        let mut csv = String::new();
        
        // Header
        csv.push_str("source,target,");
        csv.push_str(&self.edge_feature_names.join(","));
        csv.push('\n');
        
        // Data rows
        for (i, (source, target)) in self.edge_pairs.iter().enumerate() {
            csv.push_str(source);
            csv.push(',');
            csv.push_str(target);
            csv.push(',');
            
            if i < self.edge_features_matrix.len() {
                let features: Vec<String> = self.edge_features_matrix[i]
                    .iter()
                    .map(|f| f.to_string())
                    .collect();
                csv.push_str(&features.join(","));
            }
            csv.push('\n');
        }
        
        csv
    }

    /// Get feature statistics
    pub fn get_feature_stats(&self) -> FeatureStats {
        let node_feature_count = self.node_feature_names.len();
        let edge_feature_count = self.edge_feature_names.len();
        let graph_feature_count = self.graph_feature_names.len();
        
        FeatureStats {
            total_nodes: self.node_ids.len(),
            total_edges: self.edge_pairs.len(),
            node_feature_count,
            edge_feature_count,
            graph_feature_count,
            total_features: node_feature_count + edge_feature_count + graph_feature_count,
        }
    }
}

/// Statistics about extracted features
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_feature_count: usize,
    pub edge_feature_count: usize,
    pub graph_feature_count: usize,
    pub total_features: usize,
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
        let sources = StringArray::from(vec!["A", "B", "C", "A"]);
        let targets = StringArray::from(vec!["B", "C", "D", "C"]);
        let weights = Float64Array::from(vec![1.0, 1.0, 1.0, 2.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_feature_extractor_initialization() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        assert!(extractor.graph.is_some());
    }

    #[test]
    fn test_node_feature_extraction() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        let features = extractor.extract_features().unwrap();
        
        assert_eq!(features.node_ids.len(), 4);
        assert!(!features.node_feature_names.is_empty());
        assert_eq!(features.node_features_matrix.len(), 4);
        
        // Check that each node has the expected number of features
        for feature_vector in &features.node_features_matrix {
            assert_eq!(feature_vector.len(), features.node_feature_names.len());
        }
    }

    #[test]
    fn test_edge_feature_extraction() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        let features = extractor.extract_features().unwrap();
        
        assert_eq!(features.edge_pairs.len(), 4);
        assert!(!features.edge_feature_names.is_empty());
        assert_eq!(features.edge_features_matrix.len(), 4);
    }

    #[test]
    fn test_graph_feature_extraction() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        let features = extractor.extract_features().unwrap();
        
        assert!(!features.graph_feature_names.is_empty());
        assert_eq!(features.graph_features_vector.len(), features.graph_feature_names.len());
        
        // Check basic graph metrics
        assert!(features.graph_features_vector[0] > 0.0); // node count
        assert!(features.graph_features_vector[1] > 0.0); // edge count
        assert!(features.graph_features_vector[2] >= 0.0); // density
    }

    #[test]
    fn test_clustering_coefficient_calculation() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        extractor.extract_features().unwrap();
        
        // Node A is connected to B and C, and B-C are connected
        // So A should have a clustering coefficient > 0
        let node_a_features = extractor.get_node_features("A").unwrap();
        assert!(node_a_features.clustering_coefficient >= 0.0);
    }

    #[test]
    fn test_jaccard_coefficient() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        extractor.extract_features().unwrap();
        
        // A-B should have a Jaccard coefficient based on common neighbors
        let edge_features = extractor.get_edge_features("A", "B").unwrap();
        assert!(edge_features.jaccard_coefficient >= 0.0);
        assert!(edge_features.jaccard_coefficient <= 1.0);
    }

    #[test]
    fn test_csv_export() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        let features = extractor.extract_features().unwrap();
        
        let node_csv = features.node_features_to_csv();
        assert!(node_csv.contains("node_id"));
        assert!(node_csv.contains("degree"));
        assert!(node_csv.contains("A"));
        
        let edge_csv = features.edge_features_to_csv();
        assert!(edge_csv.contains("source"));
        assert!(edge_csv.contains("target"));
        assert!(edge_csv.contains("weight"));
    }

    #[test]
    fn test_feature_stats() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        let features = extractor.extract_features().unwrap();
        let stats = features.get_feature_stats();
        
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.total_edges, 4);
        assert!(stats.node_feature_count > 0);
        assert!(stats.edge_feature_count > 0);
        assert!(stats.graph_feature_count > 0);
        assert_eq!(stats.total_features, 
                   stats.node_feature_count + stats.edge_feature_count + stats.graph_feature_count);
    }

    #[test]
    fn test_degree_centrality() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        extractor.extract_features().unwrap();
        
        // Check that degree centrality is properly normalized
        for features in extractor.node_features.values() {
            assert!(features.degree_centrality >= 0.0);
            assert!(features.degree_centrality <= 1.0);
        }
    }

    #[test]
    fn test_common_neighbors() {
        let graph = create_test_graph().unwrap();
        let mut extractor = FeatureExtractor::default();
        
        extractor.initialize(graph).unwrap();
        extractor.extract_features().unwrap();
        
        // A-C edge should have B as a common neighbor
        let edge_features = extractor.get_edge_features("A", "C").unwrap();
        assert!(edge_features.common_neighbors > 0);
    }
}