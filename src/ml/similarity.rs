use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::Array;
use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Serialize, Deserialize};

/// Graph kernel processor for computing similarity between graphs
/// Implements various graph kernels for machine learning applications
#[derive(Debug)]
pub struct GraphKernel {
    kernel_type: KernelType,
    parameters: KernelParameters,
}

/// Types of graph kernels
#[derive(Debug, Clone)]
pub enum KernelType {
    RandomWalk,
    ShortestPath,
    WeisfeilerLehman,
    GraphletSampling,
    Subtree,
    Neighborhood,
}

/// Parameters for different kernel types
#[derive(Debug, Clone)]
pub struct KernelParameters {
    pub walk_length: usize,
    pub num_walks: usize,
    pub wl_iterations: usize,
    pub graphlet_size: usize,
    pub neighborhood_size: usize,
    pub lambda: f64, // Decay parameter for random walk kernel
}

impl Default for KernelParameters {
    fn default() -> Self {
        Self {
            walk_length: 10,
            num_walks: 100,
            wl_iterations: 3,
            graphlet_size: 3,
            neighborhood_size: 2,
            lambda: 0.01,
        }
    }
}

/// Graph kernel computation result
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub similarity_score: f64,
    pub kernel_type: KernelType,
    pub computation_time: f64,
    pub feature_vector_size: usize,
}

/// Graph edit distance calculator
/// Computes minimum edit distance between two graphs
#[derive(Debug)]
pub struct GraphEditDistance {
    node_cost: CostFunction,
    edge_cost: CostFunction,
    use_approximation: bool,
    max_iterations: usize,
}

/// Cost functions for edit operations
#[derive(Debug, Clone)]
pub enum CostFunction {
    Uniform(f64), // Uniform cost for all operations
    NodeDependent(HashMap<String, f64>), // Node-dependent costs
    EdgeDependent(HashMap<(String, String), f64>), // Edge-dependent costs
    Custom(fn(&str, &str) -> f64), // Custom cost function
}

/// Edit operation types
#[derive(Debug, Clone, PartialEq)]
pub enum EditOperation {
    InsertNode(String),
    DeleteNode(String),
    SubstituteNode(String, String),
    InsertEdge(String, String),
    DeleteEdge(String, String),
    SubstituteEdge((String, String), (String, String)),
}

/// Edit distance result
#[derive(Debug, Clone)]
pub struct EditDistanceResult {
    pub distance: f64,
    pub operations: Vec<EditOperation>,
    pub normalized_distance: f64,
    pub computation_time: f64,
}

/// Motif detector for finding common subgraph patterns
/// Identifies frequent subgraphs and structural motifs
#[derive(Debug)]
pub struct MotifDetector {
    min_support: f64,
    max_motif_size: usize,
    motif_type: MotifType,
    use_canonical_form: bool,
}

/// Types of motifs to detect
#[derive(Debug, Clone)]
pub enum MotifType {
    ConnectedSubgraphs,
    Trees,
    Cycles,
    Cliques,
    Stars,
    Paths,
    Custom(Box<dyn Fn(&SubGraph) -> bool>),
}

/// Subgraph representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubGraph {
    pub nodes: HashSet<String>,
    pub edges: HashSet<(String, String)>,
    pub node_labels: HashMap<String, String>,
    pub edge_labels: HashMap<(String, String), String>,
}

/// Motif detection result
#[derive(Debug, Clone)]
pub struct MotifResult {
    pub motifs: Vec<DetectedMotif>,
    pub motif_counts: HashMap<SubGraph, usize>,
    pub total_motifs: usize,
    pub computation_time: f64,
}

/// Individual detected motif
#[derive(Debug, Clone)]
pub struct DetectedMotif {
    pub pattern: SubGraph,
    pub occurrences: Vec<SubGraph>,
    pub support: f64,
    pub significance: f64,
}

/// Subgraph matching engine
/// Performs subgraph isomorphism and pattern matching
#[derive(Debug)]
pub struct SubgraphMatcher {
    algorithm: MatchingAlgorithm,
    use_node_labels: bool,
    use_edge_labels: bool,
    timeout_ms: Option<u64>,
}

/// Subgraph matching algorithms
#[derive(Debug, Clone)]
pub enum MatchingAlgorithm {
    VF2,        // VF2 algorithm
    Ullmann,    // Ullmann's algorithm
    BackTrack,  // Simple backtracking
    LAD,        // LAD algorithm
}

/// Subgraph matching result
#[derive(Debug, Clone)]
pub struct MatchingResult {
    pub is_subgraph: bool,
    pub mappings: Vec<NodeMapping>,
    pub num_mappings: usize,
    pub computation_time: f64,
}

/// Node mapping between query and target graph
#[derive(Debug, Clone)]
pub struct NodeMapping {
    pub query_to_target: HashMap<String, String>,
    pub target_to_query: HashMap<String, String>,
}

impl GraphKernel {
    /// Create a new graph kernel processor
    pub fn new(kernel_type: KernelType) -> Self {
        Self {
            kernel_type,
            parameters: KernelParameters::default(),
        }
    }

    /// Set kernel parameters
    pub fn with_parameters(mut self, parameters: KernelParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Compute kernel similarity between two graphs
    pub fn compute_similarity(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<KernelResult> {
        let start_time = std::time::Instant::now();
        
        let similarity_score = match &self.kernel_type {
            KernelType::RandomWalk => self.random_walk_kernel(graph1, graph2)?,
            KernelType::ShortestPath => self.shortest_path_kernel(graph1, graph2)?,
            KernelType::WeisfeilerLehman => self.weisfeiler_lehman_kernel(graph1, graph2)?,
            KernelType::GraphletSampling => self.graphlet_sampling_kernel(graph1, graph2)?,
            KernelType::Subtree => self.subtree_kernel(graph1, graph2)?,
            KernelType::Neighborhood => self.neighborhood_kernel(graph1, graph2)?,
        };
        
        let computation_time = start_time.elapsed().as_secs_f64();
        
        Ok(KernelResult {
            similarity_score,
            kernel_type: self.kernel_type.clone(),
            computation_time,
            feature_vector_size: self.estimate_feature_size(graph1, graph2),
        })
    }

    /// Random walk kernel implementation
    fn random_walk_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let adj1 = self.build_adjacency_matrix(graph1)?;
        let adj2 = self.build_adjacency_matrix(graph2)?;
        
        let walks1 = self.generate_random_walks(&adj1)?;
        let walks2 = self.generate_random_walks(&adj2)?;
        
        // Compute kernel based on walk similarities
        let mut kernel_value = 0.0;
        for walk1 in &walks1 {
            for walk2 in &walks2 {
                kernel_value += self.walk_similarity(walk1, walk2);
            }
        }
        
        // Normalize by number of walks
        kernel_value /= (walks1.len() * walks2.len()) as f64;
        
        Ok(kernel_value)
    }

    /// Shortest path kernel implementation
    fn shortest_path_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let paths1 = self.compute_all_shortest_paths(graph1)?;
        let paths2 = self.compute_all_shortest_paths(graph2)?;
        
        // Compare path length distributions
        let mut similarity = 0.0;
        let max_length = 10; // Limit path length for efficiency
        
        for length in 1..=max_length {
            let count1 = paths1.iter().filter(|&path| path.len() == length).count() as f64;
            let count2 = paths2.iter().filter(|&path| path.len() == length).count() as f64;
            
            // Normalize counts
            let norm1 = count1 / paths1.len() as f64;
            let norm2 = count2 / paths2.len() as f64;
            
            // Add to kernel value (using RBF-like similarity)
            similarity += (-((norm1 - norm2).powi(2)) / (2.0 * 0.1_f64.powi(2))).exp();
        }
        
        Ok(similarity / max_length as f64)
    }

    /// Weisfeiler-Lehman kernel implementation
    fn weisfeiler_lehman_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let mut labels1 = self.initialize_node_labels(graph1)?;
        let mut labels2 = self.initialize_node_labels(graph2)?;
        
        let mut kernel_value = 0.0;
        
        for iteration in 0..self.parameters.wl_iterations {
            // Count label occurrences
            let hist1 = self.count_labels(&labels1);
            let hist2 = self.count_labels(&labels2);
            
            // Compute kernel contribution for this iteration
            kernel_value += self.histogram_intersection(&hist1, &hist2);
            
            // Update labels
            labels1 = self.update_wl_labels(graph1, &labels1)?;
            labels2 = self.update_wl_labels(graph2, &labels2)?;
        }
        
        Ok(kernel_value)
    }

    /// Graphlet sampling kernel implementation
    fn graphlet_sampling_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let graphlets1 = self.sample_graphlets(graph1)?;
        let graphlets2 = self.sample_graphlets(graph2)?;
        
        // Count graphlet types
        let counts1 = self.count_graphlet_types(&graphlets1);
        let counts2 = self.count_graphlet_types(&graphlets2);
        
        // Compute normalized histogram intersection
        let total1: f64 = counts1.values().sum();
        let total2: f64 = counts2.values().sum();
        
        let mut similarity = 0.0;
        let all_types: HashSet<_> = counts1.keys().chain(counts2.keys()).collect();
        
        for graphlet_type in all_types {
            let freq1 = counts1.get(graphlet_type).unwrap_or(&0.0) / total1;
            let freq2 = counts2.get(graphlet_type).unwrap_or(&0.0) / total2;
            similarity += freq1.min(freq2);
        }
        
        Ok(similarity)
    }

    /// Subtree kernel implementation
    fn subtree_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let subtrees1 = self.enumerate_subtrees(graph1)?;
        let subtrees2 = self.enumerate_subtrees(graph2)?;
        
        // Compute common subtrees
        let mut common_count = 0;
        for subtree1 in &subtrees1 {
            for subtree2 in &subtrees2 {
                if self.subtrees_isomorphic(subtree1, subtree2) {
                    common_count += 1;
                }
            }
        }
        
        // Normalize by geometric mean
        let norm = ((subtrees1.len() * subtrees2.len()) as f64).sqrt();
        Ok(common_count as f64 / norm)
    }

    /// Neighborhood kernel implementation
    fn neighborhood_kernel(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<f64> {
        let nodes1 = self.get_node_list(graph1)?;
        let nodes2 = self.get_node_list(graph2)?;
        
        let mut similarity = 0.0;
        let mut total_comparisons = 0;
        
        for node1 in &nodes1 {
            let neighborhood1 = self.get_neighborhood(graph1, node1, self.parameters.neighborhood_size)?;
            
            for node2 in &nodes2 {
                let neighborhood2 = self.get_neighborhood(graph2, node2, self.parameters.neighborhood_size)?;
                similarity += self.neighborhood_similarity(&neighborhood1, &neighborhood2);
                total_comparisons += 1;
            }
        }
        
        Ok(similarity / total_comparisons as f64)
    }

    /// Helper methods for kernel computations
    fn build_adjacency_matrix(&self, graph: &ArrowGraph) -> Result<HashMap<String, Vec<String>>> {
        let mut adjacency = HashMap::new();
        
        // Initialize with all nodes
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                adjacency.insert(node_ids.value(i).to_string(), Vec::new());
            }
        }
        
        // Add edges
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
            
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                adjacency.entry(source.clone()).or_default().push(target.clone());
                adjacency.entry(target).or_default().push(source); // Undirected
            }
        }
        
        Ok(adjacency)
    }

    fn generate_random_walks(&self, adjacency: &HashMap<String, Vec<String>>) -> Result<Vec<Vec<String>>> {
        let mut walks = Vec::new();
        let nodes: Vec<_> = adjacency.keys().cloned().collect();
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.parameters.num_walks {
            if nodes.is_empty() {
                continue;
            }
            
            let start_node = &nodes[rng.gen_range(0..nodes.len())];
            let mut walk = vec![start_node.clone()];
            let mut current = start_node;
            
            for _ in 1..self.parameters.walk_length {
                if let Some(neighbors) = adjacency.get(current) {
                    if neighbors.is_empty() {
                        break;
                    }
                    current = &neighbors[rng.gen_range(0..neighbors.len())];
                    walk.push(current.clone());
                } else {
                    break;
                }
            }
            
            if walk.len() > 1 {
                walks.push(walk);
            }
        }
        
        Ok(walks)
    }

    fn walk_similarity(&self, walk1: &[String], walk2: &[String]) -> f64 {
        // Simple walk similarity based on common subsequences
        let mut common = 0;
        let min_len = walk1.len().min(walk2.len());
        
        for i in 0..min_len {
            if walk1[i] == walk2[i] {
                common += 1;
            }
        }
        
        common as f64 / min_len as f64
    }

    fn compute_all_shortest_paths(&self, graph: &ArrowGraph) -> Result<Vec<Vec<String>>> {
        let adjacency = self.build_adjacency_matrix(graph)?;
        let mut all_paths = Vec::new();
        
        for start_node in adjacency.keys() {
            let paths = self.bfs_shortest_paths(start_node, &adjacency);
            all_paths.extend(paths);
        }
        
        Ok(all_paths)
    }

    fn bfs_shortest_paths(&self, start: &str, adjacency: &HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut paths = Vec::new();
        
        queue.push_back(vec![start.to_string()]);
        visited.insert(start.to_string());
        
        while let Some(path) = queue.pop_front() {
            if path.len() > 6 { // Limit path length
                continue;
            }
            
            let current = path.last().unwrap();
            paths.push(path.clone());
            
            if let Some(neighbors) = adjacency.get(current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back(new_path);
                        visited.insert(neighbor.clone());
                    }
                }
            }
        }
        
        paths
    }

    fn initialize_node_labels(&self, graph: &ArrowGraph) -> Result<HashMap<String, String>> {
        let mut labels = HashMap::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i).to_string();
                // Initialize with degree as label
                labels.insert(node_id.clone(), "1".to_string()); // Simplified
            }
        }
        
        Ok(labels)
    }

    fn update_wl_labels(&self, graph: &ArrowGraph, current_labels: &HashMap<String, String>) -> Result<HashMap<String, String>> {
        let adjacency = self.build_adjacency_matrix(graph)?;
        let mut new_labels = HashMap::new();
        
        for (node, _) in current_labels {
            let mut neighbor_labels = Vec::new();
            
            if let Some(neighbors) = adjacency.get(node) {
                for neighbor in neighbors {
                    if let Some(label) = current_labels.get(neighbor) {
                        neighbor_labels.push(label.clone());
                    }
                }
            }
            
            neighbor_labels.sort();
            let combined = format!("{}:{}", current_labels.get(node).unwrap_or(&"0".to_string()), neighbor_labels.join(","));
            new_labels.insert(node.clone(), self.hash_string(&combined));
        }
        
        Ok(new_labels)
    }

    fn hash_string(&self, s: &str) -> String {
        // Simple hash function for WL labels
        format!("{}", s.len() % 1000)
    }

    fn count_labels(&self, labels: &HashMap<String, String>) -> HashMap<String, f64> {
        let mut counts = HashMap::new();
        
        for label in labels.values() {
            *counts.entry(label.clone()).or_insert(0.0) += 1.0;
        }
        
        counts
    }

    fn histogram_intersection(&self, hist1: &HashMap<String, f64>, hist2: &HashMap<String, f64>) -> f64 {
        let mut intersection = 0.0;
        
        for (label, count1) in hist1 {
            if let Some(count2) = hist2.get(label) {
                intersection += count1.min(*count2);
            }
        }
        
        intersection
    }

    fn sample_graphlets(&self, _graph: &ArrowGraph) -> Result<Vec<SubGraph>> {
        // Simplified graphlet sampling
        // In a full implementation, this would sample small connected subgraphs
        Ok(Vec::new())
    }

    fn count_graphlet_types(&self, _graphlets: &[SubGraph]) -> HashMap<String, f64> {
        // Count different types of graphlets
        HashMap::new()
    }

    fn enumerate_subtrees(&self, _graph: &ArrowGraph) -> Result<Vec<SubGraph>> {
        // Enumerate all subtrees up to a certain size
        Ok(Vec::new())
    }

    fn subtrees_isomorphic(&self, _subtree1: &SubGraph, _subtree2: &SubGraph) -> bool {
        // Check if two subtrees are isomorphic
        false
    }

    fn get_node_list(&self, graph: &ArrowGraph) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }
        
        Ok(nodes)
    }

    fn get_neighborhood(&self, graph: &ArrowGraph, node: &str, size: usize) -> Result<HashSet<String>> {
        let adjacency = self.build_adjacency_matrix(graph)?;
        let mut neighborhood = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back((node.to_string(), 0));
        visited.insert(node.to_string());
        
        while let Some((current, depth)) = queue.pop_front() {
            neighborhood.insert(current.clone());
            
            if depth < size {
                if let Some(neighbors) = adjacency.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            queue.push_back((neighbor.clone(), depth + 1));
                            visited.insert(neighbor.clone());
                        }
                    }
                }
            }
        }
        
        Ok(neighborhood)
    }

    fn neighborhood_similarity(&self, neighborhood1: &HashSet<String>, neighborhood2: &HashSet<String>) -> f64 {
        let intersection = neighborhood1.intersection(neighborhood2).count();
        let union = neighborhood1.union(neighborhood2).count();
        
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn estimate_feature_size(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> usize {
        // Estimate feature vector size based on kernel type
        let nodes1 = graph1.node_count();
        let nodes2 = graph2.node_count();
        
        match &self.kernel_type {
            KernelType::RandomWalk => self.parameters.num_walks,
            KernelType::ShortestPath => (nodes1 + nodes2) * 10,
            KernelType::WeisfeilerLehman => (nodes1 + nodes2) * self.parameters.wl_iterations,
            KernelType::GraphletSampling => 1000, // Typical graphlet count
            KernelType::Subtree => nodes1 + nodes2,
            KernelType::Neighborhood => nodes1 + nodes2,
        }
    }
}

impl GraphEditDistance {
    /// Create a new graph edit distance calculator
    pub fn new() -> Self {
        Self {
            node_cost: CostFunction::Uniform(1.0),
            edge_cost: CostFunction::Uniform(1.0),
            use_approximation: true,
            max_iterations: 1000,
        }
    }

    /// Set cost functions
    pub fn with_costs(mut self, node_cost: CostFunction, edge_cost: CostFunction) -> Self {
        self.node_cost = node_cost;
        self.edge_cost = edge_cost;
        self
    }

    /// Compute edit distance between two graphs
    pub fn compute_distance(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<EditDistanceResult> {
        let start_time = std::time::Instant::now();
        
        let (distance, operations) = if self.use_approximation {
            self.approximate_edit_distance(graph1, graph2)?
        } else {
            self.exact_edit_distance(graph1, graph2)?
        };
        
        let computation_time = start_time.elapsed().as_secs_f64();
        
        // Normalize distance by maximum possible distance
        let max_distance = self.compute_max_distance(graph1, graph2);
        let normalized_distance = if max_distance > 0.0 {
            distance / max_distance
        } else {
            0.0
        };
        
        Ok(EditDistanceResult {
            distance,
            operations,
            normalized_distance,
            computation_time,
        })
    }

    /// Approximate edit distance using greedy approach
    fn approximate_edit_distance(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<(f64, Vec<EditOperation>)> {
        let nodes1 = self.get_node_set(graph1)?;
        let nodes2 = self.get_node_set(graph2)?;
        let edges1 = self.get_edge_set(graph1)?;
        let edges2 = self.get_edge_set(graph2)?;
        
        let mut operations = Vec::new();
        let mut cost = 0.0;
        
        // Node operations
        for node in &nodes1 {
            if !nodes2.contains(node) {
                operations.push(EditOperation::DeleteNode(node.clone()));
                cost += self.get_node_cost(node, "");
            }
        }
        
        for node in &nodes2 {
            if !nodes1.contains(node) {
                operations.push(EditOperation::InsertNode(node.clone()));
                cost += self.get_node_cost("", node);
            }
        }
        
        // Edge operations
        for edge in &edges1 {
            if !edges2.contains(edge) {
                operations.push(EditOperation::DeleteEdge(edge.0.clone(), edge.1.clone()));
                cost += self.get_edge_cost(edge, &("".to_string(), "".to_string()));
            }
        }
        
        for edge in &edges2 {
            if !edges1.contains(edge) {
                operations.push(EditOperation::InsertEdge(edge.0.clone(), edge.1.clone()));
                cost += self.get_edge_cost(&("".to_string(), "".to_string()), edge);
            }
        }
        
        Ok((cost, operations))
    }

    /// Exact edit distance using dynamic programming (simplified)
    fn exact_edit_distance(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> Result<(f64, Vec<EditOperation>)> {
        // This is a simplified version - full implementation would use
        // more sophisticated algorithms like A* or branch-and-bound
        self.approximate_edit_distance(graph1, graph2)
    }

    fn get_node_set(&self, graph: &ArrowGraph) -> Result<HashSet<String>> {
        let mut nodes = HashSet::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                nodes.insert(node_ids.value(i).to_string());
            }
        }
        
        Ok(nodes)
    }

    fn get_edge_set(&self, graph: &ArrowGraph) -> Result<HashSet<(String, String)>> {
        let mut edges = HashSet::new();
        
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
            
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                edges.insert((source, target));
            }
        }
        
        Ok(edges)
    }

    fn get_node_cost(&self, node1: &str, node2: &str) -> f64 {
        match &self.node_cost {
            CostFunction::Uniform(cost) => *cost,
            CostFunction::NodeDependent(costs) => {
                costs.get(node1).or_else(|| costs.get(node2)).copied().unwrap_or(1.0)
            }
            CostFunction::Custom(func) => func(node1, node2),
            _ => 1.0,
        }
    }

    fn get_edge_cost(&self, edge1: &(String, String), edge2: &(String, String)) -> f64 {
        match &self.edge_cost {
            CostFunction::Uniform(cost) => *cost,
            CostFunction::EdgeDependent(costs) => {
                costs.get(edge1).or_else(|| costs.get(edge2)).copied().unwrap_or(1.0)
            }
            _ => 1.0,
        }
    }

    fn compute_max_distance(&self, graph1: &ArrowGraph, graph2: &ArrowGraph) -> f64 {
        let nodes1 = graph1.node_count();
        let nodes2 = graph2.node_count();
        let edges1 = graph1.edge_count();
        let edges2 = graph2.edge_count();
        
        // Maximum possible distance is deleting all from graph1 and inserting all from graph2
        (nodes1 + nodes2 + edges1 + edges2) as f64
    }
}

impl MotifDetector {
    /// Create a new motif detector
    pub fn new(min_support: f64, max_motif_size: usize) -> Self {
        Self {
            min_support,
            max_motif_size,
            motif_type: MotifType::ConnectedSubgraphs,
            use_canonical_form: true,
        }
    }

    /// Set motif type to detect
    pub fn with_motif_type(mut self, motif_type: MotifType) -> Self {
        self.motif_type = motif_type;
        self
    }

    /// Detect motifs in a single graph
    pub fn detect_motifs(&self, graph: &ArrowGraph) -> Result<MotifResult> {
        let start_time = std::time::Instant::now();
        
        let subgraphs = self.enumerate_subgraphs(graph)?;
        let motif_counts = self.count_motif_occurrences(&subgraphs);
        let motifs = self.filter_frequent_motifs(&motif_counts, subgraphs.len());
        
        let computation_time = start_time.elapsed().as_secs_f64();
        let total_motifs = motifs.len();
        
        Ok(MotifResult {
            motifs,
            motif_counts,
            total_motifs,
            computation_time,
        })
    }

    /// Detect common motifs between multiple graphs
    pub fn detect_common_motifs(&self, graphs: &[&ArrowGraph]) -> Result<MotifResult> {
        let start_time = std::time::Instant::now();
        
        let mut all_subgraphs = Vec::new();
        for graph in graphs {
            let subgraphs = self.enumerate_subgraphs(graph)?;
            all_subgraphs.extend(subgraphs);
        }
        
        let motif_counts = self.count_motif_occurrences(&all_subgraphs);
        let motifs = self.filter_frequent_motifs(&motif_counts, all_subgraphs.len());
        
        let computation_time = start_time.elapsed().as_secs_f64();
        let total_motifs = motifs.len();
        
        Ok(MotifResult {
            motifs,
            motif_counts,
            total_motifs,
            computation_time,
        })
    }

    fn enumerate_subgraphs(&self, graph: &ArrowGraph) -> Result<Vec<SubGraph>> {
        let mut subgraphs = Vec::new();
        let adjacency = self.build_adjacency_list(graph)?;
        let nodes: Vec<_> = adjacency.keys().cloned().collect();
        
        // Enumerate connected subgraphs of increasing size
        for size in 2..=self.max_motif_size {
            for node in &nodes {
                let subgraph_candidates = self.grow_subgraph(node, size, &adjacency);
                for candidate in subgraph_candidates {
                    if self.matches_motif_type(&candidate) {
                        subgraphs.push(candidate);
                    }
                }
            }
        }
        
        Ok(subgraphs)
    }

    fn build_adjacency_list(&self, graph: &ArrowGraph) -> Result<HashMap<String, HashSet<String>>> {
        let mut adjacency = HashMap::new();
        
        // Initialize with all nodes
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                adjacency.insert(node_ids.value(i).to_string(), HashSet::new());
            }
        }
        
        // Add edges
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
            
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                
                adjacency.entry(source.clone()).or_default().insert(target.clone());
                adjacency.entry(target).or_default().insert(source); // Undirected
            }
        }
        
        Ok(adjacency)
    }

    fn grow_subgraph(&self, start_node: &str, target_size: usize, adjacency: &HashMap<String, HashSet<String>>) -> Vec<SubGraph> {
        let mut subgraphs = Vec::new();
        let mut current_subgraph = SubGraph {
            nodes: [start_node.to_string()].iter().cloned().collect(),
            edges: HashSet::new(),
            node_labels: HashMap::new(),
            edge_labels: HashMap::new(),
        };
        
        self.grow_subgraph_recursive(&mut current_subgraph, target_size, adjacency, &mut subgraphs);
        subgraphs
    }

    fn grow_subgraph_recursive(
        &self,
        current: &mut SubGraph,
        target_size: usize,
        adjacency: &HashMap<String, HashSet<String>>,
        results: &mut Vec<SubGraph>,
    ) {
        if current.nodes.len() == target_size {
            results.push(current.clone());
            return;
        }
        
        if current.nodes.len() >= target_size {
            return;
        }
        
        // Find all possible next nodes
        let mut candidates = HashSet::new();
        for node in &current.nodes {
            if let Some(neighbors) = adjacency.get(node) {
                for neighbor in neighbors {
                    if !current.nodes.contains(neighbor) {
                        candidates.insert(neighbor.clone());
                    }
                }
            }
        }
        
        // Try adding each candidate
        for candidate in candidates {
            let mut new_subgraph = current.clone();
            new_subgraph.nodes.insert(candidate.clone());
            
            // Add edges to the new node
            for existing_node in &current.nodes {
                if let Some(neighbors) = adjacency.get(existing_node) {
                    if neighbors.contains(&candidate) {
                        new_subgraph.edges.insert((existing_node.clone(), candidate.clone()));
                    }
                }
            }
            
            self.grow_subgraph_recursive(&mut new_subgraph, target_size, adjacency, results);
        }
    }

    fn matches_motif_type(&self, subgraph: &SubGraph) -> bool {
        match &self.motif_type {
            MotifType::ConnectedSubgraphs => self.is_connected(subgraph),
            MotifType::Trees => self.is_tree(subgraph),
            MotifType::Cycles => self.is_cycle(subgraph),
            MotifType::Cliques => self.is_clique(subgraph),
            MotifType::Stars => self.is_star(subgraph),
            MotifType::Paths => self.is_path(subgraph),
            MotifType::Custom(_func) => true, // Simplified
        }
    }

    fn is_connected(&self, subgraph: &SubGraph) -> bool {
        if subgraph.nodes.is_empty() {
            return true;
        }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        let start_node = subgraph.nodes.iter().next().unwrap();
        queue.push_back(start_node.clone());
        visited.insert(start_node.clone());
        
        while let Some(node) = queue.pop_front() {
            for edge in &subgraph.edges {
                let neighbor = if edge.0 == node {
                    Some(&edge.1)
                } else if edge.1 == node {
                    Some(&edge.0)
                } else {
                    None
                };
                
                if let Some(neighbor) = neighbor {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        
        visited.len() == subgraph.nodes.len()
    }

    fn is_tree(&self, subgraph: &SubGraph) -> bool {
        self.is_connected(subgraph) && subgraph.edges.len() == subgraph.nodes.len() - 1
    }

    fn is_cycle(&self, subgraph: &SubGraph) -> bool {
        self.is_connected(subgraph) && subgraph.edges.len() == subgraph.nodes.len()
    }

    fn is_clique(&self, subgraph: &SubGraph) -> bool {
        let n = subgraph.nodes.len();
        let expected_edges = n * (n - 1) / 2;
        subgraph.edges.len() == expected_edges
    }

    fn is_star(&self, subgraph: &SubGraph) -> bool {
        if subgraph.nodes.len() < 3 {
            return false;
        }
        
        // Check if there's exactly one node connected to all others
        for node in &subgraph.nodes {
            let mut connected_count = 0;
            for edge in &subgraph.edges {
                if edge.0 == *node || edge.1 == *node {
                    connected_count += 1;
                }
            }
            if connected_count == subgraph.nodes.len() - 1 {
                return true;
            }
        }
        
        false
    }

    fn is_path(&self, subgraph: &SubGraph) -> bool {
        if subgraph.nodes.len() < 2 {
            return false;
        }
        
        // Check if it's a connected graph where exactly 2 nodes have degree 1
        // and all others have degree 2
        let mut degree_counts = HashMap::new();
        
        for node in &subgraph.nodes {
            let mut degree = 0;
            for edge in &subgraph.edges {
                if edge.0 == *node || edge.1 == *node {
                    degree += 1;
                }
            }
            *degree_counts.entry(degree).or_insert(0) += 1;
        }
        
        degree_counts.get(&1) == Some(&2) && 
        degree_counts.get(&2) == Some(&(subgraph.nodes.len() - 2)) &&
        degree_counts.len() == 2
    }

    fn count_motif_occurrences(&self, subgraphs: &[SubGraph]) -> HashMap<SubGraph, usize> {
        let mut counts = HashMap::new();
        
        for subgraph in subgraphs {
            let canonical_form = if self.use_canonical_form {
                self.to_canonical_form(subgraph)
            } else {
                subgraph.clone()
            };
            
            *counts.entry(canonical_form).or_insert(0) += 1;
        }
        
        counts
    }

    fn to_canonical_form(&self, subgraph: &SubGraph) -> SubGraph {
        // Simplified canonical form - in practice, would use graph isomorphism
        let mut nodes: Vec<_> = subgraph.nodes.iter().cloned().collect();
        nodes.sort();
        
        let mut edges: Vec<_> = subgraph.edges.iter().map(|(a, b)| {
            if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) }
        }).collect();
        edges.sort();
        
        SubGraph {
            nodes: nodes.into_iter().collect(),
            edges: edges.into_iter().collect(),
            node_labels: HashMap::new(),
            edge_labels: HashMap::new(),
        }
    }

    fn filter_frequent_motifs(&self, motif_counts: &HashMap<SubGraph, usize>, total_subgraphs: usize) -> Vec<DetectedMotif> {
        let mut motifs = Vec::new();
        
        for (pattern, &count) in motif_counts {
            let support = count as f64 / total_subgraphs as f64;
            
            if support >= self.min_support {
                motifs.push(DetectedMotif {
                    pattern: pattern.clone(),
                    occurrences: vec![pattern.clone()], // Simplified
                    support,
                    significance: support, // Simplified significance calculation
                });
            }
        }
        
        // Sort by support (descending)
        motifs.sort_by(|a, b| b.support.partial_cmp(&a.support).unwrap_or(std::cmp::Ordering::Equal));
        
        motifs
    }
}

impl SubgraphMatcher {
    /// Create a new subgraph matcher
    pub fn new(algorithm: MatchingAlgorithm) -> Self {
        Self {
            algorithm,
            use_node_labels: false,
            use_edge_labels: false,
            timeout_ms: Some(5000),
        }
    }

    /// Check if query graph is a subgraph of target graph
    pub fn is_subgraph(&self, query: &ArrowGraph, target: &ArrowGraph) -> Result<MatchingResult> {
        let start_time = std::time::Instant::now();
        
        let (is_subgraph, mappings) = match &self.algorithm {
            MatchingAlgorithm::VF2 => self.vf2_matching(query, target)?,
            MatchingAlgorithm::Ullmann => self.ullmann_matching(query, target)?,
            MatchingAlgorithm::BackTrack => self.backtrack_matching(query, target)?,
            MatchingAlgorithm::LAD => self.lad_matching(query, target)?,
        };
        
        let computation_time = start_time.elapsed().as_secs_f64();
        
        Ok(MatchingResult {
            is_subgraph,
            num_mappings: mappings.len(),
            mappings,
            computation_time,
        })
    }

    /// Simplified VF2 algorithm implementation
    fn vf2_matching(&self, query: &ArrowGraph, target: &ArrowGraph) -> Result<(bool, Vec<NodeMapping>)> {
        // This is a simplified version of VF2
        // A full implementation would be much more complex
        
        let query_nodes = self.get_node_list(query)?;
        let target_nodes = self.get_node_list(target)?;
        
        if query_nodes.len() > target_nodes.len() {
            return Ok((false, Vec::new()));
        }
        
        let mut mappings = Vec::new();
        let mut current_mapping = HashMap::new();
        
        if self.vf2_recursive(&query_nodes, &target_nodes, 0, &mut current_mapping, query, target)? {
            let mapping = NodeMapping {
                query_to_target: current_mapping.clone(),
                target_to_query: current_mapping.iter().map(|(k, v)| (v.clone(), k.clone())).collect(),
            };
            mappings.push(mapping);
            Ok((true, mappings))
        } else {
            Ok((false, mappings))
        }
    }

    fn vf2_recursive(
        &self,
        query_nodes: &[String],
        target_nodes: &[String],
        query_index: usize,
        current_mapping: &mut HashMap<String, String>,
        query: &ArrowGraph,
        target: &ArrowGraph,
    ) -> Result<bool> {
        if query_index >= query_nodes.len() {
            return Ok(self.is_valid_mapping(current_mapping, query, target)?);
        }
        
        let query_node = &query_nodes[query_index];
        
        for target_node in target_nodes {
            if !current_mapping.values().any(|v| v == target_node) {
                current_mapping.insert(query_node.clone(), target_node.clone());
                
                if self.vf2_recursive(query_nodes, target_nodes, query_index + 1, current_mapping, query, target)? {
                    return Ok(true);
                }
                
                current_mapping.remove(query_node);
            }
        }
        
        Ok(false)
    }

    fn is_valid_mapping(&self, mapping: &HashMap<String, String>, query: &ArrowGraph, target: &ArrowGraph) -> Result<bool> {
        let query_edges = self.get_edge_list(query)?;
        let target_edges = self.get_edge_list(target)?;
        
        for (query_src, query_tgt) in query_edges {
            if let (Some(target_src), Some(target_tgt)) = (mapping.get(&query_src), mapping.get(&query_tgt)) {
                if !target_edges.contains(&(target_src.clone(), target_tgt.clone())) &&
                   !target_edges.contains(&(target_tgt.clone(), target_src.clone())) {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    fn ullmann_matching(&self, _query: &ArrowGraph, _target: &ArrowGraph) -> Result<(bool, Vec<NodeMapping>)> {
        // Simplified placeholder for Ullmann's algorithm
        Ok((false, Vec::new()))
    }

    fn backtrack_matching(&self, _query: &ArrowGraph, _target: &ArrowGraph) -> Result<(bool, Vec<NodeMapping>)> {
        // Simplified placeholder for backtracking algorithm
        Ok((false, Vec::new()))
    }

    fn lad_matching(&self, _query: &ArrowGraph, _target: &ArrowGraph) -> Result<(bool, Vec<NodeMapping>)> {
        // Simplified placeholder for LAD algorithm
        Ok((false, Vec::new()))
    }

    fn get_node_list(&self, graph: &ArrowGraph) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                nodes.push(node_ids.value(i).to_string());
            }
        }
        
        Ok(nodes)
    }

    fn get_edge_list(&self, graph: &ArrowGraph) -> Result<Vec<(String, String)>> {
        let mut edges = Vec::new();
        
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
            
            for i in 0..source_ids.len() {
                let source = source_ids.value(i).to_string();
                let target = target_ids.value(i).to_string();
                edges.push((source, target));
            }
        }
        
        Ok(edges)
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
        let sources = StringArray::from(vec!["A", "B", "C"]);
        let targets = StringArray::from(vec!["B", "C", "D"]);
        let weights = Float64Array::from(vec![1.0, 1.0, 1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_graph_kernel_creation() {
        let kernel = GraphKernel::new(KernelType::RandomWalk);
        assert!(matches!(kernel.kernel_type, KernelType::RandomWalk));
    }

    #[test]
    fn test_random_walk_kernel() {
        let graph1 = create_test_graph().unwrap();
        let graph2 = create_test_graph().unwrap();
        
        let kernel = GraphKernel::new(KernelType::RandomWalk)
            .with_parameters(KernelParameters {
                num_walks: 10,
                walk_length: 5,
                ..Default::default()
            });
        
        let result = kernel.compute_similarity(&graph1, &graph2).unwrap();
        assert!(result.similarity_score >= 0.0);
        assert!(result.similarity_score <= 1.0);
    }

    #[test]
    fn test_shortest_path_kernel() {
        let graph1 = create_test_graph().unwrap();
        let graph2 = create_test_graph().unwrap();
        
        let kernel = GraphKernel::new(KernelType::ShortestPath);
        let result = kernel.compute_similarity(&graph1, &graph2).unwrap();
        
        assert!(result.similarity_score >= 0.0);
        assert!(result.computation_time >= 0.0);
    }

    #[test]
    fn test_weisfeiler_lehman_kernel() {
        let graph1 = create_test_graph().unwrap();
        let graph2 = create_test_graph().unwrap();
        
        let kernel = GraphKernel::new(KernelType::WeisfeilerLehman)
            .with_parameters(KernelParameters {
                wl_iterations: 2,
                ..Default::default()
            });
        
        let result = kernel.compute_similarity(&graph1, &graph2).unwrap();
        assert!(result.similarity_score >= 0.0);
    }

    #[test]
    fn test_graph_edit_distance() {
        let graph1 = create_test_graph().unwrap();
        let graph2 = create_test_graph().unwrap();
        
        let ged = GraphEditDistance::new()
            .with_costs(CostFunction::Uniform(1.0), CostFunction::Uniform(1.0));
        
        let result = ged.compute_distance(&graph1, &graph2).unwrap();
        assert!(result.distance >= 0.0);
        assert!(result.normalized_distance >= 0.0);
        assert!(result.normalized_distance <= 1.0);
    }

    #[test]
    fn test_motif_detector() {
        let graph = create_test_graph().unwrap();
        
        let detector = MotifDetector::new(0.1, 3)
            .with_motif_type(MotifType::ConnectedSubgraphs);
        
        let result = detector.detect_motifs(&graph).unwrap();
        assert!(result.computation_time >= 0.0);
        assert_eq!(result.total_motifs, result.motifs.len());
    }

    #[test]
    fn test_subgraph_matcher() {
        let query = create_test_graph().unwrap();
        let target = create_test_graph().unwrap();
        
        let matcher = SubgraphMatcher::new(MatchingAlgorithm::VF2);
        let result = matcher.is_subgraph(&query, &target).unwrap();
        
        assert!(result.computation_time >= 0.0);
        // Same graph should be a subgraph of itself
        assert!(result.is_subgraph);
    }

    #[test]
    fn test_subgraph_connected() {
        let subgraph = SubGraph {
            nodes: ["A", "B", "C"].iter().map(|s| s.to_string()).collect(),
            edges: [("A", "B"), ("B", "C")].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            node_labels: HashMap::new(),
            edge_labels: HashMap::new(),
        };
        
        let detector = MotifDetector::new(0.1, 3);
        assert!(detector.is_connected(&subgraph));
        assert!(detector.is_path(&subgraph));
        assert!(!detector.is_cycle(&subgraph));
        assert!(!detector.is_clique(&subgraph));
    }

    #[test]
    fn test_subgraph_clique() {
        let subgraph = SubGraph {
            nodes: ["A", "B", "C"].iter().map(|s| s.to_string()).collect(),
            edges: [("A", "B"), ("B", "C"), ("A", "C")].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            node_labels: HashMap::new(),
            edge_labels: HashMap::new(),
        };
        
        let detector = MotifDetector::new(0.1, 3);
        assert!(detector.is_connected(&subgraph));
        assert!(detector.is_clique(&subgraph));
        assert!(!detector.is_path(&subgraph));
        assert!(detector.is_cycle(&subgraph)); // Triangle is also a cycle
    }

    #[test]
    fn test_kernel_parameters() {
        let params = KernelParameters {
            walk_length: 20,
            num_walks: 200,
            wl_iterations: 5,
            ..Default::default()
        };
        
        let kernel = GraphKernel::new(KernelType::RandomWalk).with_parameters(params);
        assert_eq!(kernel.parameters.walk_length, 20);
        assert_eq!(kernel.parameters.num_walks, 200);
        assert_eq!(kernel.parameters.wl_iterations, 5);
    }
}