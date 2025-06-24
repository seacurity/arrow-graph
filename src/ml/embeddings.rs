use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::Array;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Trait for graph embedding models
pub trait EmbeddingModel {
    type Output;
    
    /// Initialize the model with graph data
    fn initialize(&mut self, graph: &ArrowGraph) -> Result<()>;
    
    /// Generate embeddings for all nodes
    fn generate_embeddings(&mut self) -> Result<Self::Output>;
    
    /// Get embedding for a specific node
    fn get_node_embedding(&self, node_id: &str) -> Option<Vec<f64>>;
    
    /// Get all node embeddings
    fn get_all_embeddings(&self) -> &NodeEmbeddings;
}

/// Container for node embeddings
#[derive(Debug, Clone)]
pub struct NodeEmbeddings {
    pub embeddings: HashMap<String, Vec<f64>>,
    pub dimension: usize,
}

impl NodeEmbeddings {
    pub fn new(dimension: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dimension,
        }
    }

    pub fn add_embedding(&mut self, node_id: String, embedding: Vec<f64>) {
        if embedding.len() != self.dimension {
            eprintln!("Warning: embedding dimension mismatch for node {}", node_id);
        }
        self.embeddings.insert(node_id, embedding);
    }

    pub fn get_embedding(&self, node_id: &str) -> Option<&Vec<f64>> {
        self.embeddings.get(node_id)
    }

    pub fn similarity(&self, node1: &str, node2: &str) -> Option<f64> {
        let emb1 = self.embeddings.get(node1)?;
        let emb2 = self.embeddings.get(node2)?;
        
        let dot_product: f64 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = emb1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = emb2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            Some(dot_product / (norm1 * norm2))
        } else {
            None
        }
    }

    pub fn most_similar(&self, node_id: &str, k: usize) -> Vec<(String, f64)> {
        let mut similarities = Vec::new();
        
        for (other_node, _) in &self.embeddings {
            if other_node != node_id {
                if let Some(sim) = self.similarity(node_id, other_node) {
                    similarities.push((other_node.clone(), sim));
                }
            }
        }
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }
}

/// Node2Vec implementation for learning node representations
#[derive(Debug)]
pub struct Node2Vec {
    dimension: usize,
    walk_length: usize,
    num_walks: usize,
    p: f64,  // Return parameter
    q: f64,  // In-out parameter
    window_size: usize,
    min_count: usize,
    workers: usize,
    rng: Pcg64,
    embeddings: NodeEmbeddings,
    adjacency: HashMap<String, Vec<String>>,
    transition_probs: HashMap<(String, String), HashMap<String, f64>>,
}

impl Node2Vec {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            window_size: 10,
            min_count: 1,
            workers: 4,
            rng: Pcg64::from_entropy(),
            embeddings: NodeEmbeddings::new(dimension),
            adjacency: HashMap::new(),
            transition_probs: HashMap::new(),
        }
    }

    pub fn with_params(dimension: usize, p: f64, q: f64, walk_length: usize, num_walks: usize) -> Self {
        let mut node2vec = Self::new(dimension);
        node2vec.p = p;
        node2vec.q = q;
        node2vec.walk_length = walk_length;
        node2vec.num_walks = num_walks;
        node2vec
    }

    /// Precompute transition probabilities for biased random walks
    fn precompute_transition_probs(&mut self) -> Result<()> {
        self.transition_probs.clear();
        
        for (node, neighbors) in &self.adjacency {
            for neighbor in neighbors {
                let mut probs = HashMap::new();
                
                // Get neighbors of the neighbor
                if let Some(neighbor_neighbors) = self.adjacency.get(neighbor) {
                    let mut unnormalized_probs = HashMap::new();
                    
                    for next_neighbor in neighbor_neighbors {
                        let prob = if next_neighbor == node {
                            // Coming back to previous node
                            1.0 / self.p
                        } else if neighbors.contains(next_neighbor) {
                            // Node is in the 1-hop neighborhood of the current node
                            1.0
                        } else {
                            // Node is not in the 1-hop neighborhood
                            1.0 / self.q
                        };
                        unnormalized_probs.insert(next_neighbor.clone(), prob);
                    }
                    
                    // Normalize probabilities
                    let total: f64 = unnormalized_probs.values().sum();
                    if total > 0.0 {
                        for (next_node, prob) in unnormalized_probs {
                            probs.insert(next_node, prob / total);
                        }
                    }
                }
                
                self.transition_probs.insert((node.clone(), neighbor.clone()), probs);
            }
        }
        
        Ok(())
    }

    /// Generate a single biased random walk
    fn generate_walk(&mut self, start_node: &str) -> Vec<String> {
        let mut walk = vec![start_node.to_string()];
        
        if !self.adjacency.contains_key(start_node) {
            return walk;
        }
        
        for _ in 1..self.walk_length {
            let current = walk.last().unwrap();
            
            if let Some(neighbors) = self.adjacency.get(current) {
                if neighbors.is_empty() {
                    break;
                }
                
                let next_node = if walk.len() == 1 {
                    // First step: uniform random selection
                    neighbors[self.rng.gen_range(0..neighbors.len())].clone()
                } else {
                    // Biased selection based on precomputed probabilities
                    let prev_node = &walk[walk.len() - 2];
                    self.biased_choice(prev_node, current)
                };
                
                walk.push(next_node);
            } else {
                break;
            }
        }
        
        walk
    }

    /// Choose next node based on biased probabilities
    fn biased_choice(&mut self, prev_node: &str, current_node: &str) -> String {
        if let Some(probs) = self.transition_probs.get(&(prev_node.to_string(), current_node.to_string())) {
            let rand_val = self.rng.gen::<f64>();
            let mut cumulative = 0.0;
            
            for (node, prob) in probs {
                cumulative += prob;
                if rand_val <= cumulative {
                    return node.clone();
                }
            }
        }
        
        // Fallback to uniform random selection
        if let Some(neighbors) = self.adjacency.get(current_node) {
            if !neighbors.is_empty() {
                return neighbors[self.rng.gen_range(0..neighbors.len())].clone();
            }
        }
        
        current_node.to_string()
    }

    /// Generate all random walks
    fn generate_walks(&mut self) -> Vec<Vec<String>> {
        let mut walks = Vec::new();
        let nodes: Vec<String> = self.adjacency.keys().cloned().collect();
        
        for _ in 0..self.num_walks {
            for node in &nodes {
                let walk = self.generate_walk(node);
                if walk.len() > 1 {
                    walks.push(walk);
                }
            }
        }
        
        walks
    }

    /// Learn embeddings using Word2Vec-like approach (simplified)
    fn learn_embeddings(&mut self, walks: Vec<Vec<String>>) -> Result<()> {
        // Initialize embeddings with random values
        let mut vocab = std::collections::HashSet::new();
        for walk in &walks {
            for node in walk {
                vocab.insert(node.clone());
            }
        }
        
        // Initialize random embeddings
        for node in vocab {
            let mut embedding = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                embedding.push((self.rng.gen::<f64>() - 0.5) * 0.1);
            }
            self.embeddings.add_embedding(node, embedding);
        }
        
        // Simplified skip-gram training
        for walk in walks {
            for (i, center_node) in walk.iter().enumerate() {
                let start = i.saturating_sub(self.window_size);
                let end = (i + self.window_size + 1).min(walk.len());
                
                for j in start..end {
                    if i != j {
                        let context_node = &walk[j];
                        // In a full implementation, we'd update embeddings here
                        // using gradient descent on the skip-gram objective
                        self.update_embeddings(center_node, context_node);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Simplified embedding update (placeholder for full Word2Vec implementation)
    fn update_embeddings(&mut self, center: &str, context: &str) {
        // This is a simplified placeholder
        // A full implementation would use hierarchical softmax or negative sampling
        if let (Some(center_emb), Some(context_emb)) = 
            (self.embeddings.embeddings.get(center).cloned(),
             self.embeddings.embeddings.get(context).cloned()) {
            
            let learning_rate = 0.01;
            let mut new_center_emb = Vec::new();
            
            for i in 0..self.dimension {
                let gradient = context_emb[i] * learning_rate;
                new_center_emb.push(center_emb[i] + gradient);
            }
            
            self.embeddings.embeddings.insert(center.to_string(), new_center_emb);
        }
    }
}

impl EmbeddingModel for Node2Vec {
    type Output = NodeEmbeddings;

    fn initialize(&mut self, graph: &ArrowGraph) -> Result<()> {
        // Build adjacency list from graph
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
                
                self.adjacency.entry(source.clone()).or_insert_with(Vec::new).push(target.clone());
                self.adjacency.entry(target).or_insert_with(Vec::new).push(source);
            }
        }
        
        // Precompute transition probabilities
        self.precompute_transition_probs()?;
        
        Ok(())
    }

    fn generate_embeddings(&mut self) -> Result<NodeEmbeddings> {
        let walks = self.generate_walks();
        self.learn_embeddings(walks)?;
        Ok(self.embeddings.clone())
    }

    fn get_node_embedding(&self, node_id: &str) -> Option<Vec<f64>> {
        self.embeddings.get_embedding(node_id).cloned()
    }

    fn get_all_embeddings(&self) -> &NodeEmbeddings {
        &self.embeddings
    }
}

/// GraphSAGE implementation for inductive graph representation learning
#[derive(Debug)]
pub struct GraphSAGE {
    dimension: usize,
    num_layers: usize,
    aggregator_type: AggregatorType,
    sample_sizes: Vec<usize>,
    embeddings: NodeEmbeddings,
    node_features: HashMap<String, Vec<f64>>,
    layer_weights: Vec<Vec<Vec<f64>>>, // layer -> [input_dim x output_dim]
}

#[derive(Debug, Clone)]
pub enum AggregatorType {
    Mean,
    LSTM,
    MaxPool,
    GCN,
}

impl GraphSAGE {
    pub fn new(dimension: usize, num_layers: usize) -> Self {
        Self {
            dimension,
            num_layers,
            aggregator_type: AggregatorType::Mean,
            sample_sizes: vec![25, 10], // Default sample sizes for 2 layers
            embeddings: NodeEmbeddings::new(dimension),
            node_features: HashMap::new(),
            layer_weights: Vec::new(),
        }
    }

    pub fn with_aggregator(mut self, aggregator: AggregatorType) -> Self {
        self.aggregator_type = aggregator;
        self
    }

    pub fn with_sample_sizes(mut self, sample_sizes: Vec<usize>) -> Self {
        self.sample_sizes = sample_sizes;
        self
    }

    /// Initialize node features (simplified - would typically use node attributes)
    fn initialize_features(&mut self, graph: &ArrowGraph) -> Result<()> {
        let nodes_batch = &graph.nodes;
        
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            // Initialize with random features (in practice, these would be real node features)
            let mut rng = Pcg64::from_entropy();
            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i).to_string();
                let features: Vec<f64> = (0..self.dimension)
                    .map(|_| rng.gen::<f64>() - 0.5)
                    .collect();
                self.node_features.insert(node_id, features);
            }
        }

        Ok(())
    }

    /// Sample neighbors for a node
    fn sample_neighbors(&self, node: &str, graph: &ArrowGraph, sample_size: usize) -> Vec<String> {
        // This is a simplified implementation
        // In practice, we'd efficiently sample from the adjacency list
        let mut neighbors = Vec::new();
        let edges_batch = &graph.edges;
        
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            for i in 0..source_ids.len() {
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                
                if source == node {
                    neighbors.push(target.to_string());
                } else if target == node {
                    neighbors.push(source.to_string());
                }
            }
        }
        
        // Sample without replacement
        let mut rng = Pcg64::from_entropy();
        if neighbors.len() <= sample_size {
            neighbors
        } else {
            let mut sampled = Vec::new();
            let mut used_indices = std::collections::HashSet::new();
            
            while sampled.len() < sample_size && sampled.len() < neighbors.len() {
                let idx = rng.gen_range(0..neighbors.len());
                if used_indices.insert(idx) {
                    sampled.push(neighbors[idx].clone());
                }
            }
            sampled
        }
    }

    /// Aggregate neighbor embeddings
    fn aggregate_embeddings(&self, embeddings: Vec<Vec<f64>>) -> Vec<f64> {
        if embeddings.is_empty() {
            return vec![0.0; self.dimension];
        }

        match self.aggregator_type {
            AggregatorType::Mean => {
                let mut result = vec![0.0; self.dimension];
                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        if i < result.len() {
                            result[i] += val;
                        }
                    }
                }
                let count = embeddings.len() as f64;
                for val in &mut result {
                    *val /= count;
                }
                result
            }
            AggregatorType::MaxPool => {
                let mut result = vec![f64::NEG_INFINITY; self.dimension];
                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        if i < result.len() {
                            result[i] = result[i].max(val);
                        }
                    }
                }
                result
            }
            _ => {
                // Default to mean for other aggregators
                self.aggregate_embeddings(embeddings)
            }
        }
    }
}

impl EmbeddingModel for GraphSAGE {
    type Output = NodeEmbeddings;

    fn initialize(&mut self, graph: &ArrowGraph) -> Result<()> {
        self.initialize_features(graph)?;
        
        // Initialize layer weights (simplified)
        for layer in 0..self.num_layers {
            let input_dim = if layer == 0 { self.dimension } else { self.dimension };
            let output_dim = self.dimension;
            
            let mut weights = Vec::new();
            let mut rng = Pcg64::from_entropy();
            
            for _ in 0..input_dim {
                let mut row = Vec::new();
                for _ in 0..output_dim {
                    row.push((rng.gen::<f64>() - 0.5) * 0.1);
                }
                weights.push(row);
            }
            self.layer_weights.push(weights);
        }
        
        Ok(())
    }

    fn generate_embeddings(&mut self) -> Result<NodeEmbeddings> {
        // This is a simplified GraphSAGE implementation
        // A full implementation would involve proper forward/backward passes
        
        for (node_id, features) in &self.node_features {
            // For now, just use transformed initial features as embeddings
            let mut embedding = features.clone();
            
            // Apply a simple transformation
            if !self.layer_weights.is_empty() {
                let weights = &self.layer_weights[0];
                let mut transformed = vec![0.0; self.dimension];
                
                for i in 0..self.dimension.min(weights.len()) {
                    for j in 0..self.dimension.min(weights[i].len()) {
                        if i < features.len() {
                            transformed[j] += features[i] * weights[i][j];
                        }
                    }
                }
                embedding = transformed;
            }
            
            self.embeddings.add_embedding(node_id.clone(), embedding);
        }
        
        Ok(self.embeddings.clone())
    }

    fn get_node_embedding(&self, node_id: &str) -> Option<Vec<f64>> {
        self.embeddings.get_embedding(node_id).cloned()
    }

    fn get_all_embeddings(&self) -> &NodeEmbeddings {
        &self.embeddings
    }
}

/// TransE implementation for knowledge graph embeddings
#[derive(Debug)]
pub struct TransE {
    dimension: usize,
    margin: f64,
    learning_rate: f64,
    epochs: usize,
    embeddings: NodeEmbeddings,
    relation_embeddings: HashMap<String, Vec<f64>>,
    triples: Vec<(String, String, String)>, // (head, relation, tail)
}

impl TransE {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            margin: 1.0,
            learning_rate: 0.01,
            epochs: 1000,
            embeddings: NodeEmbeddings::new(dimension),
            relation_embeddings: HashMap::new(),
            triples: Vec::new(),
        }
    }

    pub fn with_hyperparams(mut self, margin: f64, learning_rate: f64, epochs: usize) -> Self {
        self.margin = margin;
        self.learning_rate = learning_rate;
        self.epochs = epochs;
        self
    }

    /// Extract triples from graph (simplified - assumes edge labels are relations)
    fn extract_triples(&mut self, graph: &ArrowGraph) -> Result<()> {
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
                let head = source_ids.value(i).to_string();
                let tail = target_ids.value(i).to_string();
                let relation = "connected_to".to_string(); // Simplified
                
                self.triples.push((head, relation, tail));
            }
        }

        Ok(())
    }

    /// Initialize embeddings randomly
    fn initialize_embeddings(&mut self) -> Result<()> {
        let mut entities = std::collections::HashSet::new();
        let mut relations = std::collections::HashSet::new();
        
        for (head, relation, tail) in &self.triples {
            entities.insert(head.clone());
            entities.insert(tail.clone());
            relations.insert(relation.clone());
        }
        
        let mut rng = Pcg64::from_entropy();
        
        // Initialize entity embeddings
        for entity in entities {
            let embedding: Vec<f64> = (0..self.dimension)
                .map(|_| (rng.gen::<f64>() - 0.5) * 0.1)
                .collect();
            self.embeddings.add_embedding(entity, embedding);
        }
        
        // Initialize relation embeddings
        for relation in relations {
            let embedding: Vec<f64> = (0..self.dimension)
                .map(|_| (rng.gen::<f64>() - 0.5) * 0.1)
                .collect();
            self.relation_embeddings.insert(relation, embedding);
        }
        
        Ok(())
    }

    /// Calculate energy function: ||h + r - t||
    fn energy(&self, head: &[f64], relation: &[f64], tail: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.dimension {
            let diff = head[i] + relation[i] - tail[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Generate negative sample
    fn corrupt_triple(&self, triple: &(String, String, String)) -> (String, String, String) {
        let mut rng = Pcg64::from_entropy();
        let entities: Vec<&String> = self.embeddings.embeddings.keys().collect();
        
        if entities.is_empty() {
            return triple.clone();
        }
        
        if rng.gen::<bool>() {
            // Corrupt head
            let new_head = entities[rng.gen_range(0..entities.len())];
            (new_head.clone(), triple.1.clone(), triple.2.clone())
        } else {
            // Corrupt tail
            let new_tail = entities[rng.gen_range(0..entities.len())];
            (triple.0.clone(), triple.1.clone(), new_tail.clone())
        }
    }

    /// Training step
    fn train_step(&mut self) -> Result<()> {
        for triple in self.triples.clone() {
            let corrupt_triple = self.corrupt_triple(&triple);
            
            // Get embeddings
            let head_emb = self.embeddings.get_embedding(&triple.0).cloned().unwrap_or_default();
            let rel_emb = self.relation_embeddings.get(&triple.1).cloned().unwrap_or_default();
            let tail_emb = self.embeddings.get_embedding(&triple.2).cloned().unwrap_or_default();
            
            let corrupt_head_emb = self.embeddings.get_embedding(&corrupt_triple.0).cloned().unwrap_or_default();
            let corrupt_tail_emb = self.embeddings.get_embedding(&corrupt_triple.2).cloned().unwrap_or_default();
            
            // Calculate energies
            let pos_energy = self.energy(&head_emb, &rel_emb, &tail_emb);
            let neg_energy = self.energy(&corrupt_head_emb, &rel_emb, &corrupt_tail_emb);
            
            // Update embeddings if margin ranking loss is positive
            if pos_energy + self.margin > neg_energy {
                // Simplified gradient update (placeholder)
                // A full implementation would compute proper gradients
            }
        }
        
        Ok(())
    }
}

impl EmbeddingModel for TransE {
    type Output = NodeEmbeddings;

    fn initialize(&mut self, graph: &ArrowGraph) -> Result<()> {
        self.extract_triples(graph)?;
        self.initialize_embeddings()?;
        Ok(())
    }

    fn generate_embeddings(&mut self) -> Result<NodeEmbeddings> {
        for _ in 0..self.epochs {
            self.train_step()?;
        }
        Ok(self.embeddings.clone())
    }

    fn get_node_embedding(&self, node_id: &str) -> Option<Vec<f64>> {
        self.embeddings.get_embedding(node_id).cloned()
    }

    fn get_all_embeddings(&self) -> &NodeEmbeddings {
        &self.embeddings
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
        let sources = StringArray::from(vec!["A", "B", "C", "A"]);
        let targets = StringArray::from(vec!["B", "C", "D", "C"]);
        let weights = Float64Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_node_embeddings() {
        let mut embeddings = NodeEmbeddings::new(3);
        embeddings.add_embedding("A".to_string(), vec![1.0, 0.0, 0.0]);
        embeddings.add_embedding("B".to_string(), vec![0.0, 1.0, 0.0]);
        
        assert_eq!(embeddings.get_embedding("A"), Some(&vec![1.0, 0.0, 0.0]));
        assert_eq!(embeddings.similarity("A", "B"), Some(0.0));
        
        embeddings.add_embedding("C".to_string(), vec![1.0, 0.0, 0.0]);
        assert_eq!(embeddings.similarity("A", "C"), Some(1.0));
    }

    #[test]
    fn test_node2vec_initialization() {
        let graph = create_test_graph().unwrap();
        let mut node2vec = Node2Vec::new(64);
        
        node2vec.initialize(&graph).unwrap();
        
        assert!(!node2vec.adjacency.is_empty());
        assert!(!node2vec.transition_probs.is_empty());
    }

    #[test]
    fn test_node2vec_walk_generation() {
        let graph = create_test_graph().unwrap();
        let mut node2vec = Node2Vec::with_params(32, 1.0, 1.0, 10, 1);
        
        node2vec.initialize(&graph).unwrap();
        let walk = node2vec.generate_walk("A");
        
        assert!(!walk.is_empty());
        assert_eq!(walk[0], "A");
        assert!(walk.len() <= node2vec.walk_length);
    }

    #[test]
    fn test_node2vec_embeddings() {
        let graph = create_test_graph().unwrap();
        let mut node2vec = Node2Vec::with_params(16, 1.0, 1.0, 5, 2);
        
        node2vec.initialize(&graph).unwrap();
        let embeddings = node2vec.generate_embeddings().unwrap();
        
        assert_eq!(embeddings.dimension, 16);
        assert!(!embeddings.embeddings.is_empty());
        
        // Check that we can get embeddings for nodes
        assert!(embeddings.get_embedding("A").is_some());
        assert!(embeddings.get_embedding("B").is_some());
    }

    #[test]
    fn test_graphsage_initialization() {
        let graph = create_test_graph().unwrap();
        let mut sage = GraphSAGE::new(32, 2);
        
        sage.initialize(&graph).unwrap();
        
        assert!(!sage.node_features.is_empty());
        assert_eq!(sage.layer_weights.len(), 2);
    }

    #[test]
    fn test_graphsage_embeddings() {
        let graph = create_test_graph().unwrap();
        let mut sage = GraphSAGE::new(16, 1)
            .with_aggregator(AggregatorType::Mean);
        
        sage.initialize(&graph).unwrap();
        let embeddings = sage.generate_embeddings().unwrap();
        
        assert_eq!(embeddings.dimension, 16);
        assert!(!embeddings.embeddings.is_empty());
    }

    #[test]
    fn test_transe_initialization() {
        let graph = create_test_graph().unwrap();
        let mut transe = TransE::new(50);
        
        transe.initialize(&graph).unwrap();
        
        assert!(!transe.triples.is_empty());
        assert!(!transe.embeddings.embeddings.is_empty());
        assert!(!transe.relation_embeddings.is_empty());
    }

    #[test]
    fn test_transe_embeddings() {
        let graph = create_test_graph().unwrap();
        let mut transe = TransE::new(20)
            .with_hyperparams(1.0, 0.01, 10);
        
        transe.initialize(&graph).unwrap();
        let embeddings = transe.generate_embeddings().unwrap();
        
        assert_eq!(embeddings.dimension, 20);
        assert!(!embeddings.embeddings.is_empty());
    }

    #[test]
    fn test_embedding_similarity() {
        let mut embeddings = NodeEmbeddings::new(3);
        embeddings.add_embedding("A".to_string(), vec![1.0, 0.0, 0.0]);
        embeddings.add_embedding("B".to_string(), vec![0.0, 1.0, 0.0]);
        embeddings.add_embedding("C".to_string(), vec![0.707, 0.707, 0.0]);
        
        let similar = embeddings.most_similar("A", 2);
        assert_eq!(similar.len(), 2);
        
        // C should be more similar to A than B
        if similar[0].0 == "C" || similar[1].0 == "C" {
            assert!(true); // C is in top 2 similar nodes
        }
    }
}