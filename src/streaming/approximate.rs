use crate::error::Result;
use crate::streaming::incremental::IncrementalGraphProcessor;
use arrow::array::Array;
use std::hash::{Hash, Hasher};

/// Approximate algorithms for large-scale graph analytics
/// These algorithms trade accuracy for performance and memory efficiency

/// HyperLogLog implementation for approximate cardinality estimation
/// Useful for counting unique nodes, edges, or other graph elements
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    buckets: Vec<u8>,
    bucket_count: usize,
    alpha: f64,
}

impl HyperLogLog {
    /// Create new HyperLogLog with specified precision
    /// precision: number of bits for bucket selection (4-16)
    pub fn new(precision: u8) -> Self {
        let precision = precision.clamp(4, 16);
        let bucket_count = 1 << precision; // 2^precision
        
        let alpha = match bucket_count {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / bucket_count as f64),
        };

        Self {
            buckets: vec![0; bucket_count],
            bucket_count,
            alpha,
        }
    }

    /// Add an element to the HyperLogLog
    pub fn add<T: Hash>(&mut self, element: &T) {
        let hash = self.hash_element(element);
        let bucket_index = (hash >> (64 - self.precision_bits())) as usize;
        let leading_zeros = (hash << self.precision_bits()).leading_zeros() as u8 + 1;
        
        self.buckets[bucket_index] = self.buckets[bucket_index].max(leading_zeros);
    }

    /// Estimate the cardinality
    pub fn estimate(&self) -> f64 {
        let raw_estimate = self.alpha * (self.bucket_count as f64).powi(2) / 
            self.buckets.iter().map(|&b| 2.0_f64.powf(-(b as f64))).sum::<f64>();

        // Apply small range correction
        if raw_estimate <= 2.5 * self.bucket_count as f64 {
            let zero_count = self.buckets.iter().filter(|&&b| b == 0).count();
            if zero_count != 0 {
                return (self.bucket_count as f64) * (self.bucket_count as f64 / zero_count as f64).ln();
            }
        }

        // Apply large range correction for 64-bit hash
        if raw_estimate <= (1.0 / 30.0) * (1u64 << 32) as f64 {
            raw_estimate
        } else {
            -((1u64 << 32) as f64) * (1.0 - raw_estimate / ((1u64 << 32) as f64)).ln()
        }
    }

    /// Merge with another HyperLogLog
    pub fn merge(&mut self, other: &HyperLogLog) -> Result<()> {
        if self.bucket_count != other.bucket_count {
            return Err(crate::error::GraphError::graph_construction("HyperLogLog bucket counts must match for merge"));
        }

        for i in 0..self.bucket_count {
            self.buckets[i] = self.buckets[i].max(other.buckets[i]);
        }

        Ok(())
    }

    fn hash_element<T: Hash>(&self, element: &T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        element.hash(&mut hasher);
        hasher.finish()
    }

    fn precision_bits(&self) -> u32 {
        (self.bucket_count as f64).log2() as u32
    }
}

/// Count-Min Sketch for approximate frequency counting
/// Useful for tracking edge weights, node degrees, or other frequencies
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    counters: Vec<Vec<u32>>,
    hash_functions: Vec<HashFunction>,
    width: usize,
    depth: usize,
}

#[derive(Debug, Clone)]
struct HashFunction {
    a: u64,
    b: u64,
    p: u64, // Large prime
}

impl CountMinSketch {
    /// Create new Count-Min Sketch
    /// width: number of buckets per row
    /// depth: number of hash functions/rows
    pub fn new(width: usize, depth: usize) -> Self {
        let mut hash_functions = Vec::new();
        let p = 2147483647u64; // Large prime

        for i in 0..depth {
            hash_functions.push(HashFunction {
                a: (i as u64 * 2 + 1) * 1000000007,
                b: (i as u64 + 1) * 1000000009,
                p,
            });
        }

        Self {
            counters: vec![vec![0; width]; depth],
            hash_functions,
            width,
            depth,
        }
    }

    /// Add an element with count
    pub fn add<T: Hash>(&mut self, element: &T, count: u32) {
        let hash = self.hash_element(element);
        
        for (i, hash_func) in self.hash_functions.iter().enumerate() {
            let index = hash_func.hash(hash) % self.width as u64;
            self.counters[i][index as usize] += count;
        }
    }

    /// Estimate the count of an element
    pub fn estimate<T: Hash>(&self, element: &T) -> u32 {
        let hash = self.hash_element(element);
        let mut min_count = u32::MAX;
        
        for (i, hash_func) in self.hash_functions.iter().enumerate() {
            let index = hash_func.hash(hash) % self.width as u64;
            min_count = min_count.min(self.counters[i][index as usize]);
        }
        
        min_count
    }

    /// Merge with another Count-Min Sketch
    pub fn merge(&mut self, other: &CountMinSketch) -> Result<()> {
        if self.width != other.width || self.depth != other.depth {
            return Err(crate::error::GraphError::graph_construction("Count-Min Sketch dimensions must match for merge"));
        }

        for i in 0..self.depth {
            for j in 0..self.width {
                self.counters[i][j] += other.counters[i][j];
            }
        }

        Ok(())
    }

    fn hash_element<T: Hash>(&self, element: &T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        element.hash(&mut hasher);
        hasher.finish()
    }
}

impl HashFunction {
    fn hash(&self, x: u64) -> u64 {
        ((self.a.wrapping_mul(x).wrapping_add(self.b)) % self.p) % self.p
    }
}

/// Bloom Filter for approximate set membership testing
/// Useful for checking if nodes/edges exist without storing the full set
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bit_array: Vec<bool>,
    size: usize,
    hash_count: u32,
    element_count: usize,
}

impl BloomFilter {
    /// Create new Bloom Filter
    /// expected_elements: expected number of elements
    /// false_positive_rate: desired false positive rate (0.0 - 1.0)
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(expected_elements, false_positive_rate);
        let hash_count = Self::optimal_hash_count(size, expected_elements);

        Self {
            bit_array: vec![false; size],
            size,
            hash_count,
            element_count: 0,
        }
    }

    /// Add an element to the filter
    pub fn add<T: Hash>(&mut self, element: &T) {
        let hashes = self.hash_element(element);
        
        for i in 0..self.hash_count {
            let index = (hashes.0.wrapping_add((i as u64).wrapping_mul(hashes.1))) % self.size as u64;
            self.bit_array[index as usize] = true;
        }
        
        self.element_count += 1;
    }

    /// Check if an element might be in the set
    /// Returns true if element might be present (could be false positive)
    /// Returns false if element is definitely not present
    pub fn contains<T: Hash>(&self, element: &T) -> bool {
        let hashes = self.hash_element(element);
        
        for i in 0..self.hash_count {
            let index = (hashes.0.wrapping_add((i as u64).wrapping_mul(hashes.1))) % self.size as u64;
            if !self.bit_array[index as usize] {
                return false;
            }
        }
        
        true
    }

    /// Get current false positive probability estimate
    pub fn false_positive_probability(&self) -> f64 {
        let filled_ratio = self.bit_array.iter().filter(|&&b| b).count() as f64 / self.size as f64;
        filled_ratio.powf(self.hash_count as f64)
    }

    /// Union with another Bloom Filter
    pub fn union(&mut self, other: &BloomFilter) -> Result<()> {
        if self.size != other.size || self.hash_count != other.hash_count {
            return Err(crate::error::GraphError::graph_construction("Bloom Filter parameters must match for union"));
        }

        for i in 0..self.size {
            self.bit_array[i] |= other.bit_array[i];
        }

        self.element_count += other.element_count;
        Ok(())
    }

    fn optimal_size(expected_elements: usize, false_positive_rate: f64) -> usize {
        let m = -(expected_elements as f64 * false_positive_rate.ln()) / (2.0_f64.ln().powi(2));
        m.ceil() as usize
    }

    fn optimal_hash_count(size: usize, expected_elements: usize) -> u32 {
        let k = (size as f64 / expected_elements as f64) * 2.0_f64.ln();
        k.round().max(1.0) as u32
    }

    fn hash_element<T: Hash>(&self, element: &T) -> (u64, u64) {
        let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
        let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
        
        element.hash(&mut hasher1);
        (element, 1u8).hash(&mut hasher2); // Add salt for second hash
        
        (hasher1.finish(), hasher2.finish())
    }
}

/// Approximate graph analytics processor
/// Combines multiple approximate algorithms for scalable graph analysis
#[derive(Debug)]
pub struct ApproximateGraphProcessor {
    // Cardinality estimators
    unique_nodes: HyperLogLog,
    unique_edges: HyperLogLog,
    unique_triangles: HyperLogLog,
    
    // Frequency counters
    node_degrees: CountMinSketch,
    edge_weights: CountMinSketch,
    
    // Set membership
    active_nodes: BloomFilter,
    recent_edges: BloomFilter,
    
    // Configuration
    config: ApproximateConfig,
}

/// Configuration for approximate algorithms
#[derive(Debug, Clone)]
pub struct ApproximateConfig {
    pub hyperloglog_precision: u8,
    pub count_min_width: usize,
    pub count_min_depth: usize,
    pub bloom_expected_elements: usize,
    pub bloom_false_positive_rate: f64,
}

impl Default for ApproximateConfig {
    fn default() -> Self {
        Self {
            hyperloglog_precision: 12, // ~1.6% error
            count_min_width: 1000,
            count_min_depth: 5,
            bloom_expected_elements: 10000,
            bloom_false_positive_rate: 0.01, // 1% false positive rate
        }
    }
}

impl ApproximateGraphProcessor {
    /// Create new approximate graph processor
    pub fn new(config: ApproximateConfig) -> Self {
        Self {
            unique_nodes: HyperLogLog::new(config.hyperloglog_precision),
            unique_edges: HyperLogLog::new(config.hyperloglog_precision),
            unique_triangles: HyperLogLog::new(config.hyperloglog_precision),
            node_degrees: CountMinSketch::new(config.count_min_width, config.count_min_depth),
            edge_weights: CountMinSketch::new(config.count_min_width, config.count_min_depth),
            active_nodes: BloomFilter::new(config.bloom_expected_elements, config.bloom_false_positive_rate),
            recent_edges: BloomFilter::new(config.bloom_expected_elements * 2, config.bloom_false_positive_rate),
            config,
        }
    }

    /// Initialize with graph data
    pub fn initialize(&mut self, processor: &IncrementalGraphProcessor) -> Result<()> {
        let graph = processor.graph();
        let nodes_batch = &graph.nodes;
        let edges_batch = &graph.edges;

        // Process nodes
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;

            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i);
                self.unique_nodes.add(&node_id);
                self.active_nodes.add(&node_id);
            }
        }

        // Process edges
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
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                let weight = weights.value(i);

                let edge = (source, target);
                self.unique_edges.add(&edge);
                self.recent_edges.add(&edge);

                // Update degree counts
                self.node_degrees.add(&source, 1);
                self.node_degrees.add(&target, 1);

                // Update edge weight
                let weight_key = format!("{}→{}", source, target);
                self.edge_weights.add(&weight_key, (weight * 1000.0) as u32); // Scale for integer storage
            }
        }

        Ok(())
    }

    /// Update with new graph changes
    pub fn update(&mut self, processor: &IncrementalGraphProcessor) -> Result<ApproximateMetrics> {
        // For simplicity, reinitialize on updates
        // A full implementation would incrementally update each structure
        self.initialize(processor)?;

        Ok(self.get_metrics())
    }

    /// Get current approximate metrics
    pub fn get_metrics(&self) -> ApproximateMetrics {
        ApproximateMetrics {
            estimated_unique_nodes: self.unique_nodes.estimate(),
            estimated_unique_edges: self.unique_edges.estimate(),
            estimated_unique_triangles: self.unique_triangles.estimate(),
            active_nodes_bloom_fill_ratio: self.active_nodes.false_positive_probability(),
            recent_edges_bloom_fill_ratio: self.recent_edges.false_positive_probability(),
        }
    }

    /// Check if a node is likely active
    pub fn is_node_active(&self, node_id: &str) -> bool {
        self.active_nodes.contains(&node_id)
    }

    /// Check if an edge was recently added
    pub fn is_edge_recent(&self, source: &str, target: &str) -> bool {
        let edge = (source, target);
        self.recent_edges.contains(&edge)
    }

    /// Estimate node degree
    pub fn estimate_node_degree(&self, node_id: &str) -> u32 {
        self.node_degrees.estimate(&node_id)
    }

    /// Estimate edge weight
    pub fn estimate_edge_weight(&self, source: &str, target: &str) -> f64 {
        let weight_key = format!("{}→{}", source, target);
        self.edge_weights.estimate(&weight_key) as f64 / 1000.0 // Unscale
    }

    /// Merge with another approximate processor
    pub fn merge(&mut self, other: &ApproximateGraphProcessor) -> Result<()> {
        self.unique_nodes.merge(&other.unique_nodes)?;
        self.unique_edges.merge(&other.unique_edges)?;
        self.unique_triangles.merge(&other.unique_triangles)?;
        self.node_degrees.merge(&other.node_degrees)?;
        self.edge_weights.merge(&other.edge_weights)?;
        self.active_nodes.union(&other.active_nodes)?;
        self.recent_edges.union(&other.recent_edges)?;
        Ok(())
    }

    /// Add triangle for approximate triangle counting
    pub fn add_triangle(&mut self, node_a: &str, node_b: &str, node_c: &str) {
        // Create canonical triangle representation
        let mut nodes = vec![node_a, node_b, node_c];
        nodes.sort();
        let triangle = format!("{}→{}→{}", nodes[0], nodes[1], nodes[2]);
        self.unique_triangles.add(&triangle);
    }

    /// Reset all approximate structures
    pub fn reset(&mut self) {
        self.unique_nodes = HyperLogLog::new(self.config.hyperloglog_precision);
        self.unique_edges = HyperLogLog::new(self.config.hyperloglog_precision);
        self.unique_triangles = HyperLogLog::new(self.config.hyperloglog_precision);
        self.node_degrees = CountMinSketch::new(self.config.count_min_width, self.config.count_min_depth);
        self.edge_weights = CountMinSketch::new(self.config.count_min_width, self.config.count_min_depth);
        self.active_nodes = BloomFilter::new(self.config.bloom_expected_elements, self.config.bloom_false_positive_rate);
        self.recent_edges = BloomFilter::new(self.config.bloom_expected_elements * 2, self.config.bloom_false_positive_rate);
    }
}

/// Approximate metrics computed by the processor
#[derive(Debug, Clone)]
pub struct ApproximateMetrics {
    pub estimated_unique_nodes: f64,
    pub estimated_unique_edges: f64,
    pub estimated_unique_triangles: f64,
    pub active_nodes_bloom_fill_ratio: f64,
    pub recent_edges_bloom_fill_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ArrowGraph;
    use arrow::array::{StringArray, Float64Array};
    use arrow::record_batch::RecordBatch;
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    #[test]
    fn test_hyperloglog_basic() {
        let mut hll = HyperLogLog::new(12);
        
        // Add some elements
        for i in 0..1000 {
            hll.add(&format!("element_{}", i));
        }
        
        let estimate = hll.estimate();
        
        // Should be close to 1000 (within ~1.6% error for precision 12)
        assert!(estimate > 950.0 && estimate < 1050.0);
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog::new(8);
        let mut hll2 = HyperLogLog::new(8);
        
        // Add different elements to each
        for i in 0..500 {
            hll1.add(&format!("a_{}", i));
            hll2.add(&format!("b_{}", i));
        }
        
        let estimate1 = hll1.estimate();
        let estimate2 = hll2.estimate();
        
        hll1.merge(&hll2).unwrap();
        let merged_estimate = hll1.estimate();
        
        // Merged estimate should be roughly the sum
        assert!(merged_estimate > estimate1 + estimate2 - 100.0);
    }

    #[test]
    fn test_count_min_sketch() {
        let mut cms = CountMinSketch::new(100, 5);
        
        // Add some elements with counts
        cms.add(&"element1", 10);
        cms.add(&"element2", 5);
        cms.add(&"element1", 3); // Add more to element1
        
        // Estimates should be at least the actual counts
        assert!(cms.estimate(&"element1") >= 13);
        assert!(cms.estimate(&"element2") >= 5);
        assert_eq!(cms.estimate(&"nonexistent"), 0);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 0.01);
        
        // Add some elements
        bloom.add(&"element1");
        bloom.add(&"element2");
        bloom.add(&"element3");
        
        // Should contain added elements
        assert!(bloom.contains(&"element1"));
        assert!(bloom.contains(&"element2"));
        assert!(bloom.contains(&"element3"));
        
        // Should not contain non-added elements (with high probability)
        assert!(!bloom.contains(&"nonexistent"));
    }

    #[test]
    fn test_bloom_filter_union() {
        let mut bloom1 = BloomFilter::new(1000, 0.01);
        let mut bloom2 = BloomFilter::new(1000, 0.01);
        
        bloom1.add(&"element1");
        bloom2.add(&"element2");
        
        bloom1.union(&bloom2).unwrap();
        
        // Should contain elements from both filters
        assert!(bloom1.contains(&"element1"));
        assert!(bloom1.contains(&"element2"));
    }

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
        let weights = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_approximate_graph_processor() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut approx = ApproximateGraphProcessor::new(ApproximateConfig::default());
        approx.initialize(&processor).unwrap();
        
        let metrics = approx.get_metrics();
        
        // Should estimate roughly 4 nodes and 3 edges
        assert!(metrics.estimated_unique_nodes > 3.0 && metrics.estimated_unique_nodes < 5.0);
        assert!(metrics.estimated_unique_edges > 2.0 && metrics.estimated_unique_edges < 4.0);
    }

    #[test]
    fn test_node_and_edge_queries() {
        let graph = create_test_graph().unwrap();
        let processor = IncrementalGraphProcessor::new(graph).unwrap();
        
        let mut approx = ApproximateGraphProcessor::new(ApproximateConfig::default());
        approx.initialize(&processor).unwrap();
        
        // Should detect active nodes
        assert!(approx.is_node_active("A"));
        assert!(approx.is_node_active("B"));
        assert!(!approx.is_node_active("nonexistent")); // High probability
        
        // Should detect recent edges
        assert!(approx.is_edge_recent("A", "B"));
        assert!(approx.is_edge_recent("B", "C"));
        
        // Should estimate node degrees
        assert!(approx.estimate_node_degree("A") > 0);
        assert!(approx.estimate_node_degree("B") > 0);
    }

    #[test]
    fn test_triangle_counting() {
        let mut approx = ApproximateGraphProcessor::new(ApproximateConfig::default());
        
        // Add some triangles
        approx.add_triangle("A", "B", "C");
        approx.add_triangle("B", "C", "D");
        approx.add_triangle("A", "B", "C"); // Duplicate
        
        let metrics = approx.get_metrics();
        
        // Should estimate around 2 unique triangles
        assert!(metrics.estimated_unique_triangles > 1.5 && metrics.estimated_unique_triangles < 2.5);
    }

    #[test]
    fn test_processor_merge() {
        let config = ApproximateConfig::default();
        
        let mut approx1 = ApproximateGraphProcessor::new(config.clone());
        let mut approx2 = ApproximateGraphProcessor::new(config);
        
        // Add different elements to each
        approx1.unique_nodes.add(&"A");
        approx1.unique_nodes.add(&"B");
        
        approx2.unique_nodes.add(&"C");
        approx2.unique_nodes.add(&"D");
        
        let estimate1 = approx1.get_metrics().estimated_unique_nodes;
        let estimate2 = approx2.get_metrics().estimated_unique_nodes;
        
        approx1.merge(&approx2).unwrap();
        let merged_estimate = approx1.get_metrics().estimated_unique_nodes;
        
        // Merged estimate should be higher than individual estimates
        assert!(merged_estimate > estimate1);
        assert!(merged_estimate > estimate2);
    }
}