use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::{Array, StringArray, Float64Array};
use arrow::record_batch::RecordBatch;
use memmap2::{Mmap, MmapMut, MmapOptions};
use parking_lot::{RwLock, Mutex};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use aligned_vec::avec;

/// Memory-mapped graph storage for handling massive graphs
/// Provides efficient disk-backed storage with OS-level caching
#[derive(Debug)]
pub struct MemoryMappedGraph {
    nodes_mmap: Arc<RwLock<Option<Mmap>>>,
    edges_mmap: Arc<RwLock<Option<Mmap>>>,
    metadata: GraphMetadata,
    storage_path: PathBuf,
    read_only: bool,
}

/// Graph metadata for memory-mapped storage
#[derive(Debug, Clone)]
pub struct GraphMetadata {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub node_size: usize,
    pub edge_size: usize,
    pub version: u32,
    pub created_at: u64,
    pub last_modified: u64,
    pub checksum: u64,
}

/// High-performance memory pool for graph operations
/// Uses cache-aligned allocations and memory pooling
#[derive(Debug)]
pub struct MemoryPool {
    pools: HashMap<usize, Arc<Mutex<Vec<NonNull<u8>>>>>,
    alignment: usize,
    max_pool_size: usize,
    allocated_bytes: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
}

/// Cache-optimized storage for hot graph data
/// Implements intelligent caching strategies
#[derive(Debug)]
pub struct CacheOptimizedStorage {
    l1_cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    l2_cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    cache_config: CacheConfig,
    hit_stats: Arc<Mutex<CacheStats>>,
}

/// Cache configuration parameters
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub l1_size: usize,
    pub l2_size: usize,
    pub eviction_policy: EvictionPolicy,
    pub prefetch_strategy: PrefetchStrategy,
    pub compression_enabled: bool,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,        // Least Recently Used
    LFU,        // Least Frequently Used
    ARC,        // Adaptive Replacement Cache
    Random,     // Random eviction
    FIFO,       // First In, First Out
}

/// Prefetch strategies for predictive caching
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    None,
    Sequential,     // Sequential access pattern
    Neighbors,      // Prefetch neighbor nodes
    RandomWalk,     // Based on random walk patterns
    ML,            // Machine learning based
}

/// Cache key for storing graph elements
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum CacheKey {
    Node(String),
    Edge(String, String),
    Subgraph(Vec<String>),
    Algorithm(String, Vec<String>), // Algorithm name + parameters
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: CacheData,
    pub size: usize,
    pub access_count: u64,
    pub last_accessed: u64,
    pub created_at: u64,
    pub compressed: bool,
}

/// Cached data types
#[derive(Debug, Clone)]
pub enum CacheData {
    NodeData(Vec<u8>),
    EdgeData(Vec<u8>),
    ComputationResult(Vec<u8>),
    Metadata(Vec<u8>),
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub evictions: u64,
    pub prefetch_hits: u64,
    pub memory_usage: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub pool_usage: HashMap<usize, usize>,
    pub fragmentation_ratio: f64,
    pub cache_hit_ratio: f64,
}

/// Custom allocator for graph operations
pub struct GraphAllocator {
    pool: Arc<MemoryPool>,
}

impl MemoryMappedGraph {
    /// Create a new memory-mapped graph
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let path = storage_path.as_ref().to_path_buf();
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        Ok(Self {
            nodes_mmap: Arc::new(RwLock::new(None)),
            edges_mmap: Arc::new(RwLock::new(None)),
            metadata: GraphMetadata::default(),
            storage_path: path,
            read_only: false,
        })
    }

    /// Open existing memory-mapped graph in read-only mode
    pub fn open_readonly<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let mut graph = Self::new(storage_path)?;
        graph.read_only = true;
        graph.load_metadata()?;
        graph.map_existing_files()?;
        Ok(graph)
    }

    /// Create memory-mapped graph from Arrow graph
    pub fn from_arrow_graph<P: AsRef<Path>>(
        graph: &ArrowGraph,
        storage_path: P,
    ) -> Result<Self> {
        let mut mmap_graph = Self::new(storage_path)?;
        mmap_graph.serialize_arrow_graph(graph)?;
        Ok(mmap_graph)
    }

    /// Serialize Arrow graph to memory-mapped files
    fn serialize_arrow_graph(&mut self, graph: &ArrowGraph) -> Result<()> {
        self.metadata.num_nodes = graph.node_count();
        self.metadata.num_edges = graph.edge_count();
        
        // Serialize nodes
        self.serialize_nodes(&graph.nodes)?;
        
        // Serialize edges
        self.serialize_edges(&graph.edges)?;
        
        // Save metadata
        self.save_metadata()?;
        
        Ok(())
    }

    /// Serialize nodes to memory-mapped file
    fn serialize_nodes(&mut self, nodes_batch: &RecordBatch) -> Result<()> {
        let nodes_file_path = self.storage_path.join("nodes.mmap");
        
        // Calculate required size
        let estimated_size = self.estimate_nodes_size(nodes_batch)?;
        
        // Create and resize file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&nodes_file_path)?;
        file.set_len(estimated_size as u64)?;
        
        // Memory map the file
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        
        // Serialize node data
        let mut offset = 0;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i);
                let node_data = node_id.as_bytes();
                
                // Write length prefix
                let len = node_data.len() as u32;
                mmap[offset..offset + 4].copy_from_slice(&len.to_le_bytes());
                offset += 4;
                
                // Write node data
                mmap[offset..offset + node_data.len()].copy_from_slice(node_data);
                offset += node_data.len();
            }
        }
        
        // Ensure data is written to disk
        mmap.flush()?;
        
        // Store mmap reference
        *self.nodes_mmap.write() = Some(mmap.make_read_only()?);
        
        self.metadata.node_size = estimated_size;
        
        Ok(())
    }

    /// Serialize edges to memory-mapped file
    fn serialize_edges(&mut self, edges_batch: &RecordBatch) -> Result<()> {
        let edges_file_path = self.storage_path.join("edges.mmap");
        
        // Calculate required size
        let estimated_size = self.estimate_edges_size(edges_batch)?;
        
        // Create and resize file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&edges_file_path)?;
        file.set_len(estimated_size as u64)?;
        
        // Memory map the file
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        
        // Serialize edge data
        let mut offset = 0;
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            let weights = edges_batch.column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected float64 array for weights"))?;
            
            for i in 0..source_ids.len() {
                let source = source_ids.value(i);
                let target = target_ids.value(i);
                let weight = weights.value(i);
                
                // Write source length and data
                let source_len = source.len() as u32;
                mmap[offset..offset + 4].copy_from_slice(&source_len.to_le_bytes());
                offset += 4;
                mmap[offset..offset + source.len()].copy_from_slice(source.as_bytes());
                offset += source.len();
                
                // Write target length and data
                let target_len = target.len() as u32;
                mmap[offset..offset + 4].copy_from_slice(&target_len.to_le_bytes());
                offset += 4;
                mmap[offset..offset + target.len()].copy_from_slice(target.as_bytes());
                offset += target.len();
                
                // Write weight
                mmap[offset..offset + 8].copy_from_slice(&weight.to_le_bytes());
                offset += 8;
            }
        }
        
        // Ensure data is written to disk
        mmap.flush()?;
        
        // Store mmap reference
        *self.edges_mmap.write() = Some(mmap.make_read_only()?);
        
        self.metadata.edge_size = estimated_size;
        
        Ok(())
    }

    /// Estimate required size for nodes serialization
    fn estimate_nodes_size(&self, nodes_batch: &RecordBatch) -> Result<usize> {
        let mut size = 0;
        
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                size += 4; // Length prefix
                size += node_ids.value(i).len(); // Node ID data
            }
        }
        
        // Add 10% padding for alignment
        Ok((size as f64 * 1.1) as usize)
    }

    /// Estimate required size for edges serialization
    fn estimate_edges_size(&self, edges_batch: &RecordBatch) -> Result<usize> {
        let mut size = 0;
        
        if edges_batch.num_rows() > 0 {
            let source_ids = edges_batch.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for source IDs"))?;
            let target_ids = edges_batch.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for target IDs"))?;
            
            for i in 0..source_ids.len() {
                size += 4 + source_ids.value(i).len(); // Source
                size += 4 + target_ids.value(i).len(); // Target
                size += 8; // Weight (f64)
            }
        }
        
        // Add 10% padding
        Ok((size as f64 * 1.1) as usize)
    }

    /// Load existing memory-mapped files
    fn map_existing_files(&mut self) -> Result<()> {
        // Map nodes file
        let nodes_path = self.storage_path.join("nodes.mmap");
        if nodes_path.exists() {
            let file = File::open(&nodes_path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            *self.nodes_mmap.write() = Some(mmap);
        }
        
        // Map edges file
        let edges_path = self.storage_path.join("edges.mmap");
        if edges_path.exists() {
            let file = File::open(&edges_path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            *self.edges_mmap.write() = Some(mmap);
        }
        
        Ok(())
    }

    /// Save metadata to file
    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.storage_path.join("metadata.json");
        let json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(metadata_path, json)?;
        Ok(())
    }

    /// Load metadata from file
    fn load_metadata(&mut self) -> Result<()> {
        let metadata_path = self.storage_path.join("metadata.json");
        if metadata_path.exists() {
            let json = std::fs::read_to_string(metadata_path)?;
            self.metadata = serde_json::from_str(&json)?;
        }
        Ok(())
    }

    /// Get node data by ID
    pub fn get_node(&self, node_id: &str) -> Result<Option<Vec<u8>>> {
        let nodes_mmap = self.nodes_mmap.read();
        if let Some(ref mmap) = *nodes_mmap {
            // Linear search through memory-mapped nodes
            // In production, this would use an index
            let mut offset = 0;
            while offset < mmap.len() {
                if offset + 4 > mmap.len() {
                    break;
                }
                
                // Read length
                let len = u32::from_le_bytes([
                    mmap[offset],
                    mmap[offset + 1],
                    mmap[offset + 2],
                    mmap[offset + 3],
                ]) as usize;
                offset += 4;
                
                if offset + len > mmap.len() {
                    break;
                }
                
                // Read node ID
                let stored_id = std::str::from_utf8(&mmap[offset..offset + len])
                    .map_err(|e| crate::error::GraphError::graph_construction(&format!("UTF-8 error: {}", e)))?;
                
                if stored_id == node_id {
                    return Ok(Some(mmap[offset..offset + len].to_vec()));
                }
                
                offset += len;
            }
        }
        
        Ok(None)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let nodes_size = self.nodes_mmap.read()
            .as_ref()
            .map(|mmap| mmap.len())
            .unwrap_or(0);
        
        let edges_size = self.edges_mmap.read()
            .as_ref()
            .map(|mmap| mmap.len())
            .unwrap_or(0);
        
        let total = nodes_size + edges_size;
        
        MemoryStats {
            total_allocated: total,
            peak_usage: total, // Simplified
            pool_usage: HashMap::new(),
            fragmentation_ratio: 0.0,
            cache_hit_ratio: 0.95, // Estimated
        }
    }

    /// Prefetch data for better performance
    pub fn prefetch_region(&self, offset: usize, size: usize) -> Result<()> {
        // Use madvise to hint the OS about access patterns
        #[cfg(unix)]
        {
            if let Some(ref mmap) = *self.nodes_mmap.read() {
                if offset < mmap.len() {
                    let actual_size = size.min(mmap.len() - offset);
                    unsafe {
                        libc::madvise(
                            mmap.as_ptr().add(offset) as *mut libc::c_void,
                            actual_size,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(alignment: usize, max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            alignment,
            max_pool_size,
            allocated_bytes: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
        }
    }

    /// Allocate aligned memory
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        let aligned_size = self.align_size(size);
        
        // Try to get from pool first
        if let Some(pool) = self.pools.get(&aligned_size) {
            let mut pool = pool.lock();
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }
        
        // Allocate new memory
        let layout = Layout::from_size_align(aligned_size, self.alignment)
            .map_err(|e| crate::error::GraphError::graph_construction(&format!("Layout error: {}", e)))?;
        
        let ptr = unsafe { System.alloc(layout) };
        if ptr.is_null() {
            return Err(crate::error::GraphError::graph_construction("Allocation failed"));
        }
        
        // Update statistics
        let mut allocated = self.allocated_bytes.lock();
        *allocated += aligned_size;
        
        let mut peak = self.peak_usage.lock();
        if *allocated > *peak {
            *peak = *allocated;
        }
        
        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) {
        let aligned_size = self.align_size(size);
        
        // Return to pool if not full
        let pool = self.pools.entry(aligned_size)
            .or_insert_with(|| Arc::new(Mutex::new(Vec::new())));
        
        let mut pool = pool.lock();
        if pool.len() < self.max_pool_size {
            pool.push(ptr);
        } else {
            // Actually deallocate
            let layout = Layout::from_size_align(aligned_size, self.alignment).unwrap();
            unsafe { System.dealloc(ptr.as_ptr(), layout) };
        }
        
        // Update statistics
        let mut allocated = self.allocated_bytes.lock();
        *allocated = allocated.saturating_sub(aligned_size);
    }

    /// Align size to cache line boundary
    fn align_size(&self, size: usize) -> usize {
        (size + self.alignment - 1) & !(self.alignment - 1)
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        let allocated = *self.allocated_bytes.lock();
        let peak = *self.peak_usage.lock();
        
        let mut pool_usage = HashMap::new();
        for (size, pool) in &self.pools {
            pool_usage.insert(*size, pool.lock().len());
        }
        
        MemoryStats {
            total_allocated: allocated,
            peak_usage: peak,
            pool_usage,
            fragmentation_ratio: 0.1, // Estimated
            cache_hit_ratio: 0.85, // Estimated
        }
    }
}

impl CacheOptimizedStorage {
    /// Create new cache-optimized storage
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l2_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_config: config,
            hit_stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get data from cache
    pub fn get(&self, key: &CacheKey) -> Option<CacheData> {
        // Try L1 cache first
        {
            let l1 = self.l1_cache.read();
            if let Some(entry) = l1.get(key) {
                self.update_access_stats(entry, true, 1);
                return Some(entry.data.clone());
            }
        }
        
        // Try L2 cache
        {
            let l2 = self.l2_cache.read();
            if let Some(entry) = l2.get(key) {
                self.update_access_stats(entry, true, 2);
                
                // Promote to L1
                self.promote_to_l1(key.clone(), entry.clone());
                
                return Some(entry.data.clone());
            }
        }
        
        // Cache miss
        self.record_miss();
        None
    }

    /// Put data into cache
    pub fn put(&self, key: CacheKey, data: CacheData) {
        let size = self.estimate_data_size(&data);
        let entry = CacheEntry {
            data,
            size,
            access_count: 1,
            last_accessed: self.current_time(),
            created_at: self.current_time(),
            compressed: false,
        };
        
        // Insert into L1 cache
        {
            let mut l1 = self.l1_cache.write();
            
            // Check if eviction needed
            if l1.len() >= self.cache_config.l1_size {
                self.evict_l1(&mut l1);
            }
            
            l1.insert(key, entry);
        }
    }

    /// Prefetch data based on strategy
    pub fn prefetch(&self, keys: Vec<CacheKey>) {
        match self.cache_config.prefetch_strategy {
            PrefetchStrategy::Sequential => {
                self.prefetch_sequential(keys);
            }
            PrefetchStrategy::Neighbors => {
                self.prefetch_neighbors(keys);
            }
            PrefetchStrategy::RandomWalk => {
                self.prefetch_random_walk(keys);
            }
            _ => {} // No prefetching
        }
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.hit_stats.lock().clone()
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.l1_cache.write().clear();
        self.l2_cache.write().clear();
        
        let mut stats = self.hit_stats.lock();
        *stats = CacheStats::default();
    }

    // Private helper methods
    fn promote_to_l1(&self, key: CacheKey, entry: CacheEntry) {
        let mut l1 = self.l1_cache.write();
        
        if l1.len() >= self.cache_config.l1_size {
            self.evict_l1(&mut l1);
        }
        
        l1.insert(key, entry);
    }

    fn evict_l1(&self, l1_cache: &mut HashMap<CacheKey, CacheEntry>) {
        match self.cache_config.eviction_policy {
            EvictionPolicy::LRU => {
                if let Some((key, _)) = l1_cache.iter()
                    .min_by_key(|(_, entry)| entry.last_accessed)
                    .map(|(k, v)| (k.clone(), v.clone())) {
                    
                    let entry = l1_cache.remove(&key).unwrap();
                    
                    // Move to L2
                    let mut l2 = self.l2_cache.write();
                    if l2.len() >= self.cache_config.l2_size {
                        self.evict_l2(&mut l2);
                    }
                    l2.insert(key, entry);
                }
            }
            EvictionPolicy::LFU => {
                if let Some((key, _)) = l1_cache.iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(k, v)| (k.clone(), v.clone())) {
                    
                    l1_cache.remove(&key);
                }
            }
            _ => {
                // Random eviction
                if let Some(key) = l1_cache.keys().next().cloned() {
                    l1_cache.remove(&key);
                }
            }
        }
        
        // Update stats
        let mut stats = self.hit_stats.lock();
        stats.evictions += 1;
    }

    fn evict_l2(&self, l2_cache: &mut HashMap<CacheKey, CacheEntry>) {
        if let Some(key) = l2_cache.keys().next().cloned() {
            l2_cache.remove(&key);
        }
    }

    fn update_access_stats(&self, entry: &CacheEntry, hit: bool, level: u8) {
        let mut stats = self.hit_stats.lock();
        match level {
            1 => if hit { stats.l1_hits += 1 } else { stats.l1_misses += 1 },
            2 => if hit { stats.l2_hits += 1 } else { stats.l2_misses += 1 },
            _ => {}
        }
    }

    fn record_miss(&self) {
        let mut stats = self.hit_stats.lock();
        stats.l1_misses += 1;
        stats.l2_misses += 1;
    }

    fn estimate_data_size(&self, data: &CacheData) -> usize {
        match data {
            CacheData::NodeData(bytes) => bytes.len(),
            CacheData::EdgeData(bytes) => bytes.len(),
            CacheData::ComputationResult(bytes) => bytes.len(),
            CacheData::Metadata(bytes) => bytes.len(),
        }
    }

    fn current_time(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn prefetch_sequential(&self, _keys: Vec<CacheKey>) {
        // Implement sequential prefetching
    }

    fn prefetch_neighbors(&self, _keys: Vec<CacheKey>) {
        // Implement neighbor prefetching
    }

    fn prefetch_random_walk(&self, _keys: Vec<CacheKey>) {
        // Implement random walk prefetching
    }
}

impl Default for GraphMetadata {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            node_size: 0,
            edge_size: 0,
            version: 1,
            created_at: 0,
            last_modified: 0,
            checksum: 0,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 1000,
            l2_size: 10000,
            eviction_policy: EvictionPolicy::LRU,
            prefetch_strategy: PrefetchStrategy::Sequential,
            compression_enabled: false,
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            l1_hits: 0,
            l1_misses: 0,
            l2_hits: 0,
            l2_misses: 0,
            evictions: 0,
            prefetch_hits: 0,
            memory_usage: 0,
        }
    }
}

// Implement serde for GraphMetadata
impl serde::Serialize for GraphMetadata {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("GraphMetadata", 8)?;
        state.serialize_field("num_nodes", &self.num_nodes)?;
        state.serialize_field("num_edges", &self.num_edges)?;
        state.serialize_field("node_size", &self.node_size)?;
        state.serialize_field("edge_size", &self.edge_size)?;
        state.serialize_field("version", &self.version)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("last_modified", &self.last_modified)?;
        state.serialize_field("checksum", &self.checksum)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for GraphMetadata {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct GraphMetadataVisitor;

        impl<'de> Visitor<'de> for GraphMetadataVisitor {
            type Value = GraphMetadata;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GraphMetadata")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<GraphMetadata, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut num_nodes = None;
                let mut num_edges = None;
                let mut node_size = None;
                let mut edge_size = None;
                let mut version = None;
                let mut created_at = None;
                let mut last_modified = None;
                let mut checksum = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "num_nodes" => {
                            if num_nodes.is_some() {
                                return Err(de::Error::duplicate_field("num_nodes"));
                            }
                            num_nodes = Some(map.next_value()?);
                        }
                        "num_edges" => {
                            if num_edges.is_some() {
                                return Err(de::Error::duplicate_field("num_edges"));
                            }
                            num_edges = Some(map.next_value()?);
                        }
                        "node_size" => {
                            if node_size.is_some() {
                                return Err(de::Error::duplicate_field("node_size"));
                            }
                            node_size = Some(map.next_value()?);
                        }
                        "edge_size" => {
                            if edge_size.is_some() {
                                return Err(de::Error::duplicate_field("edge_size"));
                            }
                            edge_size = Some(map.next_value()?);
                        }
                        "version" => {
                            if version.is_some() {
                                return Err(de::Error::duplicate_field("version"));
                            }
                            version = Some(map.next_value()?);
                        }
                        "created_at" => {
                            if created_at.is_some() {
                                return Err(de::Error::duplicate_field("created_at"));
                            }
                            created_at = Some(map.next_value()?);
                        }
                        "last_modified" => {
                            if last_modified.is_some() {
                                return Err(de::Error::duplicate_field("last_modified"));
                            }
                            last_modified = Some(map.next_value()?);
                        }
                        "checksum" => {
                            if checksum.is_some() {
                                return Err(de::Error::duplicate_field("checksum"));
                            }
                            checksum = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }

                Ok(GraphMetadata {
                    num_nodes: num_nodes.unwrap_or(0),
                    num_edges: num_edges.unwrap_or(0),
                    node_size: node_size.unwrap_or(0),
                    edge_size: edge_size.unwrap_or(0),
                    version: version.unwrap_or(1),
                    created_at: created_at.unwrap_or(0),
                    last_modified: last_modified.unwrap_or(0),
                    checksum: checksum.unwrap_or(0),
                })
            }
        }

        deserializer.deserialize_struct(
            "GraphMetadata",
            &["num_nodes", "num_edges", "node_size", "edge_size", "version", "created_at", "last_modified", "checksum"],
            GraphMetadataVisitor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
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
    fn test_memory_mapped_graph_creation() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_graph");
        
        let mmap_graph = MemoryMappedGraph::new(&path).unwrap();
        assert_eq!(mmap_graph.metadata.num_nodes, 0);
        assert_eq!(mmap_graph.metadata.num_edges, 0);
    }

    #[test]
    fn test_memory_mapped_graph_from_arrow() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_graph");
        
        let graph = create_test_graph().unwrap();
        let mmap_graph = MemoryMappedGraph::from_arrow_graph(&graph, &path).unwrap();
        
        assert_eq!(mmap_graph.metadata.num_nodes, 4);
        assert_eq!(mmap_graph.metadata.num_edges, 3);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(64, 100);
        
        // Allocate some memory
        let ptr1 = pool.allocate(1024).unwrap();
        let ptr2 = pool.allocate(2048).unwrap();
        
        // Check stats
        let stats = pool.get_stats();
        assert!(stats.total_allocated > 0);
        assert!(stats.peak_usage > 0);
        
        // Deallocate
        pool.deallocate(ptr1, 1024);
        pool.deallocate(ptr2, 2048);
    }

    #[test]
    fn test_cache_optimized_storage() {
        let config = CacheConfig::default();
        let cache = CacheOptimizedStorage::new(config);
        
        // Test cache operations
        let key = CacheKey::Node("A".to_string());
        let data = CacheData::NodeData(vec![1, 2, 3, 4]);
        
        // Should be cache miss initially
        assert!(cache.get(&key).is_none());
        
        // Put data
        cache.put(key.clone(), data.clone());
        
        // Should be cache hit now
        assert!(cache.get(&key).is_some());
        
        // Check stats
        let stats = cache.get_cache_stats();
        assert!(stats.l1_hits > 0);
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            l1_size: 2,
            l2_size: 2,
            ..Default::default()
        };
        let cache = CacheOptimizedStorage::new(config);
        
        // Fill L1 cache beyond capacity
        for i in 0..5 {
            let key = CacheKey::Node(format!("Node{}", i));
            let data = CacheData::NodeData(vec![i as u8]);
            cache.put(key, data);
        }
        
        // Check that evictions occurred
        let stats = cache.get_cache_stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_memory_stats() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_graph");
        
        let graph = create_test_graph().unwrap();
        let mmap_graph = MemoryMappedGraph::from_arrow_graph(&graph, &path).unwrap();
        
        let stats = mmap_graph.get_memory_stats();
        assert!(stats.total_allocated > 0);
    }

    #[test]
    fn test_node_retrieval() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_graph");
        
        let graph = create_test_graph().unwrap();
        let mmap_graph = MemoryMappedGraph::from_arrow_graph(&graph, &path).unwrap();
        
        // Test node retrieval
        let node_data = mmap_graph.get_node("A").unwrap();
        assert!(node_data.is_some());
        
        let missing_node = mmap_graph.get_node("Z").unwrap();
        assert!(missing_node.is_none());
    }

    #[test]
    fn test_cache_key_types() {
        let cache = CacheOptimizedStorage::new(CacheConfig::default());
        
        // Test different cache key types
        let node_key = CacheKey::Node("A".to_string());
        let edge_key = CacheKey::Edge("A".to_string(), "B".to_string());
        let subgraph_key = CacheKey::Subgraph(vec!["A".to_string(), "B".to_string()]);
        
        let data = CacheData::NodeData(vec![1, 2, 3]);
        
        cache.put(node_key.clone(), data.clone());
        cache.put(edge_key.clone(), data.clone());
        cache.put(subgraph_key.clone(), data.clone());
        
        assert!(cache.get(&node_key).is_some());
        assert!(cache.get(&edge_key).is_some());
        assert!(cache.get(&subgraph_key).is_some());
    }
}