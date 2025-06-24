use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::collections::HashMap;
use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
use crate::graph::ArrowGraph;
use crate::error::{GraphError, Result};

/// Union-Find (Disjoint Set) data structure for efficient connected components
#[derive(Debug)]
struct UnionFind {
    parent: HashMap<String, String>,
    rank: HashMap<String, usize>,
    component_sizes: HashMap<String, usize>,
}

impl UnionFind {
    fn new() -> Self {
        UnionFind {
            parent: HashMap::new(),
            rank: HashMap::new(),
            component_sizes: HashMap::new(),
        }
    }
    
    fn make_set(&mut self, node: String) {
        if !self.parent.contains_key(&node) {
            self.parent.insert(node.clone(), node.clone());
            self.rank.insert(node.clone(), 0);
            self.component_sizes.insert(node.clone(), 1);
        }
    }
    
    fn find(&mut self, node: &str) -> Option<String> {
        if !self.parent.contains_key(node) {
            return None;
        }
        
        // Path compression
        let parent = self.parent.get(node).unwrap().clone();
        if parent != node {
            let root = self.find(&parent)?;
            self.parent.insert(node.to_string(), root.clone());
            Some(root)
        } else {
            Some(parent)
        }
    }
    
    fn union(&mut self, node1: &str, node2: &str) -> bool {
        let root1 = match self.find(node1) {
            Some(r) => r,
            None => return false,
        };
        
        let root2 = match self.find(node2) {
            Some(r) => r,
            None => return false,
        };
        
        if root1 == root2 {
            return false; // Already in same component
        }
        
        // Union by rank
        let rank1 = *self.rank.get(&root1).unwrap_or(&0);
        let rank2 = *self.rank.get(&root2).unwrap_or(&0);
        
        let (new_root, old_root) = if rank1 > rank2 {
            (root1, root2)
        } else if rank1 < rank2 {
            (root2, root1)
        } else {
            // Equal ranks, choose root1 and increment its rank
            self.rank.insert(root1.clone(), rank1 + 1);
            (root1, root2)
        };
        
        // Update parent
        self.parent.insert(old_root.clone(), new_root.clone());
        
        // Update component sizes
        let size1 = *self.component_sizes.get(&new_root).unwrap_or(&0);
        let size2 = *self.component_sizes.get(&old_root).unwrap_or(&0);
        self.component_sizes.insert(new_root, size1 + size2);
        self.component_sizes.remove(&old_root);
        
        true
    }
    
    fn get_components(&mut self) -> HashMap<String, Vec<String>> {
        let mut components: HashMap<String, Vec<String>> = HashMap::new();
        
        // Get all nodes and their root components
        let nodes: Vec<String> = self.parent.keys().cloned().collect();
        for node in nodes {
            if let Some(root) = self.find(&node) {
                components.entry(root).or_default().push(node);
            }
        }
        
        components
    }
    
    fn component_count(&mut self) -> usize {
        self.get_components().len()
    }
}

pub struct WeaklyConnectedComponents;

impl WeaklyConnectedComponents {
    /// Find weakly connected components using Union-Find
    fn compute_components(&self, graph: &ArrowGraph) -> Result<HashMap<String, u32>> {
        let mut uf = UnionFind::new();
        
        // Initialize all nodes
        for node_id in graph.node_ids() {
            uf.make_set(node_id.clone());
        }
        
        // Union nodes connected by edges (treat as undirected)
        for node_id in graph.node_ids() {
            if let Some(neighbors) = graph.neighbors(node_id) {
                for neighbor in neighbors {
                    uf.union(node_id, neighbor);
                }
            }
        }
        
        // Assign component IDs
        let components = uf.get_components();
        let mut node_to_component: HashMap<String, u32> = HashMap::new();
        
        for (component_id, (_root, nodes)) in components.into_iter().enumerate() {
            for node in nodes {
                node_to_component.insert(node, component_id as u32);
            }
        }
        
        Ok(node_to_component)
    }
}

impl GraphAlgorithm for WeaklyConnectedComponents {
    fn execute(&self, graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        let component_map = self.compute_components(graph)?;
        
        if component_map.is_empty() {
            // Return empty result with proper schema
            let schema = Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("component_id", DataType::UInt32, false),
            ]));
            
            return RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(Vec::<String>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                ],
            ).map_err(GraphError::from);
        }
        
        // Sort by component ID for consistent output
        let mut sorted_nodes: Vec<(&String, &u32)> = component_map.iter().collect();
        sorted_nodes.sort_by_key(|(_, &component_id)| component_id);
        
        let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
        let component_ids: Vec<u32> = sorted_nodes.iter().map(|(_, &comp)| comp).collect();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("component_id", DataType::UInt32, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(UInt32Array::from(component_ids)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "weakly_connected_components"
    }
    
    fn description(&self) -> &'static str {
        "Find weakly connected components using Union-Find with path compression"
    }
}

pub struct StronglyConnectedComponents;

impl StronglyConnectedComponents {
    /// Tarjan's algorithm for strongly connected components
    fn tarjan_scc(&self, graph: &ArrowGraph) -> Result<HashMap<String, u32>> {
        let mut index_counter = 0;
        let mut stack = Vec::new();
        let mut indices: HashMap<String, usize> = HashMap::new();
        let mut lowlinks: HashMap<String, usize> = HashMap::new();
        let mut on_stack: HashMap<String, bool> = HashMap::new();
        let mut components: Vec<Vec<String>> = Vec::new();
        
        // Initialize
        for node_id in graph.node_ids() {
            on_stack.insert(node_id.clone(), false);
        }
        
        // Run DFS from each unvisited node
        for node_id in graph.node_ids() {
            if !indices.contains_key(node_id) {
                self.tarjan_strongconnect(
                    node_id,
                    graph,
                    &mut index_counter,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut components,
                )?;
            }
        }
        
        // Create component mapping
        let mut node_to_component: HashMap<String, u32> = HashMap::new();
        for (comp_id, component) in components.into_iter().enumerate() {
            for node in component {
                node_to_component.insert(node, comp_id as u32);
            }
        }
        
        Ok(node_to_component)
    }
    
    fn tarjan_strongconnect(
        &self,
        node: &str,
        graph: &ArrowGraph,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        on_stack: &mut HashMap<String, bool>,
        components: &mut Vec<Vec<String>>,
    ) -> Result<()> {
        // Set the depth index for this node
        indices.insert(node.to_string(), *index_counter);
        lowlinks.insert(node.to_string(), *index_counter);
        *index_counter += 1;
        
        // Push node onto stack
        stack.push(node.to_string());
        on_stack.insert(node.to_string(), true);
        
        // Consider successors
        if let Some(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if !indices.contains_key(neighbor) {
                    // Successor has not yet been visited; recurse on it
                    self.tarjan_strongconnect(
                        neighbor,
                        graph,
                        index_counter,
                        stack,
                        indices,
                        lowlinks,
                        on_stack,
                        components,
                    )?;
                    
                    let neighbor_lowlink = *lowlinks.get(neighbor).unwrap_or(&0);
                    let current_lowlink = *lowlinks.get(node).unwrap_or(&0);
                    lowlinks.insert(node.to_string(), current_lowlink.min(neighbor_lowlink));
                } else if *on_stack.get(neighbor).unwrap_or(&false) {
                    // Successor is in stack and hence in the current SCC
                    let neighbor_index = *indices.get(neighbor).unwrap_or(&0);
                    let current_lowlink = *lowlinks.get(node).unwrap_or(&0);
                    lowlinks.insert(node.to_string(), current_lowlink.min(neighbor_index));
                }
            }
        }
        
        // If node is a root node, pop the stack and create an SCC
        let node_index = *indices.get(node).unwrap_or(&0);
        let node_lowlink = *lowlinks.get(node).unwrap_or(&0);
        
        if node_lowlink == node_index {
            let mut component = Vec::new();
            loop {
                if let Some(w) = stack.pop() {
                    on_stack.insert(w.clone(), false);
                    component.push(w.clone());
                    if w == node {
                        break;
                    }
                } else {
                    break;
                }
            }
            components.push(component);
        }
        
        Ok(())
    }
}

impl GraphAlgorithm for StronglyConnectedComponents {
    fn execute(&self, graph: &ArrowGraph, _params: &AlgorithmParams) -> Result<RecordBatch> {
        let component_map = self.tarjan_scc(graph)?;
        
        if component_map.is_empty() {
            // Return empty result with proper schema
            let schema = Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("component_id", DataType::UInt32, false),
            ]));
            
            return RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(Vec::<String>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                ],
            ).map_err(GraphError::from);
        }
        
        // Sort by component ID for consistent output
        let mut sorted_nodes: Vec<(&String, &u32)> = component_map.iter().collect();
        sorted_nodes.sort_by_key(|(_, &component_id)| component_id);
        
        let node_ids: Vec<String> = sorted_nodes.iter().map(|(node, _)| (*node).clone()).collect();
        let component_ids: Vec<u32> = sorted_nodes.iter().map(|(_, &comp)| comp).collect();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("component_id", DataType::UInt32, false),
        ]));
        
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(node_ids)),
                Arc::new(UInt32Array::from(component_ids)),
            ],
        ).map_err(GraphError::from)
    }
    
    fn name(&self) -> &'static str {
        "strongly_connected_components"
    }
    
    fn description(&self) -> &'static str {
        "Find strongly connected components using Tarjan's algorithm"
    }
}