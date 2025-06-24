use crate::error::Result;
use crate::graph::ArrowGraph;
use arrow::array::{StringArray, Float64Array};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// Knowledge graph processor for RDF-style triple management
/// Supports RDF/OWL constructs and SPARQL-like queries
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    triples: Vec<Triple>,
    namespaces: HashMap<String, String>,
    ontology: Ontology,
    indexes: TripleIndexes,
}

/// RDF triple representation: subject-predicate-object
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    pub subject: Resource,
    pub predicate: Resource,
    pub object: RDFValue,
}

/// RDF resource (URI or blank node)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    URI(String),
    BlankNode(String),
    Literal(String),
}

/// RDF value (resource or literal)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RDFValue {
    Resource(Resource),
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
}

/// Basic ontology support for OWL constructs
#[derive(Debug, Clone, Default)]
pub struct Ontology {
    classes: HashSet<Resource>,
    properties: HashSet<Resource>,
    subclass_relations: HashMap<Resource, HashSet<Resource>>, // subclass -> superclasses
    subproperty_relations: HashMap<Resource, HashSet<Resource>>, // subproperty -> superproperties
    domain_restrictions: HashMap<Resource, Resource>, // property -> domain class
    range_restrictions: HashMap<Resource, Resource>, // property -> range class
    functional_properties: HashSet<Resource>,
    inverse_functional_properties: HashSet<Resource>,
    symmetric_properties: HashSet<Resource>,
    transitive_properties: HashSet<Resource>,
}

/// Triple indexes for efficient querying
#[derive(Debug, Clone, Default)]
pub struct TripleIndexes {
    spo: HashMap<(Resource, Resource), HashSet<RDFValue>>, // subject-predicate -> objects
    pos: HashMap<(Resource, RDFValue), HashSet<Resource>>, // predicate-object -> subjects
    osp: HashMap<(RDFValue, Resource), HashSet<Resource>>, // object-subject -> predicates
    subject_index: HashMap<Resource, HashSet<usize>>, // subject -> triple indices
    predicate_index: HashMap<Resource, HashSet<usize>>, // predicate -> triple indices
    object_index: HashMap<RDFValue, HashSet<usize>>, // object -> triple indices
}

/// SPARQL-like query structure
#[derive(Debug, Clone)]
pub struct SPARQLQuery {
    pub select_vars: Vec<String>,
    pub where_patterns: Vec<TriplePattern>,
    pub filters: Vec<Filter>,
    pub order_by: Option<Vec<OrderBy>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Triple pattern for SPARQL WHERE clauses
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
}

/// Pattern element (variable or constant)
#[derive(Debug, Clone)]
pub enum PatternElement {
    Variable(String),
    Resource(Resource),
    Value(RDFValue),
}

/// Filter conditions for SPARQL queries
#[derive(Debug, Clone)]
pub enum Filter {
    Equal(String, RDFValue),
    NotEqual(String, RDFValue),
    LessThan(String, String), // Numeric comparison
    GreaterThan(String, String),
    Regex(String, String), // Variable, pattern
    Contains(String, String), // String contains
    Lang(String, String), // Language tag filter
    Datatype(String, String), // Datatype filter
}

/// Ordering specification
#[derive(Debug, Clone)]
pub struct OrderBy {
    pub variable: String,
    pub ascending: bool,
}

/// Query result binding
#[derive(Debug, Clone)]
pub struct Binding {
    pub variables: HashMap<String, RDFValue>,
}

/// Query results
#[derive(Debug, Clone)]
pub struct QueryResults {
    pub bindings: Vec<Binding>,
    pub variables: Vec<String>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            namespaces: HashMap::new(),
            ontology: Ontology::default(),
            indexes: TripleIndexes::default(),
        }
    }

    /// Add a namespace prefix
    pub fn add_namespace(&mut self, prefix: String, uri: String) {
        self.namespaces.insert(prefix, uri);
    }

    /// Add a triple to the knowledge graph
    pub fn add_triple(&mut self, triple: Triple) -> Result<()> {
        let index = self.triples.len();
        
        // Add to indexes
        self.indexes.spo.entry((triple.subject.clone(), triple.predicate.clone()))
            .or_default()
            .insert(triple.object.clone());
        
        self.indexes.pos.entry((triple.predicate.clone(), triple.object.clone()))
            .or_default()
            .insert(triple.subject.clone());
        
        self.indexes.osp.entry((triple.object.clone(), triple.subject.clone()))
            .or_default()
            .insert(triple.predicate.clone());
        
        self.indexes.subject_index.entry(triple.subject.clone())
            .or_default()
            .insert(index);
        
        self.indexes.predicate_index.entry(triple.predicate.clone())
            .or_default()
            .insert(index);
        
        self.indexes.object_index.entry(triple.object.clone())
            .or_default()
            .insert(index);
        
        self.triples.push(triple);
        Ok(())
    }

    /// Import from Arrow Graph
    pub fn from_arrow_graph(&mut self, graph: &ArrowGraph) -> Result<()> {
        // Convert nodes to RDF triples
        let nodes_batch = &graph.nodes;
        if nodes_batch.num_rows() > 0 {
            let node_ids = nodes_batch.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| crate::error::GraphError::graph_construction("Expected string array for node IDs"))?;
            
            for i in 0..node_ids.len() {
                let node_id = node_ids.value(i);
                let triple = Triple {
                    subject: Resource::URI(format!("node:{}", node_id)),
                    predicate: Resource::URI("rdf:type".to_string()),
                    object: RDFValue::Resource(Resource::URI("graph:Node".to_string())),
                };
                self.add_triple(triple)?;
            }
        }

        // Convert edges to RDF triples
        let edges_batch = &graph.edges;
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

                // Edge relationship triple
                let edge_triple = Triple {
                    subject: Resource::URI(format!("node:{}", source)),
                    predicate: Resource::URI("graph:connectedTo".to_string()),
                    object: RDFValue::Resource(Resource::URI(format!("node:{}", target))),
                };
                self.add_triple(edge_triple)?;

                // Edge weight triple
                let weight_triple = Triple {
                    subject: Resource::URI(format!("edge:{}_{}", source, target)),
                    predicate: Resource::URI("graph:weight".to_string()),
                    object: RDFValue::Literal {
                        value: weight.to_string(),
                        datatype: Some("xsd:double".to_string()),
                        language: None,
                    },
                };
                self.add_triple(weight_triple)?;
            }
        }

        Ok(())
    }

    /// Execute a SPARQL-like query
    pub fn query(&self, query: &SPARQLQuery) -> Result<QueryResults> {
        let mut bindings = vec![Binding { variables: HashMap::new() }];

        // Process each triple pattern
        for pattern in &query.where_patterns {
            bindings = self.match_pattern(pattern, bindings)?;
            if bindings.is_empty() {
                break; // No solutions found
            }
        }

        // Apply filters
        for filter in &query.filters {
            bindings = self.apply_filter(filter, bindings)?;
        }

        // Apply ordering
        if let Some(order_specs) = &query.order_by {
            self.sort_bindings(&mut bindings, order_specs);
        }

        // Apply limit and offset
        if let Some(offset) = query.offset {
            if offset < bindings.len() {
                bindings = bindings.into_iter().skip(offset).collect();
            } else {
                bindings.clear();
            }
        }

        if let Some(limit) = query.limit {
            bindings.truncate(limit);
        }

        Ok(QueryResults {
            bindings,
            variables: query.select_vars.clone(),
        })
    }

    /// Match a triple pattern against the knowledge graph
    fn match_pattern(&self, pattern: &TriplePattern, current_bindings: Vec<Binding>) -> Result<Vec<Binding>> {
        let mut new_bindings = Vec::new();

        for binding in current_bindings {
            // Resolve pattern elements with current bindings
            let subjects = self.resolve_pattern_element(&pattern.subject, &binding)?;
            let predicates = self.resolve_pattern_element(&pattern.predicate, &binding)?;
            let objects = self.resolve_pattern_element(&pattern.object, &binding)?;

            // Find matching triples
            for subject in &subjects {
                for predicate in &predicates {
                    for object in &objects {
                        if self.has_triple(subject, predicate, object) {
                            let mut new_binding = binding.clone();
                            
                            // Bind variables
                            if let PatternElement::Variable(var) = &pattern.subject {
                                if let Some(subj_val) = self.resource_to_rdf_value(subject) {
                                    new_binding.variables.insert(var.clone(), subj_val);
                                }
                            }
                            if let PatternElement::Variable(var) = &pattern.predicate {
                                if let Some(pred_val) = self.resource_to_rdf_value(predicate) {
                                    new_binding.variables.insert(var.clone(), pred_val);
                                }
                            }
                            if let PatternElement::Variable(var) = &pattern.object {
                                new_binding.variables.insert(var.clone(), object.clone());
                            }
                            
                            new_bindings.push(new_binding);
                        }
                    }
                }
            }
        }

        Ok(new_bindings)
    }

    /// Resolve a pattern element to concrete values
    fn resolve_pattern_element(&self, element: &PatternElement, binding: &Binding) -> Result<Vec<PatternValue>> {
        match element {
            PatternElement::Variable(var) => {
                if let Some(value) = binding.variables.get(var) {
                    Ok(vec![PatternValue::Value(value.clone())])
                } else {
                    // Unbound variable - return all possible values
                    Ok(self.get_all_values_for_position(element))
                }
            }
            PatternElement::Resource(resource) => {
                Ok(vec![PatternValue::Resource(resource.clone())])
            }
            PatternElement::Value(value) => {
                Ok(vec![PatternValue::Value(value.clone())])
            }
        }
    }

    /// Get all possible values for a pattern position
    fn get_all_values_for_position(&self, element: &PatternElement) -> Vec<PatternValue> {
        match element {
            PatternElement::Variable(_) => {
                // Return all subjects, predicates, or objects depending on context
                let mut values = Vec::new();
                
                // Add all subjects
                for subject in self.indexes.subject_index.keys() {
                    if let Some(rdf_value) = self.resource_to_rdf_value(subject) {
                        values.push(PatternValue::Value(rdf_value));
                    }
                }
                
                // Add all predicates
                for predicate in self.indexes.predicate_index.keys() {
                    if let Some(rdf_value) = self.resource_to_rdf_value(predicate) {
                        values.push(PatternValue::Value(rdf_value));
                    }
                }
                
                // Add all objects
                for object in self.indexes.object_index.keys() {
                    values.push(PatternValue::Value(object.clone()));
                }
                
                values
            }
            _ => Vec::new(),
        }
    }

    /// Check if a triple exists in the knowledge graph
    fn has_triple(&self, subject: &PatternValue, predicate: &PatternValue, object: &PatternValue) -> bool {
        if let (Some(s), Some(p), Some(o)) = (
            self.pattern_value_to_resource(subject),
            self.pattern_value_to_resource(predicate),
            self.pattern_value_to_rdf_value(object)
        ) {
            if let Some(objects) = self.indexes.spo.get(&(s, p)) {
                return objects.contains(&o);
            }
        }
        false
    }

    /// Convert resource to RDF value
    fn resource_to_rdf_value(&self, resource: &Resource) -> Option<RDFValue> {
        Some(RDFValue::Resource(resource.clone()))
    }

    /// Convert pattern value to resource
    fn pattern_value_to_resource(&self, value: &PatternValue) -> Option<Resource> {
        match value {
            PatternValue::Resource(r) => Some(r.clone()),
            PatternValue::Value(RDFValue::Resource(r)) => Some(r.clone()),
            _ => None,
        }
    }

    /// Convert pattern value to RDF value
    fn pattern_value_to_rdf_value(&self, value: &PatternValue) -> Option<RDFValue> {
        match value {
            PatternValue::Value(v) => Some(v.clone()),
            PatternValue::Resource(r) => Some(RDFValue::Resource(r.clone())),
        }
    }

    /// Apply a filter to bindings
    fn apply_filter(&self, filter: &Filter, bindings: Vec<Binding>) -> Result<Vec<Binding>> {
        let mut filtered = Vec::new();

        for binding in bindings {
            let matches = match filter {
                Filter::Equal(var, value) => {
                    binding.variables.get(var) == Some(value)
                }
                Filter::NotEqual(var, value) => {
                    binding.variables.get(var) != Some(value)
                }
                Filter::Regex(var, pattern) => {
                    if let Some(RDFValue::Literal { value, .. }) = binding.variables.get(var) {
                        // Simplified regex matching
                        value.contains(pattern)
                    } else {
                        false
                    }
                }
                Filter::Contains(var, substring) => {
                    if let Some(RDFValue::Literal { value, .. }) = binding.variables.get(var) {
                        value.contains(substring)
                    } else {
                        false
                    }
                }
                Filter::Lang(var, expected_lang) => {
                    if let Some(RDFValue::Literal { language: Some(lang), .. }) = binding.variables.get(var) {
                        lang == expected_lang
                    } else {
                        false
                    }
                }
                Filter::Datatype(var, expected_type) => {
                    if let Some(RDFValue::Literal { datatype: Some(dt), .. }) = binding.variables.get(var) {
                        dt == expected_type
                    } else {
                        false
                    }
                }
                _ => true, // TODO: Implement numeric comparisons
            };

            if matches {
                filtered.push(binding);
            }
        }

        Ok(filtered)
    }

    /// Sort bindings according to ORDER BY specification
    fn sort_bindings(&self, bindings: &mut Vec<Binding>, order_specs: &[OrderBy]) {
        bindings.sort_by(|a, b| {
            for order_spec in order_specs {
                let a_val = a.variables.get(&order_spec.variable);
                let b_val = b.variables.get(&order_spec.variable);
                
                let cmp = match (a_val, b_val) {
                    (Some(a), Some(b)) => self.compare_rdf_values(a, b),
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                
                let final_cmp = if order_spec.ascending {
                    cmp
                } else {
                    cmp.reverse()
                };
                
                if final_cmp != std::cmp::Ordering::Equal {
                    return final_cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    /// Compare two RDF values for sorting
    fn compare_rdf_values(&self, a: &RDFValue, b: &RDFValue) -> std::cmp::Ordering {
        match (a, b) {
            (RDFValue::Literal { value: a_val, .. }, RDFValue::Literal { value: b_val, .. }) => {
                // Try numeric comparison first
                if let (Ok(a_num), Ok(b_num)) = (a_val.parse::<f64>(), b_val.parse::<f64>()) {
                    a_num.partial_cmp(&b_num).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a_val.cmp(b_val)
                }
            }
            (RDFValue::Resource(Resource::URI(a)), RDFValue::Resource(Resource::URI(b))) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Export knowledge graph to Arrow format
    pub fn to_arrow_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("subject", DataType::Utf8, false),
            Field::new("predicate", DataType::Utf8, false),
            Field::new("object", DataType::Utf8, false),
            Field::new("object_type", DataType::Utf8, false),
        ]));

        let mut subjects = Vec::new();
        let mut predicates = Vec::new();
        let mut objects = Vec::new();
        let mut object_types = Vec::new();

        for triple in &self.triples {
            subjects.push(self.resource_to_string(&triple.subject));
            predicates.push(self.resource_to_string(&triple.predicate));
            
            match &triple.object {
                RDFValue::Resource(resource) => {
                    objects.push(self.resource_to_string(resource));
                    object_types.push("resource".to_string());
                }
                RDFValue::Literal { value, datatype, language } => {
                    objects.push(value.clone());
                    if let Some(dt) = datatype {
                        object_types.push(format!("literal:{}", dt));
                    } else if let Some(lang) = language {
                        object_types.push(format!("literal@{}", lang));
                    } else {
                        object_types.push("literal".to_string());
                    }
                }
            }
        }

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(subjects)),
                Arc::new(StringArray::from(predicates)),
                Arc::new(StringArray::from(objects)),
                Arc::new(StringArray::from(object_types)),
            ],
        )?;

        Ok(batch)
    }

    /// Convert resource to string representation
    fn resource_to_string(&self, resource: &Resource) -> String {
        match resource {
            Resource::URI(uri) => uri.clone(),
            Resource::BlankNode(id) => format!("_:{}", id),
            Resource::Literal(value) => value.clone(),
        }
    }

    /// Get statistics about the knowledge graph
    pub fn get_statistics(&self) -> KnowledgeGraphStats {
        let mut unique_subjects = HashSet::new();
        let mut unique_predicates = HashSet::new();
        let mut unique_objects = HashSet::new();

        for triple in &self.triples {
            unique_subjects.insert(&triple.subject);
            unique_predicates.insert(&triple.predicate);
            unique_objects.insert(&triple.object);
        }

        KnowledgeGraphStats {
            total_triples: self.triples.len(),
            unique_subjects: unique_subjects.len(),
            unique_predicates: unique_predicates.len(),
            unique_objects: unique_objects.len(),
            classes: self.ontology.classes.len(),
            properties: self.ontology.properties.len(),
        }
    }
}

/// Pattern value helper enum
#[derive(Debug, Clone)]
enum PatternValue {
    Resource(Resource),
    Value(RDFValue),
}

/// Knowledge graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphStats {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub classes: usize,
    pub properties: usize,
}

/// Helper functions for creating common RDF constructs
impl KnowledgeGraph {
    /// Add an RDF class declaration
    pub fn add_class(&mut self, class_uri: String) -> Result<()> {
        let class_resource = Resource::URI(class_uri.clone());
        self.ontology.classes.insert(class_resource.clone());
        
        let triple = Triple {
            subject: class_resource,
            predicate: Resource::URI("rdf:type".to_string()),
            object: RDFValue::Resource(Resource::URI("owl:Class".to_string())),
        };
        self.add_triple(triple)
    }

    /// Add a property declaration
    pub fn add_property(&mut self, property_uri: String, property_type: PropertyType) -> Result<()> {
        let property_resource = Resource::URI(property_uri.clone());
        self.ontology.properties.insert(property_resource.clone());
        
        let type_uri = match property_type {
            PropertyType::ObjectProperty => "owl:ObjectProperty",
            PropertyType::DatatypeProperty => "owl:DatatypeProperty",
            PropertyType::AnnotationProperty => "owl:AnnotationProperty",
        };
        
        let triple = Triple {
            subject: property_resource,
            predicate: Resource::URI("rdf:type".to_string()),
            object: RDFValue::Resource(Resource::URI(type_uri.to_string())),
        };
        self.add_triple(triple)
    }

    /// Add a subclass relationship
    pub fn add_subclass_of(&mut self, subclass: String, superclass: String) -> Result<()> {
        let sub_resource = Resource::URI(subclass);
        let super_resource = Resource::URI(superclass);
        
        self.ontology.subclass_relations
            .entry(sub_resource.clone())
            .or_default()
            .insert(super_resource.clone());
        
        let triple = Triple {
            subject: sub_resource,
            predicate: Resource::URI("rdfs:subClassOf".to_string()),
            object: RDFValue::Resource(super_resource),
        };
        self.add_triple(triple)
    }
}

/// Property types for OWL
#[derive(Debug, Clone)]
pub enum PropertyType {
    ObjectProperty,
    DatatypeProperty,
    AnnotationProperty,
}

/// Helper function to create SPARQL queries
pub fn sparql_select(vars: Vec<&str>) -> SPARQLQueryBuilder {
    SPARQLQueryBuilder::new(vars.into_iter().map(|s| s.to_string()).collect())
}

/// Builder for SPARQL queries
#[derive(Debug)]
pub struct SPARQLQueryBuilder {
    query: SPARQLQuery,
}

impl SPARQLQueryBuilder {
    pub fn new(select_vars: Vec<String>) -> Self {
        Self {
            query: SPARQLQuery {
                select_vars,
                where_patterns: Vec::new(),
                filters: Vec::new(),
                order_by: None,
                limit: None,
                offset: None,
            },
        }
    }

    pub fn where_triple(mut self, subject: PatternElement, predicate: PatternElement, object: PatternElement) -> Self {
        self.query.where_patterns.push(TriplePattern { subject, predicate, object });
        self
    }

    pub fn filter(mut self, filter: Filter) -> Self {
        self.query.filters.push(filter);
        self
    }

    pub fn order_by(mut self, variable: String, ascending: bool) -> Self {
        let order_spec = OrderBy { variable, ascending };
        if let Some(ref mut order_specs) = self.query.order_by {
            order_specs.push(order_spec);
        } else {
            self.query.order_by = Some(vec![order_spec]);
        }
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.query.offset = Some(offset);
        self
    }

    pub fn build(self) -> SPARQLQuery {
        self.query
    }
}

/// Helper functions for creating pattern elements
pub fn var(name: &str) -> PatternElement {
    PatternElement::Variable(name.to_string())
}

pub fn uri(uri: &str) -> PatternElement {
    PatternElement::Resource(Resource::URI(uri.to_string()))
}

pub fn literal(value: &str) -> PatternElement {
    PatternElement::Value(RDFValue::Literal {
        value: value.to_string(),
        datatype: None,
        language: None,
    })
}

pub fn typed_literal(value: &str, datatype: &str) -> PatternElement {
    PatternElement::Value(RDFValue::Literal {
        value: value.to_string(),
        datatype: Some(datatype.to_string()),
        language: None,
    })
}

pub fn lang_literal(value: &str, language: &str) -> PatternElement {
    PatternElement::Value(RDFValue::Literal {
        value: value.to_string(),
        datatype: None,
        language: Some(language.to_string()),
    })
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
        let node_ids = StringArray::from(vec!["Person1", "Person2", "Organization1"]);
        let nodes_batch = RecordBatch::try_new(
            nodes_schema,
            vec![Arc::new(node_ids)],
        )?;

        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let sources = StringArray::from(vec!["Person1", "Person2"]);
        let targets = StringArray::from(vec!["Organization1", "Organization1"]);
        let weights = Float64Array::from(vec![1.0, 0.8]);
        let edges_batch = RecordBatch::try_new(
            edges_schema,
            vec![Arc::new(sources), Arc::new(targets), Arc::new(weights)],
        )?;

        ArrowGraph::new(nodes_batch, edges_batch)
    }

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.triples.len(), 0);
        assert_eq!(kg.namespaces.len(), 0);
    }

    #[test]
    fn test_add_triple() {
        let mut kg = KnowledgeGraph::new();
        
        let triple = Triple {
            subject: Resource::URI("ex:person1".to_string()),
            predicate: Resource::URI("ex:name".to_string()),
            object: RDFValue::Literal {
                value: "John Doe".to_string(),
                datatype: Some("xsd:string".to_string()),
                language: None,
            },
        };
        
        kg.add_triple(triple).unwrap();
        assert_eq!(kg.triples.len(), 1);
        assert_eq!(kg.indexes.subject_index.len(), 1);
    }

    #[test]
    fn test_from_arrow_graph() {
        let graph = create_test_graph().unwrap();
        let mut kg = KnowledgeGraph::new();
        
        kg.from_arrow_graph(&graph).unwrap();
        
        // Should have triples for nodes and edges
        assert!(kg.triples.len() > 0);
        
        let stats = kg.get_statistics();
        assert!(stats.total_triples >= 5); // At least 3 nodes + 2 edges
    }

    #[test]
    fn test_sparql_query_builder() {
        let query = sparql_select(vec!["?person", "?name"])
            .where_triple(
                var("person"),
                uri("ex:name"),
                var("name")
            )
            .filter(Filter::Contains("name".to_string(), "John".to_string()))
            .limit(10)
            .build();
        
        assert_eq!(query.select_vars.len(), 2);
        assert_eq!(query.where_patterns.len(), 1);
        assert_eq!(query.filters.len(), 1);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_add_ontology_constructs() {
        let mut kg = KnowledgeGraph::new();
        
        kg.add_class("ex:Person".to_string()).unwrap();
        kg.add_property("ex:name".to_string(), PropertyType::DatatypeProperty).unwrap();
        kg.add_subclass_of("ex:Employee".to_string(), "ex:Person".to_string()).unwrap();
        
        assert_eq!(kg.ontology.classes.len(), 1);
        assert_eq!(kg.ontology.properties.len(), 1);
        assert_eq!(kg.ontology.subclass_relations.len(), 1);
        
        // Should have created triples for these constructs
        assert!(kg.triples.len() >= 3);
    }

    #[test]
    fn test_namespace_management() {
        let mut kg = KnowledgeGraph::new();
        
        kg.add_namespace("ex".to_string(), "http://example.org/".to_string());
        kg.add_namespace("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        
        assert_eq!(kg.namespaces.len(), 2);
        assert_eq!(kg.namespaces.get("ex"), Some(&"http://example.org/".to_string()));
    }

    #[test]
    fn test_to_arrow_batch() {
        let mut kg = KnowledgeGraph::new();
        
        let triple = Triple {
            subject: Resource::URI("ex:person1".to_string()),
            predicate: Resource::URI("ex:name".to_string()),
            object: RDFValue::Literal {
                value: "John".to_string(),
                datatype: Some("xsd:string".to_string()),
                language: None,
            },
        };
        
        kg.add_triple(triple).unwrap();
        
        let batch = kg.to_arrow_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 4);
    }

    #[test]
    fn test_simple_sparql_query() {
        let mut kg = KnowledgeGraph::new();
        
        // Add some test data
        let triple1 = Triple {
            subject: Resource::URI("ex:person1".to_string()),
            predicate: Resource::URI("ex:name".to_string()),
            object: RDFValue::Literal {
                value: "John".to_string(),
                datatype: None,
                language: None,
            },
        };
        
        let triple2 = Triple {
            subject: Resource::URI("ex:person1".to_string()),
            predicate: Resource::URI("ex:age".to_string()),
            object: RDFValue::Literal {
                value: "30".to_string(),
                datatype: Some("xsd:integer".to_string()),
                language: None,
            },
        };
        
        kg.add_triple(triple1).unwrap();
        kg.add_triple(triple2).unwrap();
        
        // Query for all properties of person1
        let query = sparql_select(vec!["?property", "?value"])
            .where_triple(
                uri("ex:person1"),
                var("property"),
                var("value")
            )
            .build();
        
        let results = kg.query(&query).unwrap();
        assert_eq!(results.bindings.len(), 2);
        assert_eq!(results.variables.len(), 2);
    }

    #[test]
    fn test_rdf_value_comparison() {
        let kg = KnowledgeGraph::new();
        
        let literal1 = RDFValue::Literal {
            value: "10".to_string(),
            datatype: Some("xsd:integer".to_string()),
            language: None,
        };
        
        let literal2 = RDFValue::Literal {
            value: "20".to_string(),
            datatype: Some("xsd:integer".to_string()),
            language: None,
        };
        
        let cmp = kg.compare_rdf_values(&literal1, &literal2);
        assert_eq!(cmp, std::cmp::Ordering::Less);
    }
}