/// Standard Query Interfaces Module for arrow-graph v0.8.0 "Performance & Standards"
/// 
/// This module provides standard graph query language support:
/// - GraphQL interface for graph queries
/// - Cypher query language support (Neo4j compatible)
/// - SPARQL for RDF/knowledge graphs
/// - SQL-like graph extensions
/// - Query optimization and execution planning

pub mod graphql;
pub mod cypher;
pub mod sparql;
pub mod sql_extensions;
pub mod query_planner;

// Re-export core interface components
pub use graphql::{
    GraphQLSchema, GraphQLResolver, GraphQLQuery, GraphQLMutation,
    GraphQLSubscription, GraphQLContext
};

pub use cypher::{
    CypherParser, CypherExecutor, CypherQuery, CypherResult,
    MatchPattern, WhereClause, ReturnClause
};

pub use sparql::{
    SPARQLParser, SPARQLExecutor, SPARQLQuery, TriplePattern,
    RDFGraph, SPARQLResult
};

pub use sql_extensions::{
    GraphSQLExtension, PathQuery, ReachabilityQuery,
    ShortestPathFunction, ConnectedComponentsFunction
};

pub use query_planner::{
    QueryPlanner, ExecutionPlan, QueryOptimizer, CostEstimator,
    IndexSelector, JoinStrategy
};

/// Central query interface engine
#[derive(Debug)]
pub struct QueryInterfaceEngine {
    graphql_schema: Option<GraphQLSchema>,
    cypher_parser: CypherParser,
    sparql_parser: SPARQLParser,
    sql_extensions: GraphSQLExtension,
    query_planner: QueryPlanner,
    config: InterfaceConfig,
}

/// Configuration for query interfaces
#[derive(Debug, Clone)]
pub struct InterfaceConfig {
    pub enable_graphql: bool,
    pub enable_cypher: bool,
    pub enable_sparql: bool,
    pub enable_sql_extensions: bool,
    pub query_timeout: std::time::Duration,
    pub max_query_complexity: usize,
    pub enable_query_caching: bool,
    pub cache_size: usize,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            enable_graphql: true,
            enable_cypher: true,
            enable_sparql: false, // RDF support optional
            enable_sql_extensions: true,
            query_timeout: std::time::Duration::from_secs(30),
            max_query_complexity: 1000,
            enable_query_caching: true,
            cache_size: 10_000,
        }
    }
}

impl QueryInterfaceEngine {
    /// Create a new query interface engine
    pub fn new(config: InterfaceConfig) -> crate::error::Result<Self> {
        let graphql_schema = if config.enable_graphql {
            Some(GraphQLSchema::new()?)
        } else {
            None
        };
        
        Ok(Self {
            graphql_schema,
            cypher_parser: CypherParser::new(),
            sparql_parser: SPARQLParser::new(),
            sql_extensions: GraphSQLExtension::new(),
            query_planner: QueryPlanner::new(&config)?,
            config,
        })
    }
    
    /// Execute a GraphQL query
    pub async fn execute_graphql(
        &mut self,
        query: &str,
        variables: Option<&serde_json::Value>,
    ) -> crate::error::Result<GraphQLResult> {
        if !self.config.enable_graphql {
            return Err(crate::error::ArrowGraphError::UnsupportedOperation(
                "GraphQL interface is disabled".to_string()
            ));
        }
        
        let schema = self.graphql_schema.as_ref()
            .ok_or_else(|| crate::error::ArrowGraphError::Configuration(
                "GraphQL schema not initialized".to_string()
            ))?;
        
        let parsed_query = schema.parse_query(query)?;
        let execution_plan = self.query_planner.plan_graphql(&parsed_query)?;
        
        // Execute with timeout
        let result = tokio::time::timeout(
            self.config.query_timeout,
            schema.execute_query(parsed_query, variables, execution_plan)
        ).await??;
        
        Ok(result)
    }
    
    /// Execute a Cypher query
    pub async fn execute_cypher(
        &mut self,
        query: &str,
        parameters: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> crate::error::Result<CypherResult> {
        if !self.config.enable_cypher {
            return Err(crate::error::ArrowGraphError::UnsupportedOperation(
                "Cypher interface is disabled".to_string()
            ));
        }
        
        let parsed_query = self.cypher_parser.parse(query)?;
        let execution_plan = self.query_planner.plan_cypher(&parsed_query)?;
        
        // Execute with timeout
        let result = tokio::time::timeout(
            self.config.query_timeout,
            self.cypher_parser.execute(parsed_query, parameters, execution_plan)
        ).await??;
        
        Ok(result)
    }
    
    /// Execute a SPARQL query (for RDF/knowledge graphs)
    pub async fn execute_sparql(
        &mut self,
        query: &str,
    ) -> crate::error::Result<SPARQLResult> {
        if !self.config.enable_sparql {
            return Err(crate::error::ArrowGraphError::UnsupportedOperation(
                "SPARQL interface is disabled".to_string()
            ));
        }
        
        let parsed_query = self.sparql_parser.parse(query)?;
        let execution_plan = self.query_planner.plan_sparql(&parsed_query)?;
        
        // Execute with timeout
        let result = tokio::time::timeout(
            self.config.query_timeout,
            self.sparql_parser.execute(parsed_query, execution_plan)
        ).await??;
        
        Ok(result)
    }
    
    /// Execute SQL with graph extensions
    pub async fn execute_sql_graph(
        &mut self,
        sql: &str,
    ) -> crate::error::Result<arrow::record_batch::RecordBatch> {
        if !self.config.enable_sql_extensions {
            return Err(crate::error::ArrowGraphError::UnsupportedOperation(
                "SQL graph extensions are disabled".to_string()
            ));
        }
        
        let parsed_query = self.sql_extensions.parse_sql(sql)?;
        let execution_plan = self.query_planner.plan_sql(&parsed_query)?;
        
        // Execute with timeout
        let result = tokio::time::timeout(
            self.config.query_timeout,
            self.sql_extensions.execute(parsed_query, execution_plan)
        ).await??;
        
        Ok(result)
    }
    
    /// Get query execution statistics
    pub fn get_query_stats(&self) -> QueryStats {
        QueryStats {
            total_queries: self.query_planner.total_queries(),
            cached_queries: self.query_planner.cached_queries(),
            average_execution_time: self.query_planner.average_execution_time(),
            query_type_distribution: self.query_planner.query_type_distribution(),
        }
    }
}

/// Query execution result types
#[derive(Debug)]
pub struct GraphQLResult {
    pub data: Option<serde_json::Value>,
    pub errors: Vec<GraphQLError>,
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct GraphQLError {
    pub message: String,
    pub locations: Vec<GraphQLLocation>,
    pub path: Option<Vec<serde_json::Value>>,
}

#[derive(Debug)]
pub struct GraphQLLocation {
    pub line: usize,
    pub column: usize,
}

/// Query execution statistics
#[derive(Debug)]
pub struct QueryStats {
    pub total_queries: usize,
    pub cached_queries: usize,
    pub average_execution_time: std::time::Duration,
    pub query_type_distribution: std::collections::HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interface_config_default() {
        let config = InterfaceConfig::default();
        assert!(config.enable_graphql);
        assert!(config.enable_cypher);
        assert!(config.enable_sql_extensions);
        assert_eq!(config.query_timeout, std::time::Duration::from_secs(30));
    }
    
    #[tokio::test]
    async fn test_query_engine_creation() {
        let config = InterfaceConfig::default();
        let engine = QueryInterfaceEngine::new(config);
        assert!(engine.is_ok());
    }
}