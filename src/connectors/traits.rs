/// Core traits for database connectors
/// 
/// This module defines the standard interface that all database connectors
/// must implement to integrate with the arrow-graph system.

use crate::error::Result;
use crate::graph::ArrowGraph;
use crate::connectors::{ImportConfig, ExportConfig, ConnectorStats};
use async_trait::async_trait;
use std::collections::HashMap;

/// Core trait for database connectivity
#[async_trait]
pub trait DatabaseConnector: Send + Sync {
    /// Connect to the database
    async fn connect(&mut self) -> Result<()>;
    
    /// Disconnect from the database
    async fn disconnect(&mut self) -> Result<()>;
    
    /// Test connection health
    async fn health_check(&self) -> Result<bool>;
    
    /// Import graph data from the database
    async fn import_graph(
        &self,
        connection: Box<dyn Connection>,
        config: &ImportConfig,
    ) -> Result<ArrowGraph>;
    
    /// Export graph data to the database
    async fn export_graph(
        &self,
        connection: Box<dyn Connection>,
        graph: &ArrowGraph,
        config: &ExportConfig,
    ) -> Result<()>;
    
    /// Create a connection pool
    async fn create_pool(
        &self,
        config: &crate::connectors::ConnectorConfig,
    ) -> Result<Box<dyn ConnectionPool>>;
    
    /// Get connector statistics
    fn get_stats(&self) -> ConnectorStats;
    
    /// Get connector name/type
    fn connector_type(&self) -> &str;
}

/// Database connection trait
#[async_trait]
pub trait Connection: Send + Sync {
    /// Execute a query and return results
    async fn execute_query(&mut self, query: &str) -> Result<QueryResult>;
    
    /// Execute a prepared statement
    async fn execute_prepared(
        &mut self,
        statement: &PreparedStatement,
        params: &[QueryParam],
    ) -> Result<QueryResult>;
    
    /// Begin a transaction
    async fn begin_transaction(&mut self) -> Result<Box<dyn Transaction>>;
    
    /// Check if connection is valid
    async fn is_valid(&self) -> bool;
    
    /// Get connection metadata
    fn get_metadata(&self) -> ConnectionMetadata;
}

/// Transaction management trait
#[async_trait]
pub trait Transaction: Send + Sync {
    /// Commit the transaction
    async fn commit(self: Box<Self>) -> Result<()>;
    
    /// Rollback the transaction
    async fn rollback(self: Box<Self>) -> Result<()>;
    
    /// Execute query within transaction
    async fn execute_query(&mut self, query: &str) -> Result<QueryResult>;
    
    /// Get transaction ID
    fn transaction_id(&self) -> String;
}

/// Connection pool management trait
#[async_trait]
pub trait ConnectionPool: Send + Sync {
    /// Get a connection from the pool
    async fn get_connection(&self) -> Result<Box<dyn Connection>>;
    
    /// Return a connection to the pool
    async fn return_connection(&self, connection: Box<dyn Connection>) -> Result<()>;
    
    /// Get pool statistics
    fn get_pool_stats(&self) -> PoolStats;
    
    /// Close all connections in the pool
    async fn close_all(&mut self) -> Result<()>;
}

/// Query execution trait
#[async_trait]
pub trait QueryExecutor: Send + Sync {
    /// Execute a single query
    async fn execute(&mut self, query: &Query) -> Result<QueryResult>;
    
    /// Execute multiple queries in batch
    async fn execute_batch(&mut self, queries: &[Query]) -> Result<Vec<QueryResult>>;
    
    /// Prepare a statement for repeated execution
    async fn prepare(&mut self, sql: &str) -> Result<PreparedStatement>;
    
    /// Get query execution plan
    async fn explain(&mut self, query: &Query) -> Result<ExecutionPlan>;
}

/// Data import trait
#[async_trait]
pub trait DataImporter: Send + Sync {
    /// Import nodes from database
    async fn import_nodes(
        &self,
        connection: &mut dyn Connection,
        config: &ImportConfig,
    ) -> Result<arrow::record_batch::RecordBatch>;
    
    /// Import edges from database
    async fn import_edges(
        &self,
        connection: &mut dyn Connection,
        config: &ImportConfig,
    ) -> Result<arrow::record_batch::RecordBatch>;
    
    /// Import with custom query
    async fn import_custom(
        &self,
        connection: &mut dyn Connection,
        query: &str,
    ) -> Result<arrow::record_batch::RecordBatch>;
}

/// Data export trait
#[async_trait]
pub trait DataExporter: Send + Sync {
    /// Export nodes to database
    async fn export_nodes(
        &self,
        connection: &mut dyn Connection,
        nodes: &arrow::record_batch::RecordBatch,
        config: &ExportConfig,
    ) -> Result<usize>;
    
    /// Export edges to database
    async fn export_edges(
        &self,
        connection: &mut dyn Connection,
        edges: &arrow::record_batch::RecordBatch,
        config: &ExportConfig,
    ) -> Result<usize>;
    
    /// Export with custom target
    async fn export_custom(
        &self,
        connection: &mut dyn Connection,
        data: &arrow::record_batch::RecordBatch,
        target_table: &str,
    ) -> Result<usize>;
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct Query {
    pub sql: String,
    pub parameters: Vec<QueryParam>,
    pub timeout: Option<std::time::Duration>,
}

#[derive(Debug, Clone)]
pub enum QueryParam {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Binary(Vec<u8>),
}

#[derive(Debug)]
pub struct QueryResult {
    pub rows_affected: usize,
    pub data: Option<arrow::record_batch::RecordBatch>,
    pub execution_time: std::time::Duration,
    pub metadata: QueryMetadata,
}

#[derive(Debug)]
pub struct QueryMetadata {
    pub column_names: Vec<String>,
    pub column_types: Vec<String>,
    pub row_count: usize,
    pub query_id: Option<String>,
}

#[derive(Debug)]
pub struct PreparedStatement {
    pub statement_id: String,
    pub sql: String,
    pub parameter_count: usize,
    pub parameter_types: Vec<String>,
}

#[derive(Debug)]
pub struct ExecutionPlan {
    pub plan_text: String,
    pub estimated_cost: f64,
    pub estimated_rows: usize,
    pub operations: Vec<PlanOperation>,
}

#[derive(Debug)]
pub struct PlanOperation {
    pub operation_type: String,
    pub description: String,
    pub cost: f64,
    pub rows: usize,
}

#[derive(Debug)]
pub struct ConnectionMetadata {
    pub database_name: String,
    pub database_version: String,
    pub server_version: String,
    pub connection_id: String,
    pub features: HashMap<String, bool>,
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub active_connections: usize,
    pub idle_connections: usize,
    pub total_connections: usize,
    pub max_connections: usize,
    pub connections_created: usize,
    pub connections_closed: usize,
    pub average_checkout_time: std::time::Duration,
}

/// Error types specific to connectors
#[derive(Debug, thiserror::Error)]
pub enum ConnectorError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Query execution failed: {0}")]
    QueryFailed(String),
    
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
    
    #[error("Pool exhausted: no connections available")]
    PoolExhausted,
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Data conversion error: {0}")]
    DataConversionError(String),
    
    #[error("Timeout: operation exceeded {0:?}")]
    Timeout(std::time::Duration),
}

impl From<ConnectorError> for crate::error::ArrowGraphError {
    fn from(err: ConnectorError) -> Self {
        crate::error::ArrowGraphError::ConnectorError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_query_param_creation() {
        let param = QueryParam::String("test".to_string());
        match param {
            QueryParam::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected String parameter"),
        }
    }
    
    #[test]
    fn test_pool_stats() {
        let stats = PoolStats {
            active_connections: 5,
            idle_connections: 3,
            total_connections: 8,
            max_connections: 10,
            connections_created: 8,
            connections_closed: 0,
            average_checkout_time: std::time::Duration::from_millis(100),
        };
        
        assert_eq!(stats.total_connections, 8);
        assert_eq!(stats.active_connections + stats.idle_connections, 8);
    }
}