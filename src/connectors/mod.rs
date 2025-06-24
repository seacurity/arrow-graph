/// Database Connectors Module for arrow-graph v0.8.0 "Performance & Standards"
/// 
/// This module provides a plugin architecture for database connectivity:
/// - PostgreSQL graph extensions (AGE, pg_graph)
/// - Neo4j bolt protocol connector
/// - ClickHouse columnar storage
/// - DuckDB integration for analytics
/// - Apache AGE (PostgreSQL graph extension)
/// - Standard connector trait system

pub mod postgresql;
pub mod neo4j;
pub mod clickhouse;
pub mod duckdb;
pub mod age;
pub mod traits;

// Re-export core connector components
pub use traits::{
    DatabaseConnector, QueryExecutor, DataImporter, DataExporter,
    ConnectionPool, Transaction, ConnectorError
};

pub use postgresql::{
    PostgreSQLConnector, PostgreSQLConfig, PostgreSQLPool,
    GraphExtensionSupport
};

pub use neo4j::{
    Neo4jConnector, Neo4jConfig, BoltProtocol,
    CypherTranslator, Neo4jPool
};

pub use clickhouse::{
    ClickHouseConnector, ClickHouseConfig, ClickHousePool,
    ColumnarGraphStorage
};

pub use duckdb::{
    DuckDBConnector, DuckDBConfig, DuckDBInMemory,
    AnalyticsIntegration
};

pub use age::{
    AGEConnector, AGEConfig, AGEPool,
    CypherToAGETranslator
};

/// Central connector management system
#[derive(Debug)]
pub struct ConnectorManager {
    connectors: std::collections::HashMap<String, Box<dyn DatabaseConnector>>,
    connection_pools: std::collections::HashMap<String, Box<dyn ConnectionPool>>,
    config: ConnectorConfig,
}

/// Configuration for database connectors
#[derive(Debug, Clone)]
pub struct ConnectorConfig {
    pub max_connections_per_pool: usize,
    pub connection_timeout: std::time::Duration,
    pub query_timeout: std::time::Duration,
    pub retry_attempts: usize,
    pub enable_connection_pooling: bool,
    pub enable_prepared_statements: bool,
    pub batch_size: usize,
}

impl Default for ConnectorConfig {
    fn default() -> Self {
        Self {
            max_connections_per_pool: 10,
            connection_timeout: std::time::Duration::from_secs(30),
            query_timeout: std::time::Duration::from_secs(300), // 5 minutes
            retry_attempts: 3,
            enable_connection_pooling: true,
            enable_prepared_statements: true,
            batch_size: 10_000,
        }
    }
}

impl ConnectorManager {
    /// Create a new connector manager
    pub fn new(config: ConnectorConfig) -> Self {
        Self {
            connectors: std::collections::HashMap::new(),
            connection_pools: std::collections::HashMap::new(),
            config,
        }
    }
    
    /// Register a PostgreSQL connector
    pub async fn add_postgresql(
        &mut self,
        name: &str,
        config: PostgreSQLConfig,
    ) -> crate::error::Result<()> {
        let connector = PostgreSQLConnector::new(config).await?;
        let pool = connector.create_pool(&self.config).await?;
        
        self.connectors.insert(name.to_string(), Box::new(connector));
        self.connection_pools.insert(name.to_string(), pool);
        
        Ok(())
    }
    
    /// Register a Neo4j connector
    pub async fn add_neo4j(
        &mut self,
        name: &str,
        config: Neo4jConfig,
    ) -> crate::error::Result<()> {
        let connector = Neo4jConnector::new(config).await?;
        let pool = connector.create_pool(&self.config).await?;
        
        self.connectors.insert(name.to_string(), Box::new(connector));
        self.connection_pools.insert(name.to_string(), pool);
        
        Ok(())
    }
    
    /// Register a ClickHouse connector
    pub async fn add_clickhouse(
        &mut self,
        name: &str,
        config: ClickHouseConfig,
    ) -> crate::error::Result<()> {
        let connector = ClickHouseConnector::new(config).await?;
        let pool = connector.create_pool(&self.config).await?;
        
        self.connectors.insert(name.to_string(), Box::new(connector));
        self.connection_pools.insert(name.to_string(), pool);
        
        Ok(())
    }
    
    /// Register a DuckDB connector
    pub async fn add_duckdb(
        &mut self,
        name: &str,
        config: DuckDBConfig,
    ) -> crate::error::Result<()> {
        let connector = DuckDBConnector::new(config).await?;
        let pool = connector.create_pool(&self.config).await?;
        
        self.connectors.insert(name.to_string(), Box::new(connector));
        self.connection_pools.insert(name.to_string(), pool);
        
        Ok(())
    }
    
    /// Register an Apache AGE connector
    pub async fn add_age(
        &mut self,
        name: &str,
        config: AGEConfig,
    ) -> crate::error::Result<()> {
        let connector = AGEConnector::new(config).await?;
        let pool = connector.create_pool(&self.config).await?;
        
        self.connectors.insert(name.to_string(), Box::new(connector));
        self.connection_pools.insert(name.to_string(), pool);
        
        Ok(())
    }
    
    /// Get a connector by name
    pub fn get_connector(&self, name: &str) -> Option<&dyn DatabaseConnector> {
        self.connectors.get(name).map(|c| c.as_ref())
    }
    
    /// Get a connection pool by name
    pub fn get_pool(&self, name: &str) -> Option<&dyn ConnectionPool> {
        self.connection_pools.get(name).map(|p| p.as_ref())
    }
    
    /// Import graph data from a database
    pub async fn import_graph(
        &self,
        connector_name: &str,
        import_config: &ImportConfig,
    ) -> crate::error::Result<crate::graph::ArrowGraph> {
        let connector = self.get_connector(connector_name)
            .ok_or_else(|| crate::error::ArrowGraphError::Configuration(
                format!("Connector '{}' not found", connector_name)
            ))?;
        
        let pool = self.get_pool(connector_name)
            .ok_or_else(|| crate::error::ArrowGraphError::Configuration(
                format!("Connection pool '{}' not found", connector_name)
            ))?;
        
        let connection = pool.get_connection().await?;
        connector.import_graph(connection, import_config).await
    }
    
    /// Export graph data to a database
    pub async fn export_graph(
        &self,
        connector_name: &str,
        graph: &crate::graph::ArrowGraph,
        export_config: &ExportConfig,
    ) -> crate::error::Result<()> {
        let connector = self.get_connector(connector_name)
            .ok_or_else(|| crate::error::ArrowGraphError::Configuration(
                format!("Connector '{}' not found", connector_name)
            ))?;
        
        let pool = self.get_pool(connector_name)
            .ok_or_else(|| crate::error::ArrowGraphError::Configuration(
                format!("Connection pool '{}' not found", connector_name)
            ))?;
        
        let connection = pool.get_connection().await?;
        connector.export_graph(connection, graph, export_config).await
    }
    
    /// List all registered connectors
    pub fn list_connectors(&self) -> Vec<&str> {
        self.connectors.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get connector statistics
    pub fn get_connector_stats(&self, name: &str) -> Option<ConnectorStats> {
        let connector = self.get_connector(name)?;
        Some(connector.get_stats())
    }
}

/// Import configuration
#[derive(Debug, Clone)]
pub struct ImportConfig {
    pub node_query: String,
    pub edge_query: String,
    pub node_id_column: String,
    pub edge_source_column: String,
    pub edge_target_column: String,
    pub property_columns: Vec<String>,
    pub batch_size: usize,
    pub parallel_import: bool,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub target_table_nodes: String,
    pub target_table_edges: String,
    pub create_tables: bool,
    pub batch_size: usize,
    pub parallel_export: bool,
    pub upsert_mode: bool,
}

/// Connector statistics
#[derive(Debug, Clone)]
pub struct ConnectorStats {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub average_query_time: std::time::Duration,
    pub active_connections: usize,
    pub total_connections: usize,
    pub data_transferred_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_connector_config_default() {
        let config = ConnectorConfig::default();
        assert_eq!(config.max_connections_per_pool, 10);
        assert!(config.enable_connection_pooling);
        assert!(config.enable_prepared_statements);
    }
    
    #[test]
    fn test_connector_manager_creation() {
        let config = ConnectorConfig::default();
        let manager = ConnectorManager::new(config);
        assert!(manager.list_connectors().is_empty());
    }
    
    #[test]
    fn test_import_config_creation() {
        let import_config = ImportConfig {
            node_query: "SELECT id, name FROM nodes".to_string(),
            edge_query: "SELECT source, target FROM edges".to_string(),
            node_id_column: "id".to_string(),
            edge_source_column: "source".to_string(),
            edge_target_column: "target".to_string(),
            property_columns: vec!["name".to_string()],
            batch_size: 10_000,
            parallel_import: true,
        };
        
        assert_eq!(import_config.batch_size, 10_000);
        assert!(import_config.parallel_import);
    }
}