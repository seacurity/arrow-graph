use datafusion::execution::context::SessionContext;
use crate::error::Result;
use crate::sql::graph_functions::register_all_graph_functions;
use crate::sql::window_functions::register_window_functions;

pub struct GraphSqlExtension;

impl GraphSqlExtension {
    pub fn new() -> Self {
        Self
    }
    
    /// Register all graph SQL functions and window functions with DataFusion context
    pub fn register_functions(&self, ctx: &mut SessionContext) -> Result<()> {
        // Register scalar functions
        register_all_graph_functions(ctx)
            .map_err(|e| crate::error::GraphError::algorithm(format!("Failed to register graph functions: {}", e)))?;
        
        // Register window functions
        register_window_functions(ctx)
            .map_err(|e| crate::error::GraphError::algorithm(format!("Failed to register window functions: {}", e)))?;
        
        Ok(())
    }
}

impl Default for GraphSqlExtension {
    fn default() -> Self {
        Self::new()
    }
}