use datafusion::execution::context::SessionContext;
use crate::error::Result;

pub struct GraphSqlExtension;

impl GraphSqlExtension {
    pub fn new() -> Self {
        Self
    }
    
    pub fn register_functions(&self, ctx: &mut SessionContext) -> Result<()> {
        todo!("Register all graph SQL functions with DataFusion context")
    }
}

impl Default for GraphSqlExtension {
    fn default() -> Self {
        Self::new()
    }
}