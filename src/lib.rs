pub mod graph;
pub mod sql;
pub mod algorithms;
pub mod error;

pub use graph::{ArrowGraph, GraphIndexes};
pub use sql::GraphSqlExtension;
pub use algorithms::{GraphAlgorithm, AlgorithmParams};
pub use error::{GraphError, Result};

pub mod prelude {
    pub use crate::graph::{ArrowGraph, GraphIndexes};
    pub use crate::sql::GraphSqlExtension;
    pub use crate::algorithms::{GraphAlgorithm, AlgorithmParams};
    pub use crate::error::{GraphError, Result};
}