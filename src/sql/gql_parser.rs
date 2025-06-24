use std::collections::HashMap;
use crate::error::{GraphError, Result};

/// GQL pattern parser for basic graph pattern matching
/// Supports patterns like: MATCH (a)-[r]->(b) WHERE a.type = 'user'
#[derive(Debug, Clone)]
pub struct GqlParser;

/// Represents a node in a GQL pattern
#[derive(Debug, Clone, PartialEq)]
pub struct GqlNode {
    pub variable: String,
    pub label: Option<String>,
    pub properties: HashMap<String, String>,
}

/// Represents an edge in a GQL pattern
#[derive(Debug, Clone, PartialEq)]
pub struct GqlEdge {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub direction: EdgeDirection,
    pub properties: HashMap<String, String>,
}

/// Edge direction in GQL patterns
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeDirection {
    Outgoing,  // -->
    Incoming,  // <--
    Undirected, // --
}

/// Represents a complete GQL pattern
#[derive(Debug, Clone)]
pub struct GqlPattern {
    pub nodes: Vec<GqlNode>,
    pub edges: Vec<GqlEdge>,
    pub where_conditions: Vec<WhereCondition>,
}

/// WHERE clause conditions
#[derive(Debug, Clone, PartialEq)]
pub struct WhereCondition {
    pub variable: String,
    pub property: String,
    pub operator: ComparisonOperator,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
}

impl GqlParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a GQL MATCH pattern
    /// Example: "MATCH (a:User)-[r:FOLLOWS]->(b:User) WHERE a.name = 'Alice'"
    pub fn parse_match_pattern(&self, pattern: &str) -> Result<GqlPattern> {
        let pattern = pattern.trim();
        
        // Split MATCH and WHERE clauses
        let (match_part, where_part) = if pattern.to_uppercase().contains("WHERE") {
            let parts: Vec<&str> = pattern.splitn(2, "WHERE").collect();
            if parts.len() == 2 {
                (parts[0].trim(), Some(parts[1].trim()))
            } else {
                (pattern, None)
            }
        } else {
            (pattern, None)
        };

        // Remove MATCH keyword
        let match_part = if match_part.to_uppercase().starts_with("MATCH") {
            match_part[5..].trim()
        } else {
            match_part
        };

        // Parse the graph pattern
        let (nodes, edges) = self.parse_graph_pattern(match_part)?;
        
        // Parse WHERE conditions
        let where_conditions = if let Some(where_str) = where_part {
            self.parse_where_conditions(where_str)?
        } else {
            Vec::new()
        };

        Ok(GqlPattern {
            nodes,
            edges,
            where_conditions,
        })
    }

    /// Parse the graph pattern part: (a:User)-[r:FOLLOWS]->(b:User)
    fn parse_graph_pattern(&self, pattern: &str) -> Result<(Vec<GqlNode>, Vec<GqlEdge>)> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Simple regex-like parsing for basic patterns
        // This is a simplified implementation - a full parser would use a proper grammar
        
        let pattern = pattern.trim();
        
        // Example pattern: (a:User)-[r:FOLLOWS]->(b:User)
        // We'll parse this step by step
        
        if pattern.starts_with('(') {
            // Find node patterns and edge patterns
            let chars: Vec<char> = pattern.chars().collect();
            let mut i = 0;
            
            while i < chars.len() {
                if chars[i] == '(' {
                    // Parse node
                    let (node, end_pos) = self.parse_node_pattern(&chars, i)?;
                    nodes.push(node);
                    i = end_pos + 1;
                } else if chars[i] == '-' || chars[i] == '<' {
                    // Parse edge
                    let (edge, end_pos) = self.parse_edge_pattern(&chars, i)?;
                    edges.push(edge);
                    i = end_pos + 1;
                } else {
                    i += 1;
                }
            }
        } else {
            return Err(GraphError::sql_parsing("Invalid GQL pattern: must start with node"));
        }

        Ok((nodes, edges))
    }

    /// Parse a node pattern: (a:User {name: 'Alice'})
    fn parse_node_pattern(&self, chars: &[char], start: usize) -> Result<(GqlNode, usize)> {
        if start >= chars.len() || chars[start] != '(' {
            return Err(GraphError::sql_parsing("Node pattern must start with ("));
        }

        let mut i = start + 1;
        let mut content = String::new();
        let mut paren_count = 1;

        // Find the matching closing parenthesis
        while i < chars.len() && paren_count > 0 {
            if chars[i] == '(' {
                paren_count += 1;
            } else if chars[i] == ')' {
                paren_count -= 1;
            }
            
            if paren_count > 0 {
                content.push(chars[i]);
            }
            i += 1;
        }

        if paren_count > 0 {
            return Err(GraphError::sql_parsing("Unclosed node pattern"));
        }

        // Parse node content: variable:label {properties}
        let (variable, label, properties) = self.parse_node_content(&content)?;

        Ok((GqlNode {
            variable,
            label,
            properties,
        }, i - 1))
    }

    /// Parse node content: a:User {name: 'Alice'}
    fn parse_node_content(&self, content: &str) -> Result<(String, Option<String>, HashMap<String, String>)> {
        let content = content.trim();
        
        // Split by { to separate variable:label from properties
        let (var_label_part, properties_part) = if content.contains('{') {
            let parts: Vec<&str> = content.splitn(2, '{').collect();
            (parts[0].trim(), Some(parts[1].trim_end_matches('}').trim()))
        } else {
            (content, None)
        };

        // Parse variable and label
        let (variable, label) = if var_label_part.contains(':') {
            let parts: Vec<&str> = var_label_part.splitn(2, ':').collect();
            (parts[0].trim().to_string(), Some(parts[1].trim().to_string()))
        } else {
            (var_label_part.to_string(), None)
        };

        // Parse properties (simplified)
        let properties = if let Some(props_str) = properties_part {
            self.parse_properties(props_str)?
        } else {
            HashMap::new()
        };

        Ok((variable, label, properties))
    }

    /// Parse edge pattern: -[r:FOLLOWS]->
    fn parse_edge_pattern(&self, chars: &[char], start: usize) -> Result<(GqlEdge, usize)> {
        let mut i = start;
        let mut direction = EdgeDirection::Undirected;
        let mut edge_content = String::new();

        // Determine direction
        if i < chars.len() && chars[i] == '<' {
            direction = EdgeDirection::Incoming;
            i += 1;
        }

        // Skip dashes
        while i < chars.len() && chars[i] == '-' {
            i += 1;
        }

        // Parse edge content if present
        if i < chars.len() && chars[i] == '[' {
            i += 1; // Skip [
            while i < chars.len() && chars[i] != ']' {
                edge_content.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                i += 1; // Skip ]
            }
        }

        // Skip more dashes
        while i < chars.len() && chars[i] == '-' {
            i += 1;
        }

        // Check for outgoing direction
        if i < chars.len() && chars[i] == '>' {
            if direction == EdgeDirection::Incoming {
                return Err(GraphError::sql_parsing("Invalid edge direction: cannot be both incoming and outgoing"));
            }
            direction = EdgeDirection::Outgoing;
            i += 1;
        }

        // Parse edge content
        let (variable, label, properties) = if edge_content.is_empty() {
            (None, None, HashMap::new())
        } else {
            let (var, lab, props) = self.parse_node_content(&edge_content)?;
            (Some(var), lab, props)
        };

        Ok((GqlEdge {
            variable,
            label,
            direction,
            properties,
        }, i - 1))
    }

    /// Parse WHERE conditions: a.name = 'Alice' AND b.age > 25
    fn parse_where_conditions(&self, where_str: &str) -> Result<Vec<WhereCondition>> {
        let mut conditions = Vec::new();
        
        // Split by AND/OR (simplified - just AND for now)
        let condition_parts: Vec<&str> = where_str.split(" AND ").collect();
        
        for condition_str in condition_parts {
            let condition = self.parse_single_condition(condition_str.trim())?;
            conditions.push(condition);
        }

        Ok(conditions)
    }

    /// Parse a single condition: a.name = 'Alice'
    fn parse_single_condition(&self, condition: &str) -> Result<WhereCondition> {
        // Find the operator
        let operators = [">=", "<=", "!=", "=", ">", "<"];
        
        for op_str in &operators {
            if condition.contains(op_str) {
                let parts: Vec<&str> = condition.splitn(2, op_str).collect();
                if parts.len() == 2 {
                    let left = parts[0].trim();
                    let right = parts[1].trim().trim_matches('\'').trim_matches('"');
                    
                    // Parse left side (variable.property)
                    if let Some(dot_pos) = left.find('.') {
                        let variable = left[..dot_pos].to_string();
                        let property = left[dot_pos + 1..].to_string();
                        
                        let operator = match *op_str {
                            "=" => ComparisonOperator::Equal,
                            "!=" => ComparisonOperator::NotEqual,
                            ">" => ComparisonOperator::GreaterThan,
                            "<" => ComparisonOperator::LessThan,
                            ">=" => ComparisonOperator::GreaterThanOrEqual,
                            "<=" => ComparisonOperator::LessThanOrEqual,
                            _ => return Err(GraphError::sql_parsing(&format!("Unknown operator: {}", op_str))),
                        };
                        
                        return Ok(WhereCondition {
                            variable,
                            property,
                            operator,
                            value: right.to_string(),
                        });
                    }
                }
            }
        }

        Err(GraphError::sql_parsing(&format!("Invalid WHERE condition: {}", condition)))
    }

    /// Parse properties: name: 'Alice', age: 25
    fn parse_properties(&self, props_str: &str) -> Result<HashMap<String, String>> {
        let mut properties = HashMap::new();
        
        // Split by comma
        let prop_parts: Vec<&str> = props_str.split(',').collect();
        
        for prop_str in prop_parts {
            let prop_str = prop_str.trim();
            if prop_str.contains(':') {
                let parts: Vec<&str> = prop_str.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim().to_string();
                    let value = parts[1].trim().trim_matches('\'').trim_matches('"').to_string();
                    properties.insert(key, value);
                }
            }
        }

        Ok(properties)
    }
}

impl Default for GqlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_node_pattern() {
        let parser = GqlParser::new();
        let pattern = "MATCH (a:User)";
        let result = parser.parse_match_pattern(pattern).unwrap();
        
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].variable, "a");
        assert_eq!(result.nodes[0].label, Some("User".to_string()));
        assert!(result.edges.is_empty());
        assert!(result.where_conditions.is_empty());
    }

    #[test]
    fn test_parse_simple_edge_pattern() {
        let parser = GqlParser::new();
        let pattern = "MATCH (a)-[r]->(b)";
        let result = parser.parse_match_pattern(pattern).unwrap();
        
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].variable, Some("r".to_string()));
        assert_eq!(result.edges[0].direction, EdgeDirection::Outgoing);
    }

    #[test]
    fn test_parse_with_where_condition() {
        let parser = GqlParser::new();
        let pattern = "MATCH (a:User) WHERE a.name = 'Alice'";
        let result = parser.parse_match_pattern(pattern).unwrap();
        
        assert_eq!(result.where_conditions.len(), 1);
        assert_eq!(result.where_conditions[0].variable, "a");
        assert_eq!(result.where_conditions[0].property, "name");
        assert_eq!(result.where_conditions[0].value, "Alice");
        assert_eq!(result.where_conditions[0].operator, ComparisonOperator::Equal);
    }

    #[test]
    fn test_parse_complex_pattern() {
        let parser = GqlParser::new();
        let pattern = "MATCH (a:User)-[r:FOLLOWS]->(b:User) WHERE a.name = 'Alice' AND b.age > 25";
        let result = parser.parse_match_pattern(pattern).unwrap();
        
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.where_conditions.len(), 2);
        
        // Check nodes
        assert_eq!(result.nodes[0].variable, "a");
        assert_eq!(result.nodes[0].label, Some("User".to_string()));
        assert_eq!(result.nodes[1].variable, "b");
        assert_eq!(result.nodes[1].label, Some("User".to_string()));
        
        // Check edge
        assert_eq!(result.edges[0].variable, Some("r".to_string()));
        assert_eq!(result.edges[0].label, Some("FOLLOWS".to_string()));
        assert_eq!(result.edges[0].direction, EdgeDirection::Outgoing);
    }

    #[test]
    fn test_parse_invalid_pattern() {
        let parser = GqlParser::new();
        let pattern = "INVALID PATTERN";
        let result = parser.parse_match_pattern(pattern);
        assert!(result.is_err());
    }
}