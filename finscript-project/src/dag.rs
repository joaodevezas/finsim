//! Dependency graph (DAG) over a program's top-level variable
//! assignments, plus serialization back to a "DAG file" (`.fic`).

use crate::interpreter::CLASS_T;
use crate::parser::{Expr, Program, Stmt};
use std::collections::HashMap;
use std::fmt::Write as _;

pub type NodeId = usize;

/// The pure semantic operations. Edges are explicit NodeIds, not string lookups.
#[derive(Debug, Clone)]
#[allow(dead_code)] // <--- ADD THIS
pub enum Operation {
    /// A leaf value that becomes an overridable parameter (`--name=value`).
    Param { expr: Expr },
    /// `name = t<TICKER>;` -- fetches external data. 
    Ticker { symbol: String },
    /// Anything else. We retain the AST `expr` so we can render it later,
    /// but the true dependencies are hard-linked via `deps`.
    Derived { expr: Expr, deps: Vec<NodeId> },
    /// `mut` bindings get reassigned, so they act as opaque pass-throughs.
    Mutable { expr: Expr },
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // <--- ADD THIS
pub struct DagNode {
    pub id: NodeId,
    pub name: String,
    pub op: Operation,
}

#[derive(Debug, Clone, Default)]
pub struct Dag {
    pub nodes: Vec<DagNode>,
    /// The Environment (rho): Maps variable names to their exact NodeId
    pub env: HashMap<String, NodeId>,
}

#[derive(Debug, Clone)]
pub struct DagError {
    pub message: String,
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for DagError {}

impl Dag {
    /// Gets the *latest* node assigned to this variable name
    pub fn get(&self, name: &str) -> Option<&DagNode> {
        self.env.get(name).map(|&id| &self.nodes[id])
    }

    // By checking `self.env.values()`, we ensure we only yield the *final* 
    // assignments, gracefully handling variable shadowing.
    
    pub fn parameters(&self) -> impl Iterator<Item = &DagNode> {
        self.env.values()
            .map(|&id| &self.nodes[id])
            .filter(|n| matches!(n.op, Operation::Param { .. }))
    }

    pub fn tickers(&self) -> impl Iterator<Item = &DagNode> {
        self.env.values()
            .map(|&id| &self.nodes[id])
            .filter(|n| matches!(n.op, Operation::Ticker { .. }))
    }

    pub fn mutable_names(&self) -> impl Iterator<Item = &DagNode> {
        self.env.values()
            .map(|&id| &self.nodes[id])
            .filter(|n| matches!(n.op, Operation::Mutable { .. }))
    }

    pub fn derived(&self) -> impl Iterator<Item = &DagNode> {
        self.env.values()
            .map(|&id| &self.nodes[id])
            .filter(|n| matches!(n.op, Operation::Derived { .. }))
    }
}

/// Build a semantic DAG from the top-level assignments in `program`.
pub fn build_dag(program: &Program) -> Result<Dag, DagError> {
    let mut dag = Dag::default();

    for stmt in program {
        if let Stmt::Assign { name, value, mutable } = stmt {
            let op = if *mutable {
                Operation::Mutable { expr: value.clone() }
            } else {
                // Denotational mapping: find the NodeIds this relies on
                let mut deps = Vec::new();
                collect_semantic_deps(value, &dag.env, &mut deps)?;
                classify(value, deps)
            };

            // Create a new node in the graph
            let id = dag.nodes.len();
            dag.nodes.push(DagNode { id, name: name.clone(), op });
            
            // Update the environment so future nodes point to this new ID
            dag.env.insert(name.clone(), id);
        }
    }

    // Notice: NO check_acyclic() call! Because we build the graph sequentially 
    // and only look up variables already in `dag.env`, cycles are impossible.
    
    Ok(dag)
}

fn classify(expr: &Expr, deps: Vec<NodeId>) -> Operation {
    match expr {
        Expr::Subset { base, name } => {
            if let Expr::Var(class_name) = base.as_ref() {
                if class_name == CLASS_T {
                    return Operation::Ticker { symbol: name.clone() };
                }
            }
            Operation::Derived { expr: expr.clone(), deps }
        }
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => Operation::Param { expr: expr.clone() },
        other => Operation::Derived { expr: other.clone(), deps },
    }
}

/// Recursively find dependencies by looking them up in the Environment.
fn collect_semantic_deps(
    expr: &Expr,
    env: &HashMap<String, NodeId>,
    out: &mut Vec<NodeId>,
) -> Result<(), DagError> {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => {}
        Expr::Var(name) => {
            if let Some(&id) = env.get(name) {
                if !out.contains(&id) {
                    out.push(id);
                }
            } else {
                return Err(DagError {
                    message: format!("Semantic error: variable '{}' is referenced before it is defined.", name),
                });
            }
        }
        Expr::Field { base, .. } => collect_semantic_deps(base, env, out)?,
        Expr::Subset { base, name: _ } => {
            if let Expr::Var(class_name) = base.as_ref() {
                if class_name == CLASS_T {
                    return Ok(());
                }
            }
            collect_semantic_deps(base, env, out)?;
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_semantic_deps(lhs, env, out)?;
            collect_semantic_deps(rhs, env, out)?;
        }
        Expr::If { condition, then_branch, else_branch } => {
            collect_semantic_deps(condition, env, out)?;
            collect_semantic_deps(then_branch, env, out)?;
            collect_semantic_deps(else_branch, env, out)?;
        }
        Expr::Call { name: _, args } => {
            for a in args {
                collect_semantic_deps(a, env, out)?;
            }
        }
        Expr::Unsafe { .. } => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Serialization: turn a Program + its Dag back into `.fic` source text.
// ---------------------------------------------------------------------

pub fn render_dag_file(program: &Program, dag: &Dag) -> String {
    let mut out = String::new();

    writeln!(out, "# Auto-generated DAG file -- do not hand-edit unless you mean to.").ok();
    writeln!(out, "# Regenerate it by re-running the original .fi program.").ok();
    
    if dag.parameters().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Parameters (all required; fill in with --name=value):").ok();
        for node in dag.parameters() {
            writeln!(out, "#   --{} (was: {})", node.name, display_expr(&literal_expr(node))).ok();
        }
    }
    
    if dag.tickers().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Hardcoded ticker fetches (not parameterized -- these are baked in):").ok();
        for node in dag.tickers() {
            if let Operation::Ticker { symbol } = &node.op {
                writeln!(out, "#   {} = t<{}>", node.name, symbol).ok();
            }
        }
    }

    if dag.derived().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Derived values (computed from other variables, kept as-is):").ok();
        for node in dag.derived() {
            if let Operation::Derived { expr, .. } = &node.op {
                writeln!(out, "#   {} = {}", node.name, display_expr(expr)).ok();
            }
        }
    }

    if dag.mutable_names().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# `mut` bindings (reassigned over their lifetime, never parameterized):").ok();
        for node in dag.mutable_names() {
            writeln!(out, "#   {}", node.name).ok();
        }
    }
    writeln!(out).ok();

    for stmt in program {
        render_stmt(stmt, dag, 0, &mut out);
    }

    out
}

fn literal_expr(node: &DagNode) -> Expr {
    match &node.op {
        Operation::Param { expr } => expr.clone(),
        _ => Expr::Str(String::new()),
    }
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

fn render_stmt(stmt: &Stmt, dag: &Dag, depth: usize, out: &mut String) {
    let pad = indent(depth);
    match stmt {
        Stmt::Assign { name, value, mutable } => {
            let is_param_slot = depth == 0 && matches!(dag.get(name).map(|n| &n.op), Some(Operation::Param { .. }));
            let kw = if *mutable { "mut " } else { "" };
            
            if is_param_slot {
                writeln!(out, "{pad}{kw}{name} = param();  # was: {name} = {};", display_expr(value)).ok();
            } else {
                writeln!(out, "{pad}{kw}{name} = {};", display_expr(value)).ok();
            }
        }
        Stmt::Print(exprs) => {
            let args: Vec<String> = exprs.iter().map(display_expr).collect();
            writeln!(out, "{pad}print({});", args.join(", ")).ok();
        }
        Stmt::FnDef { name, params, body } => {
            writeln!(out, "{pad}fn {name}({}) {{", params.join(", ")).ok();
            for s in body {
                render_stmt(s, dag, depth + 1, out);
            }
            writeln!(out, "{pad}}}").ok();
        }
        Stmt::While { condition, body } => {
            writeln!(out, "{pad}while {} {{", display_expr(condition)).ok();
            for s in body {
                render_stmt(s, dag, depth + 1, out);
            }
            writeln!(out, "{pad}}}").ok();
        }
        Stmt::Return(expr) => {
            writeln!(out, "{pad}return {};", display_expr(expr)).ok();
        }
    }
}

fn display_expr(expr: &Expr) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Float(n) => n.to_string(),
        Expr::Str(s) => format!("{s:?}"),
        Expr::Var(name) => name.clone(),
        Expr::Field { base, name } => format!("{}.{}", display_expr(base), name),
        Expr::Subset { base, name } => format!("{}<{}>", display_expr(base), name),
        Expr::Binary { op, lhs, rhs } => format!("{} {} {}", display_expr(lhs), op, display_expr(rhs)),
        Expr::If { condition, then_branch, else_branch } => format!(
            "if {} {{ {} }} else {{ {} }}",
            display_expr(condition),
            display_expr(then_branch),
            display_expr(else_branch)
        ),
        Expr::Call { name, args } => {
            let args: Vec<String> = args.iter().map(display_expr).collect();
            format!("{name}({})", args.join(", "))
        }
        Expr::Unsafe { body } => {
            let mut s = String::from("[unsafe] {\n");
            let empty_dag = Dag::default();
            for stmt in body {
                render_stmt(stmt, &empty_dag, 1, &mut s);
            }
            s.push('}');
            s
        }
    }
}