//! Dependency graph (DAG) over a program's top-level variable
//! assignments, plus serialization back to a "DAG file" (`.fic`).
//!
//! The idea:
//! - A `.fi` program is executed as normal, but along the way we also
//!   build a DAG of its top-level assignments (`name = expr;`), tracking
//!   which other variables each one depends on.
//! - That DAG is rendered back out as `.fic` source: same code, same
//!   statement order, *except* that top-level assignments whose value is
//!   a plain literal (`x = 5;`) are rewritten into an unfilled parameter
//!   slot (`x = param();`), with the original literal preserved as a
//!   trailing comment.
//! - `t<TICKER>` fetches are treated as special/hardcoded: they always
//!   come out of the DAG file exactly as they went in, never turned into
//!   a parameter, since they're not "just a value" -- they're a request
//!   for external data tied to a specific ticker.
//! - Running a `.fic` file with `--x=5 --y=2` fills those parameter slots
//!   back in at run time (see `Interpreter::resolve_param` /
//!   `main::parse_cli_args`), without touching the file itself.

use crate::interpreter::CLASS_T;
use crate::parser::{Expr, Program, Stmt};
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Debug, Clone)]
pub enum NodeKind {
    /// `name = t<TICKER>;` -- fetches external data. Never parameterized;
    /// hardcoded into the DAG file exactly as written.
    Ticker { ticker: String },
    /// `name = <int|float|string literal>;` -- a leaf value that becomes
    /// an overridable parameter (`--name=value`) in the DAG file.
    Literal { expr: Expr },
    /// Anything else (an expression referencing other variables, a
    /// function call, etc). Kept verbatim in the DAG file -- not
    /// parameterized, since there's no single "value" to substitute.
    Derived { expr: Expr },
}

#[derive(Debug, Clone)]
pub struct DagNode {
    pub name: String,
    pub kind: NodeKind,
    /// Names of other top-level variables this node's expression reads.
    /// Does NOT include ticker fetches (`t<...>`) -- those are external
    /// data, not a dependency on another node in this DAG.
    pub deps: Vec<String>,
    pub mutable: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Dag {
    pub nodes: Vec<DagNode>,
    index: HashMap<String, usize>,
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
    pub fn get(&self, name: &str) -> Option<&DagNode> {
        self.index.get(name).map(|&i| &self.nodes[i])
    }

    /// Every node whose value is a plain literal -- i.e. every slot that
    /// `render_dag_file` will turn into `--name=value` on the CLI.
    pub fn parameters(&self) -> impl Iterator<Item = &DagNode> {
        self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Literal { .. }))
    }

    /// Every node that's a hardcoded `t<TICKER>` fetch -- these are never
    /// parameterized, only ever listed for reference.
    pub fn tickers(&self) -> impl Iterator<Item = &DagNode> {
        self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Ticker { .. }))
    }

    /// Every node that's a `mut` binding. These are excluded from
    /// parameterization entirely -- see `classify` -- since they're
    /// reassigned over their lifetime rather than holding one fixed value.
    pub fn mutable_names(&self) -> impl Iterator<Item = &DagNode> {
        self.nodes.iter().filter(|n| n.mutable)
    }
}

/// Build a DAG from the top-level assignments in `program`.
///
/// Only top-level `Stmt::Assign` statements become nodes -- assignments
/// inside `fn` bodies, `while` loops, or `[unsafe]` blocks aren't part of
/// this graph, since they don't define variables visible outside their
/// own scope. If the same top-level name is assigned more than once, the
/// last assignment wins (matching the interpreter's own reassignment
/// semantics), and it replaces the earlier node.
pub fn build_dag(program: &Program) -> Result<Dag, DagError> {
    let mut dag = Dag::default();

    for stmt in program {
        if let Stmt::Assign { name, value, mutable } = stmt {
            let deps = collect_deps(value);
            let kind = classify(value, *mutable);
            let node = DagNode { name: name.clone(), kind, deps, mutable: *mutable };

            if let Some(&i) = dag.index.get(name) {
                dag.nodes[i] = node;
            } else {
                dag.index.insert(name.clone(), dag.nodes.len());
                dag.nodes.push(node);
            }
        }
    }

    check_acyclic(&dag)?;
    Ok(dag)
}

fn classify(expr: &Expr, mutable: bool) -> NodeKind {
    // A `mut` binding gets reassigned over its lifetime, so there's no
    // single value to lift out as a CLI parameter -- always treat it as
    // derived (kept verbatim), even if its current value happens to be a
    // plain literal or a ticker fetch.
    if mutable {
        return NodeKind::Derived { expr: expr.clone() };
    }
    match expr {
        Expr::Subset { base, name } => {
            if let Expr::Var(class_name) = base.as_ref() {
                if class_name == CLASS_T {
                    return NodeKind::Ticker { ticker: name.clone() };
                }
            }
            NodeKind::Derived { expr: expr.clone() }
        }
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => NodeKind::Literal { expr: expr.clone() },
        other => NodeKind::Derived { expr: other.clone() },
    }
}

fn collect_deps(expr: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    collect_deps_inner(expr, &mut out);
    out
}

fn collect_deps_inner(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => {}
        Expr::Var(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Expr::Field { base, .. } => collect_deps_inner(base, out),
        Expr::Subset { base, name: _ } => {
            // `t<TICKER>` is an external fetch, not a dependency on
            // another node in this DAG.
            if let Expr::Var(class_name) = base.as_ref() {
                if class_name == CLASS_T {
                    return;
                }
            }
            collect_deps_inner(base, out);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_deps_inner(lhs, out);
            collect_deps_inner(rhs, out);
        }
        Expr::If { condition, then_branch, else_branch } => {
            collect_deps_inner(condition, out);
            collect_deps_inner(then_branch, out);
            collect_deps_inner(else_branch, out);
        }
        Expr::Call { name, args } => {
            // `param()` is a pseudo-call handled specially by the
            // interpreter; it has no dependency of its own beyond an
            // optional literal default, which `collect_deps_inner` below
            // will correctly find nothing in.
            let _ = name;
            for a in args {
                collect_deps_inner(a, out);
            }
        }
        Expr::Unsafe { .. } => {
            // `[unsafe]` blocks run in their own snapshot scope; we don't
            // attempt to trace variable reads through them here.
        }
    }
}

fn check_acyclic(dag: &Dag) -> Result<(), DagError> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Unvisited,
        Visiting,
        Done,
    }

    fn visit(i: usize, dag: &Dag, marks: &mut [Mark], stack: &mut Vec<String>) -> Result<(), DagError> {
        match marks[i] {
            Mark::Done => return Ok(()),
            Mark::Visiting => {
                return Err(DagError {
                    message: format!(
                        "cycle detected in variable dependencies: {} -> {}",
                        stack.join(" -> "),
                        dag.nodes[i].name
                    ),
                });
            }
            Mark::Unvisited => {}
        }

        marks[i] = Mark::Visiting;
        stack.push(dag.nodes[i].name.clone());
        for dep in &dag.nodes[i].deps {
            if let Some(&j) = dag.index.get(dep) {
                visit(j, dag, marks, stack)?;
            }
        }
        stack.pop();
        marks[i] = Mark::Done;
        Ok(())
    }

    let mut marks = vec![Mark::Unvisited; dag.nodes.len()];
    for i in 0..dag.nodes.len() {
        let mut stack = Vec::new();
        visit(i, dag, &mut marks, &mut stack)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Serialization: turn a Program + its Dag back into `.fic` source text.
// ---------------------------------------------------------------------

/// Render `program` as `.fic` DAG-file source.
///
/// Every top-level statement is re-emitted in its original order. The
/// only rewrite is: a top-level `name = <literal>;` becomes
/// `name = param();  # was: name = <literal>;` -- an empty parameter
/// slot. Running the `.fic` file requires filling every such slot from
/// the command line (`--name=value`); the original literal is kept only
/// as a comment, for reference.
///
/// `t<TICKER>` fetches, and anything that isn't a plain literal (derived
/// expressions, function defs, `while` loops, etc.), are emitted
/// unchanged.
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
        writeln!(
            out,
            "# Hardcoded ticker fetches (not parameterized -- these are baked in):"
        )
        .ok();
        for node in dag.tickers() {
            if let NodeKind::Ticker { ticker } = &node.kind {
                writeln!(out, "#   {} = t<{}>", node.name, ticker).ok();
            }
        }
    }

    let derived: Vec<&DagNode> = dag
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Derived { .. }))
        .collect();
    if !derived.is_empty() {
        writeln!(out, "#").ok();
        writeln!(out, "# Derived values (computed from other variables, kept as-is):").ok();
        for node in derived {
            if let NodeKind::Derived { expr } = &node.kind {
                writeln!(out, "#   {} = {}", node.name, display_expr(expr)).ok();
            }
        }
    }

    if dag.mutable_names().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(
            out,
            "# `mut` bindings (reassigned over their lifetime, never parameterized):"
        )
        .ok();
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
    match &node.kind {
        NodeKind::Literal { expr } => expr.clone(),
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
            // Only top-level assignments are ever parameterized -- a
            // node lookup that succeeds at depth > 0 would just be a
            // same-named local shadowing the top-level variable, which
            // we must NOT rewrite.
            let is_param_slot =
                depth == 0 && matches!(dag.get(name).map(|n| &n.kind), Some(NodeKind::Literal { .. }));

            let kw = if *mutable { "mut " } else { "" };
            if is_param_slot {
                writeln!(
                    out,
                    "{pad}{kw}{name} = param();  # was: {name} = {};",
                    display_expr(value)
                )
                .ok();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    fn parse(src: &str) -> Program {
        Parser::new(tokenize(src).unwrap()).parse_program().unwrap()
    }

    #[test]
    fn tickers_are_hardcoded_literals_are_parameterized() {
        let prog = parse(
            r#"
            print("Hello world \n");
            target = t<AAPL>;
            x = 5;
            "#,
        );
        let dag = build_dag(&prog).unwrap();

        let target = dag.get("target").unwrap();
        assert!(matches!(&target.kind, NodeKind::Ticker { ticker } if ticker == "AAPL"));

        let x = dag.get("x").unwrap();
        assert!(matches!(x.kind, NodeKind::Literal { .. }));

        let rendered = render_dag_file(&prog, &dag);
        assert!(rendered.contains("target = t<AAPL>;"));
        assert!(rendered.contains("x = param();"));
        assert!(rendered.contains("# was: x = 5;"));
    }

    #[test]
    fn detects_cycles() {
        // Not reachable through the normal parser/interpreter (which
        // resolve names top-to-bottom), but the DAG builder should still
        // refuse to accept one if it ever sees it.
        let a = Stmt::Assign { name: "a".into(), value: Expr::Var("b".into()), mutable: false };
        let b = Stmt::Assign { name: "b".into(), value: Expr::Var("a".into()), mutable: false };
        let prog: Program = vec![a, b];
        assert!(build_dag(&prog).is_err());
    }

    #[test]
    fn derived_expressions_are_not_parameterized() {
        let prog = parse("x = 5;\ny = x + 1;\n");
        let dag = build_dag(&prog).unwrap();
        let y = dag.get("y").unwrap();
        assert!(matches!(y.kind, NodeKind::Derived { .. }));
        assert_eq!(y.deps, vec!["x".to_string()]);
    }
}