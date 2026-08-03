//! A real dependency graph over a program's top-level variable
//! assignments, built by actually tracing the program once, and
//! serialized back out as a runnable, parameterizable "DAG file" (`.fic`).
//!
//! This is NOT just a syntax rewrite. `build_dag` walks the program in
//! execution order and, for every top-level `name = expr;`, works out
//! whether `expr`'s value:
//!
//!  - is a plain literal -- becomes an overridable `--name=value` slot
//!    (`Operation::Param`);
//!  - is a hardcoded `t<TICKER>` fetch -- always kept live, since it's a
//!    request for fresh external data, not "a value" (`Operation::Ticker`);
//!  - is a `mut` binding -- reassigned over its lifetime, so there's no
//!    single value to freeze (`Operation::Mutable`);
//!  - reduces, via a real symbolic trace, to a constant that will be the
//!    same on every run (`Operation::Fixed`) -- e.g. `2 ^ 10`, or a loop
//!    whose result never touches a param or a ticker;
//!  - reduces to a closed-form, loop-free *formula* in terms of one or
//!    more params (`Operation::Formula`) -- e.g. a loop whose trip count
//!    doesn't depend on any param gets fully unrolled into straight-line
//!    arithmetic;
//!  - or can't be reduced at all (`Operation::Live`) -- it touches a
//!    ticker, calls a user-defined function, or its control flow (a
//!    loop's trip count, or a branch) itself depends on a param. These
//!    are the only nodes that still "do work" (re-run their original
//!    code) when the `.fic` file is executed.
//!
//! Known scope limits (documented rather than silently papered over):
//! user-defined function calls are never traced into -- they're always
//! `Live`. Loop unrolling has a hard iteration cap (`MAX_UNROLL_ITERS`)
//! to avoid hanging the DAG builder on a runaway (but genuinely
//! param-independent) loop.

use crate::interpreter::CLASS_T;
use crate::parser::{BinOp, Expr, Program, Stmt};
use std::collections::HashMap;
use std::fmt::Write as _;

pub type NodeId = usize;

/// A concrete, frozen value: what actually gets written as a literal in
/// the `.fic` file for a `Fixed` node.
#[derive(Debug, Clone)]
pub enum FrozenLit {
    Number(f64),
    Str(String),
}

impl FrozenLit {
    fn to_expr(&self) -> Expr {
        match self {
            FrozenLit::Number(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    Expr::Int(*n as i64)
                } else {
                    Expr::Float(*n)
                }
            }
            FrozenLit::Str(s) => Expr::Str(s.clone()),
        }
    }
}

/// Why a node ended up `Live` (couldn't be frozen) -- kept purely for
/// documentation in the rendered `.fic` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveReason {
    /// Reads a ticker fetch, directly or transitively.
    Ticker,
    /// Calls a user-defined (or native) function -- not traced into.
    FunctionCall,
    /// A loop's trip count depends on a param; can't be unrolled into a
    /// fixed-size formula without knowing the param's value.
    ParamControlledLoop,
    /// An `if` condition depends on a param in a way that couldn't be
    /// folded into a formula (one branch itself was live).
    ParamControlledBranch,
    /// A `mut` binding: reassigned over its lifetime.
    MutableBinding,
    /// Something we lost track of (e.g. a genuine type error the real
    /// interpreter will surface, or division by zero at trace time).
    Undefined,
}

impl std::fmt::Display for LiveReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LiveReason::Ticker => "reads a ticker fetch",
            LiveReason::FunctionCall => "calls a function (not traced)",
            LiveReason::ParamControlledLoop => "a loop's trip count depends on a param",
            LiveReason::ParamControlledBranch => "a branch depends on a param and can't be reduced",
            LiveReason::MutableBinding => "a `mut` binding, reassigned over its lifetime",
            LiveReason::Undefined => "couldn't be reduced to a formula",
        };
        write!(f, "{s}")
    }
}

/// The result of symbolically tracing one expression -- carried both
/// through the tracer's local environment (for locals inside `[unsafe]`
/// blocks) and, for top-level names, derived on demand from their node's
/// `Operation` (see `node_prov`).
#[derive(Debug, Clone)]
enum Prov {
    /// Same value on every run.
    Fixed(FrozenLit),
    /// A pure, loop-free function of one or more params (and/or fixed
    /// constants).
    Formula(Expr),
    /// Can't be reduced -- see `LiveReason`.
    Live(LiveReason),
}

fn formula_expr(p: &Prov) -> Option<Expr> {
    match p {
        Prov::Fixed(lit) => Some(lit.to_expr()),
        Prov::Formula(e) => Some(e.clone()),
        Prov::Live(_) => None,
    }
}

/// The semantic operation a top-level node performs. Unlike a raw AST
/// node, this records *what we learned by tracing*, not just syntax.
#[derive(Debug, Clone)]
pub enum Operation {
    /// A leaf literal -- becomes an overridable `--name=value` parameter.
    Param { default: Expr },
    /// `name = t<TICKER>;` -- external data fetch, always kept live.
    Ticker { symbol: String },
    /// A `mut` binding -- reassigned over its lifetime, never frozen.
    Mutable { expr: Expr },
    /// Frozen to a concrete constant -- cheapest possible re-run.
    Fixed { value: FrozenLit, original: Expr },
    /// Frozen to a closed-form, loop-free expression in terms of
    /// param(s) -- e.g. a fully-unrolled fixed-trip-count loop.
    Formula { expr: Expr, original: Expr },
    /// Can't be reduced -- kept as the original, re-runnable code.
    Live { expr: Expr, reason: LiveReason },
}

#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: NodeId,
    pub name: String,
    pub op: Operation,
    /// Other top-level nodes this one's *original* expression visibly
    /// references (a plain syntactic scan -- not the same thing as "is
    /// tainted live by", which is what the trace itself decides).
    pub deps: Vec<NodeId>,
}

#[derive(Debug, Clone, Default)]
pub struct Dag {
    pub nodes: Vec<DagNode>,
    /// Maps a variable name to the *most recent* node assigned to it,
    /// so reassignment/shadowing resolves the same way the interpreter
    /// itself resolves names.
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
    pub fn get(&self, name: &str) -> Option<&DagNode> {
        self.env.get(name).map(|&id| &self.nodes[id])
    }

    fn latest(&self) -> impl Iterator<Item = &DagNode> {
        self.env.values().map(|&id| &self.nodes[id])
    }

    pub fn parameters(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Param { .. }))
    }

    pub fn tickers(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Ticker { .. }))
    }

    pub fn fixed(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Fixed { .. }))
    }

    pub fn formulas(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Formula { .. }))
    }

    pub fn mutable(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Mutable { .. }))
    }

    pub fn live(&self) -> impl Iterator<Item = &DagNode> {
        self.latest().filter(|n| matches!(n.op, Operation::Live { .. }))
    }

    /// Render the graph as Graphviz DOT, for anyone who wants to actually
    /// *look* at the DAG rather than just read the `.fic` text. Not
    /// wired into the CLI automatically -- available for tooling/debugging.
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        writeln!(out, "digraph fin_dag {{").ok();
        writeln!(out, "  rankdir=LR;").ok();
        for node in &self.nodes {
            let (shape, color) = match &node.op {
                Operation::Param { .. } => ("box", "lightblue"),
                Operation::Ticker { .. } => ("box", "gold"),
                Operation::Mutable { .. } => ("box", "lightgray"),
                Operation::Fixed { .. } => ("ellipse", "palegreen"),
                Operation::Formula { .. } => ("ellipse", "khaki"),
                Operation::Live { .. } => ("ellipse", "lightpink"),
            };
            let label_suffix: String = match &node.op {
                Operation::Param { .. } => " (param)".to_string(),
                Operation::Ticker { symbol } => format!(" (t<{symbol}>)"),
                Operation::Mutable { .. } => " (mut)".to_string(),
                Operation::Fixed { .. } => " (fixed)".to_string(),
                Operation::Formula { .. } => " (formula)".to_string(),
                Operation::Live { reason, .. } => format!(" (live: {reason})"),
            };
            writeln!(
                out,
                "  n{} [label=\"{}{}\" shape={} style=filled fillcolor={}];",
                node.id, node.name, label_suffix, shape, color
            )
            .ok();
        }
        for node in &self.nodes {
            for &dep in &node.deps {
                writeln!(out, "  n{} -> n{};", dep, node.id).ok();
            }
        }
        writeln!(out, "}}").ok();
        out
    }
}

/// Build the DAG by actually tracing `program` once, in execution order.
pub fn build_dag(program: &Program) -> Result<Dag, DagError> {
    let mut dag = Dag::default();

    for stmt in program {
        if let Stmt::Assign { name, value, mutable } = stmt {
            let deps = collect_syntactic_deps(value, &dag.env);

            let op = if *mutable {
                Operation::Mutable { expr: value.clone() }
            } else if let Some(default) = literal_shape(value) {
                Operation::Param { default }
            } else if let Some(symbol) = ticker_shape(value) {
                Operation::Ticker { symbol }
            } else {
                let mut locals: Vec<HashMap<String, Prov>> = vec![HashMap::new()];
                match trace_expr(value, &dag, &mut locals) {
                    Prov::Fixed(v) => Operation::Fixed { value: v, original: value.clone() },
                    Prov::Formula(e) => Operation::Formula { expr: e, original: value.clone() },
                    Prov::Live(reason) => Operation::Live { expr: value.clone(), reason },
                }
            };

            let id = dag.nodes.len();
            dag.nodes.push(DagNode { id, name: name.clone(), op, deps });
            dag.env.insert(name.clone(), id);
        }
    }

    // No cycle check needed: nodes are only ever linked to names already
    // present in `dag.env` at the time they're built, so a cycle is
    // structurally impossible (same reason the interpreter itself never
    // sees forward references).
    Ok(dag)
}

fn literal_shape(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => Some(expr.clone()),
        _ => None,
    }
}

fn ticker_shape(expr: &Expr) -> Option<String> {
    if let Expr::Subset { base, name } = expr {
        if let Expr::Var(class_name) = base.as_ref() {
            if class_name == CLASS_T {
                return Some(name.clone());
            }
        }
    }
    None
}

/// A plain syntactic scan for which top-level names an expression
/// mentions -- used only for `deps` (documentation / `to_dot`), not for
/// the freeze/live decision itself (the trace handles that).
fn collect_syntactic_deps(expr: &Expr, env: &HashMap<String, NodeId>) -> Vec<NodeId> {
    let mut out = Vec::new();
    fn walk(expr: &Expr, env: &HashMap<String, NodeId>, out: &mut Vec<NodeId>) {
        match expr {
            Expr::Int(_) | Expr::Float(_) | Expr::Str(_) => {}
            Expr::Var(name) => {
                if let Some(&id) = env.get(name) {
                    if !out.contains(&id) {
                        out.push(id);
                    }
                }
            }
            Expr::Field { base, .. } => walk(base, env, out),
            Expr::Subset { base, name: _ } => walk(base, env, out),
            Expr::Binary { lhs, rhs, .. } => {
                walk(lhs, env, out);
                walk(rhs, env, out);
            }
            Expr::If { condition, then_branch, else_branch } => {
                walk(condition, env, out);
                walk(then_branch, env, out);
                walk(else_branch, env, out);
            }
            Expr::Call { args, .. } => {
                for a in args {
                    walk(a, env, out);
                }
            }
            // We don't chase reads through `[unsafe]` bodies here; the
            // symbolic tracer (which does look inside) is what actually
            // decides freeze/live, this is just documentation metadata.
            Expr::Unsafe { .. } => {}
        }
    }
    walk(expr, env, &mut out);
    out
}

// ---------------------------------------------------------------------
// The symbolic tracer.
// ---------------------------------------------------------------------

fn node_prov(node: &DagNode) -> Prov {
    match &node.op {
        Operation::Param { .. } => Prov::Formula(Expr::Var(node.name.clone())),
        Operation::Ticker { .. } => Prov::Live(LiveReason::Ticker),
        Operation::Mutable { .. } => Prov::Live(LiveReason::MutableBinding),
        Operation::Fixed { value, .. } => Prov::Fixed(value.clone()),
        Operation::Formula { expr, .. } => Prov::Formula(expr.clone()),
        Operation::Live { reason, .. } => Prov::Live(*reason),
    }
}

fn lookup(dag: &Dag, locals: &[HashMap<String, Prov>], name: &str) -> Prov {
    for scope in locals.iter().rev() {
        if let Some(p) = scope.get(name) {
            return p.clone();
        }
    }
    match dag.get(name) {
        Some(node) => node_prov(node),
        None => Prov::Live(LiveReason::Undefined),
    }
}

fn trace_expr(expr: &Expr, dag: &Dag, locals: &mut Vec<HashMap<String, Prov>>) -> Prov {
    match expr {
        Expr::Int(n) => Prov::Fixed(FrozenLit::Number(*n as f64)),
        Expr::Float(n) => Prov::Fixed(FrozenLit::Number(*n)),
        Expr::Str(s) => Prov::Fixed(FrozenLit::Str(s.clone())),
        Expr::Var(name) => lookup(dag, locals, name),
        Expr::Field { base, .. } => {
            let _ = trace_expr(base, dag, locals);
            Prov::Live(LiveReason::Ticker)
        }
        Expr::Subset { base, name: _ } => {
            let _ = trace_expr(base, dag, locals);
            Prov::Live(LiveReason::Ticker)
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = trace_expr(lhs, dag, locals);
            let r = trace_expr(rhs, dag, locals);
            combine_binary(*op, l, r)
        }
        Expr::If { condition, then_branch, else_branch } => {
            match trace_expr(condition, dag, locals) {
                Prov::Live(reason) => Prov::Live(reason),
                Prov::Fixed(FrozenLit::Number(n)) => {
                    if n != 0.0 {
                        trace_expr(then_branch, dag, locals)
                    } else {
                        trace_expr(else_branch, dag, locals)
                    }
                }
                Prov::Fixed(FrozenLit::Str(_)) => Prov::Live(LiveReason::Undefined),
                Prov::Formula(cond_expr) => {
                    let t = trace_expr(then_branch, dag, locals);
                    let e = trace_expr(else_branch, dag, locals);
                    match (formula_expr(&t), formula_expr(&e)) {
                        (Some(te), Some(ee)) => Prov::Formula(Expr::If {
                            condition: Box::new(cond_expr),
                            then_branch: Box::new(te),
                            else_branch: Box::new(ee),
                        }),
                        _ => Prov::Live(LiveReason::ParamControlledBranch),
                    }
                }
            }
        }
        Expr::Call { .. } => Prov::Live(LiveReason::FunctionCall),
        Expr::Unsafe { body } => trace_unsafe(body, dag, locals),
    }
}

fn combine_binary(op: BinOp, l: Prov, r: Prov) -> Prov {
    if let Prov::Live(reason) = l {
        return Prov::Live(reason);
    }
    if let Prov::Live(reason) = r {
        return Prov::Live(reason);
    }
    if let (Prov::Fixed(FrozenLit::Number(a)), Prov::Fixed(FrozenLit::Number(b))) = (&l, &r) {
        return match eval_binop(op, *a, *b) {
            Some(v) => Prov::Fixed(FrozenLit::Number(v)),
            // e.g. division by zero -- don't guess, let the real run
            // surface the actual error.
            None => Prov::Live(LiveReason::Undefined),
        };
    }
    match (formula_expr(&l), formula_expr(&r)) {
        (Some(le), Some(re)) => Prov::Formula(Expr::Binary { op, lhs: Box::new(le), rhs: Box::new(re) }),
        _ => Prov::Live(LiveReason::Undefined),
    }
}

fn eval_binop(op: BinOp, l: f64, r: f64) -> Option<f64> {
    Some(match op {
        BinOp::Add => l + r,
        BinOp::Sub => l - r,
        BinOp::Mul => l * r,
        BinOp::Div => {
            if r == 0.0 {
                return None;
            }
            l / r
        }
        BinOp::Power => l.powf(r),
        BinOp::Lt => bool_num(l < r),
        BinOp::Gt => bool_num(l > r),
        BinOp::Le => bool_num(l <= r),
        BinOp::Ge => bool_num(l >= r),
        BinOp::Eq => bool_num(l == r),
        BinOp::Ne => bool_num(l != r),
    })
}

fn bool_num(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Cap on how many iterations we'll concretely unroll a fixed-trip-count
/// loop before giving up and falling back to `Live`. Guards against a
/// pathological (but genuinely param-independent) infinite/huge loop
/// hanging the DAG builder.
const MAX_UNROLL_ITERS: usize = 10_000;

fn trace_unsafe(body: &[Stmt], dag: &Dag, locals: &mut Vec<HashMap<String, Prov>>) -> Prov {
    locals.push(HashMap::new());
    let result = trace_body(body, dag, locals).unwrap_or(Prov::Live(LiveReason::Undefined));
    locals.pop();
    result
}

/// Traces a statement list, mirroring `Interpreter::exec_body`'s
/// return-propagation. `Some(prov)` means a `return` was reached (its
/// value's provenance); `None` means execution fell through without one.
fn trace_body(body: &[Stmt], dag: &Dag, locals: &mut Vec<HashMap<String, Prov>>) -> Option<Prov> {
    for stmt in body {
        match stmt {
            Stmt::Return(expr) => return Some(trace_expr(expr, dag, locals)),
            Stmt::Assign { name, value, .. } => {
                let v = trace_expr(value, dag, locals);
                locals.last_mut().expect("unsafe scope always has a frame").insert(name.clone(), v);
            }
            Stmt::While { condition, body: wbody } => {
                if let Some(ret) = trace_while(condition, wbody, dag, locals) {
                    return Some(ret);
                }
            }
            // Purity checking elsewhere guarantees these don't actually
            // appear inside a function/[unsafe] body; if they somehow
            // do, skip rather than crash the DAG builder.
            Stmt::Print(_) | Stmt::FnDef { .. } => {}
        }
    }
    None
}

fn trace_while(
    condition: &Expr,
    body: &[Stmt],
    dag: &Dag,
    locals: &mut Vec<HashMap<String, Prov>>,
) -> Option<Prov> {
    let mut iterations = 0usize;
    loop {
        let truth = match trace_expr(condition, dag, locals) {
            Prov::Fixed(FrozenLit::Number(n)) => n != 0.0,
            // The trip count depends on a param (Formula) or is
            // otherwise unrepresentable (Live): can't safely unroll into
            // a closed form. Per policy, this whole loop -- and whatever
            // it was computing -- stays live.
            Prov::Formula(_) => return Some(Prov::Live(LiveReason::ParamControlledLoop)),
            Prov::Live(reason) => return Some(Prov::Live(reason)),
            Prov::Fixed(FrozenLit::Str(_)) => return Some(Prov::Live(LiveReason::Undefined)),
        };
        if !truth {
            break;
        }

        iterations += 1;
        if iterations > MAX_UNROLL_ITERS {
            return Some(Prov::Live(LiveReason::ParamControlledLoop));
        }

        locals.push(HashMap::new());
        let ret = trace_body(body, dag, locals);
        locals.pop();
        if let Some(v) = ret {
            return Some(v);
        }
    }
    None
}

// ---------------------------------------------------------------------
// Serialization: render a Program + its Dag back into `.fic` source.
// ---------------------------------------------------------------------

/// A comment in this language runs to end-of-line, so any text placed
/// after `#` must not contain a literal newline (an unsafe block's
/// display form does). Collapse it to one line for safe embedding.
fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn render_dag_file(program: &Program, dag: &Dag) -> String {
    let mut out = String::new();

    writeln!(out, "# Auto-generated DAG file -- do not hand-edit unless you mean to.").ok();
    writeln!(out, "# Regenerate it by re-running the original .fi program.").ok();

    if dag.parameters().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Parameters (all required; fill in with --name=value):").ok();
        for node in dag.parameters() {
            if let Operation::Param { default } = &node.op {
                writeln!(out, "#   --{} (was: {})", node.name, oneline(&display_expr(default))).ok();
            }
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

    if dag.fixed().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Frozen constants (computed once; no loop runs again):").ok();
        for node in dag.fixed() {
            if let Operation::Fixed { value, original } = &node.op {
                writeln!(
                    out,
                    "#   {} = {}  (was: {})",
                    node.name,
                    display_expr(&value.to_expr()),
                    oneline(&display_expr(original))
                )
                .ok();
            }
        }
    }

    if dag.formulas().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Unrolled formulas (pure functions of param(s); no loop runs again):").ok();
        for node in dag.formulas() {
            if let Operation::Formula { expr, original } = &node.op {
                writeln!(
                    out,
                    "#   {} = {}  (was: {})",
                    node.name,
                    oneline(&display_expr(expr)),
                    oneline(&display_expr(original))
                )
                .ok();
            }
        }
    }

    if dag.mutable().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# `mut` bindings (reassigned over their lifetime, never parameterized):").ok();
        for node in dag.mutable() {
            if let Operation::Mutable { expr } = &node.op {
                writeln!(out, "#   {} = {}", node.name, oneline(&display_expr(expr))).ok();
            }
        }
    }

    if dag.live().next().is_some() {
        writeln!(out, "#").ok();
        writeln!(out, "# Live nodes (still run their original code every time):").ok();
        for node in dag.live() {
            if let Operation::Live { expr, reason } = &node.op {
                writeln!(out, "#   {} -- {} (was: {})", node.name, reason, oneline(&display_expr(expr))).ok();
            }
        }
    }

    writeln!(out).ok();

    for stmt in program {
        render_stmt(stmt, dag, 0, &mut out);
    }

    out
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

fn render_stmt(stmt: &Stmt, dag: &Dag, depth: usize, out: &mut String) {
    let pad = indent(depth);
    match stmt {
        Stmt::Assign { name, value, mutable } => {
            let kw = if *mutable { "mut " } else { "" };
            let op = if depth == 0 { dag.get(name).map(|n| &n.op) } else { None };

            match op {
                Some(Operation::Param { default }) => {
                    writeln!(
                        out,
                        "{pad}{kw}{name} = param();  # was: {name} = {};",
                        oneline(&display_expr(default))
                    )
                    .ok();
                }
                Some(Operation::Fixed { value, original }) => {
                    writeln!(
                        out,
                        "{pad}{kw}{name} = {};  # frozen from one full run (was: {name} = {};)",
                        display_expr(&value.to_expr()),
                        oneline(&display_expr(original))
                    )
                    .ok();
                }
                Some(Operation::Formula { expr, original }) => {
                    writeln!(
                        out,
                        "{pad}{kw}{name} = {};  # unrolled formula (was: {name} = {};)",
                        oneline(&display_expr(expr)),
                        oneline(&display_expr(original))
                    )
                    .ok();
                }
                // Ticker, Mutable, Live, or not a tracked top-level node
                // (e.g. a shadowing local inside a fn/while/unsafe body)
                // -- emit exactly as written.
                _ => {
                    writeln!(out, "{pad}{kw}{name} = {};", display_expr(value)).ok();
                }
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
        assert!(matches!(&target.op, Operation::Ticker { symbol } if symbol == "AAPL"));

        let x = dag.get("x").unwrap();
        assert!(matches!(x.op, Operation::Param { .. }));

        let rendered = render_dag_file(&prog, &dag);
        assert!(rendered.contains("target = t<AAPL>;"));
        assert!(rendered.contains("x = param();"));
        assert!(rendered.contains("# was: x = 5;"));
    }

    #[test]
    fn pure_constant_expression_freezes() {
        let prog = parse("z = 2 ^ 10;\n");
        let dag = build_dag(&prog).unwrap();
        let z = dag.get("z").unwrap();
        match &z.op {
            Operation::Fixed { value: FrozenLit::Number(n), .. } => assert_eq!(*n, 1024.0),
            other => panic!("expected Fixed, got {other:?}"),
        }
    }

    #[test]
    fn derived_arithmetic_becomes_a_formula() {
        let prog = parse("x = 5;\ny = x + 1;\n");
        let dag = build_dag(&prog).unwrap();
        let y = dag.get("y").unwrap();
        assert!(matches!(y.op, Operation::Formula { .. }));
    }

    #[test]
    fn fixed_trip_count_loop_unrolls_into_a_formula() {
        let prog = parse(
            r#"
            x = 5;
            y = [unsafe] {
                mut total = 0;
                mut i = 0;
                while i < 3 {
                    total = total + x;
                    i = i + 1;
                }
                return total;
            };
            "#,
        );
        let dag = build_dag(&prog).unwrap();
        let y = dag.get("y").unwrap();
        match &y.op {
            Operation::Formula { expr, .. } => {
                let rendered = display_expr(expr);
                assert!(!rendered.contains("while"), "formula should contain no loop: {rendered}");
                assert!(rendered.contains('x'), "formula should still reference the param: {rendered}");
            }
            other => panic!("expected Formula, got {other:?}"),
        }
    }

    #[test]
    fn param_controlled_trip_count_stays_live() {
        let prog = parse(
            r#"
            x = 5;
            y = [unsafe] {
                mut total = 0;
                mut i = 0;
                while i < x {
                    total = total + 1;
                    i = i + 1;
                }
                return total;
            };
            "#,
        );
        let dag = build_dag(&prog).unwrap();
        let y = dag.get("y").unwrap();
        assert!(matches!(
            &y.op,
            Operation::Live { reason: LiveReason::ParamControlledLoop, .. }
        ));

        let rendered = render_dag_file(&prog, &dag);
        assert!(rendered.contains("while i < x"), "live loop should be kept verbatim:\n{rendered}");
    }

    #[test]
    fn ticker_derived_values_stay_live() {
        let prog = parse(
            r#"
            target = t<AAPL>;
            price = target.price;
            doubled = price + price;
            "#,
        );
        let dag = build_dag(&prog).unwrap();
        assert!(matches!(dag.get("price").unwrap().op, Operation::Live { reason: LiveReason::Ticker, .. }));
        assert!(matches!(dag.get("doubled").unwrap().op, Operation::Live { reason: LiveReason::Ticker, .. }));
    }

    #[test]
    fn function_calls_stay_live() {
        let prog = parse("x = 5;\ny = some_fn(x);\n");
        let dag = build_dag(&prog).unwrap();
        assert!(matches!(
            dag.get("y").unwrap().op,
            Operation::Live { reason: LiveReason::FunctionCall, .. }
        ));
    }

    #[test]
    fn reassignment_creates_a_new_node_and_env_points_to_latest() {
        let prog = parse("x = 5;\nx = 6;\n");
        let dag = build_dag(&prog).unwrap();
        assert_eq!(dag.nodes.len(), 2);
        assert_eq!(dag.get("x").unwrap().id, 1);
    }
}