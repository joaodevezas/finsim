//! Interpreter: walks the AST from `parser.rs` and actually runs the
//! program.
//!
//! The one non-obvious piece of runtime behavior is what happens with `t`:
//!
//! ```text
//! company = t<APPL>;
//! ```
//!
//! `t` is a reserved "class" name. When the interpreter evaluates
//! `t<APPL>`, it does NOT look up a variable called `t` -- instead it:
//!
//!   1. shells out to a Python script (`generate_data()`'s job, for now
//!      just random test data) which writes `data/APPL.json`,
//!   2. reads that JSON back in as a `Value::Set`,
//!   3. binds it to BOTH `company` (the name on the left of `=`) AND
//!      `APPL` (the ticker itself) in the environment.
//!
//! That's why later lines can say `APPL.earnings <ttm>` directly, without
//! ever mentioning `company` again -- `APPL` became a first-class variable
//! the moment it was constructed. `.field` and `<subset>` are otherwise
//! the same operation (look up a key in a JSON object); see `subset_lookup`.

use crate::parser::{BinOp, Expr, Program, Stmt};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The name that triggers ticker construction. Reserved -- see module docs.
const CLASS_T: &str = "t";

/// A stored `fn` definition: parameters plus a body, whose last statement
/// is guaranteed by the parser to be a `Return`.
#[derive(Debug, Clone)]
struct FnDecl {
    params: Vec<String>,
    body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Str(String),
    /// A JSON object: a "set" of named "subsets" (which may themselves be
    /// sets, e.g. `APPL` -> `{"earnings": {"last": .., "ttm": ..}, ...}`).
    Set(Json),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{n}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Set(json) => {
                let pretty = serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string());
                write!(f, "{pretty}")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub value: Value,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime error: {}", self.message)
    }
}
impl std::error::Error for RuntimeError {}

/// A built-in function implemented in Rust rather than in the DSL itself
/// (see `src/stdlib/`). It receives its arguments already evaluated --
/// exactly like a user-defined `fn` receives its parameters -- and has no
/// access to `&mut self`, so it's automatically as "pure" as the
/// `check_purity` pass forces user-defined functions to be: its only
/// possible input is `args`, its only possible output is its return value.
pub type NativeFn = fn(&[Value]) -> Result<Value, RuntimeError>;

#[derive(Debug)]
pub struct Interpreter {
    // True lexical block scopes, searching from the end (innermost) backwards.
    env: Vec<HashMap<String, Binding>>,
    functions: HashMap<String, FnDecl>,
    /// Prewritten Rust functions (see `src/stdlib/`), registered under a
    /// name and callable from the DSL exactly like a user `fn` -- e.g.
    /// `dcf(...)`. Checked before `functions` in `call_function`, so a
    /// native registration always wins if a name collides.
    #[allow(clippy::type_complexity)]
    natives: HashMap<String, NativeFn>,
    /// Where generated `<TICKER>.json` files are written/read.
    data_dir: PathBuf,
    /// Path to the python script that (for now) invents random data.
    generator_script: PathBuf,
    /// Which python executable to invoke. `python3` on most systems.
    python_bin: String,
    /// Tickers already fetched by a prefetch pass (see `collect_tickers`
    /// + `main.rs`) *this run*, so `construct_ticker` knows it's safe to
    /// skip shelling out again. Deliberately NOT the same thing as "does
    /// the file exist on disk" -- a leftover file from a previous,
    /// unrelated run would otherwise get reused silently forever with no
    /// freshness check at all. Cross-run caching (with an actual TTL) is
    /// a separate, not-yet-built feature; this only ever short-circuits
    /// work this same process already did moments ago.
    prefetched: std::collections::HashSet<String>,
    in_unsafe: usize,
}

#[derive(Debug, Clone)]
enum Flow {
    Normal,
    Return(Value),
}

/// Walks a parsed `Program` once and returns every distinct ticker name
/// referenced anywhere via `t<TICKER>`, without running anything.
///
/// This is possible at all because ticker names are always static text in
/// this language -- there's no string interpolation and no way to build a
/// ticker name at runtime -- so the complete set of tickers a script will
/// ever touch is knowable just from its AST. `main.rs` uses this to fetch
/// everything concurrently *before* `Interpreter::run` starts, instead of
/// discovering (and fetching) tickers one at a time as execution reaches
/// each `t<TICKER>` expression.
///
/// Deliberately does NOT recurse into `FnDef` bodies: `check_purity`
/// already guarantees a function body can never contain `t<...>` at all
/// (see `check_expr_purity`'s `Subset` arm), so there's nothing to find
/// in there -- skipping them is just avoiding pointless work, not
/// missing anything.
pub fn collect_tickers(program: &Program) -> std::collections::HashSet<String> {
    let mut tickers = std::collections::HashSet::new();
    for stmt in program {
        collect_tickers_stmt(stmt, &mut tickers);
    }
    tickers
}

fn collect_tickers_stmt(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
    match stmt {
        Stmt::Assign { value, .. } => collect_tickers_expr(value, out),
        Stmt::Print(exprs) => {
            for e in exprs {
                collect_tickers_expr(e, out);
            }
        }
        Stmt::Return(expr) => collect_tickers_expr(expr, out),
        // A `fn` body is guaranteed ticker-free by `check_purity` at
        // definition time -- nothing to collect inside one.
        Stmt::FnDef { .. } => {}
        Stmt::While { condition, body } => {
            collect_tickers_expr(condition, out);
            for s in body {
                collect_tickers_stmt(s, out);
            }
        }
    }
}

fn collect_tickers_expr(expr: &Expr, out: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Var(_) => {}
        Expr::Field { base, .. } => collect_tickers_expr(base, out),
        Expr::Subset { base, name } => {
            if let Expr::Var(class_name) = base.as_ref() {
                if class_name == CLASS_T {
                    out.insert(name.clone());
                    return;
                }
            }
            collect_tickers_expr(base, out);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_tickers_expr(lhs, out);
            collect_tickers_expr(rhs, out);
        }
        Expr::If { condition, then_branch, else_branch } => {
            collect_tickers_expr(condition, out);
            collect_tickers_expr(then_branch, out);
            collect_tickers_expr(else_branch, out);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                collect_tickers_expr(arg, out);
            }
        }
        Expr::Unsafe { body } => {
            for s in body {
                collect_tickers_stmt(s, out);
            }
        }
    }
}

impl Interpreter {
    pub fn new(data_dir: impl Into<PathBuf>, generator_script: impl Into<PathBuf>) -> Self {
        Interpreter {
            env: vec![HashMap::new()],
            functions: HashMap::new(),
            natives: HashMap::new(),
            data_dir: data_dir.into(),
            generator_script: generator_script.into(),
            python_bin: "python3".to_string(),
            prefetched: std::collections::HashSet::new(),
            in_unsafe: 0,
        }
    }

    fn env_get(&self, name: &str) -> Option<&Binding> {
        for scope in self.env.iter().rev() {
            if let Some(b) = scope.get(name) {
                return Some(b);
            }
        }
        None
    }

    fn env_get_mut(&mut self, name: &str) -> Option<&mut Binding> {
        for scope in self.env.iter_mut().rev() {
            if let Some(b) = scope.get_mut(name) {
                return Some(b);
            }
        }
        None
    }

    fn env_insert(&mut self, name: String, binding: Binding) {
        self.env.last_mut().unwrap().insert(name, binding);
    }

    // Add this attribute right above the function signature!
    #[allow(dead_code)]
    pub fn with_python_bin(mut self, bin: impl Into<String>) -> Self {
        self.python_bin = bin.into();
        self
    }

    /// Records that `ticker`'s JSON was already fetched by a prefetch
    /// pass this run, so `construct_ticker` can skip shelling out again
    /// when `t<TICKER>` is actually evaluated. Call this once per
    /// successfully-prefetched ticker, before `run`.
    pub fn mark_prefetched(&mut self, ticker: impl Into<String>) {
        self.prefetched.insert(ticker.into());
    }

    pub fn data_dir(&self) -> &Path { &self.data_dir }
    pub fn python_bin(&self) -> &str { &self.python_bin }

    /// Registers a prewritten Rust function under `name`, so DSL source
    /// can call it exactly like a user-defined `fn` -- e.g.
    /// `register_native("dcf", stdlib::dcf::dcf)` makes `dcf(...)` work in
    /// the script. Intended to be called once, right after `Interpreter::new`,
    /// before `run`. See `src/stdlib/mod.rs::register_all` for the actual
    /// list of what gets registered.
    pub fn register_native(&mut self, name: impl Into<String>, f: NativeFn) {
        self.natives.insert(name.into(), f);
    }

    pub fn run(&mut self, program: &Program) -> Result<(), RuntimeError> {
        for stmt in program {
            self.exec_stmt(stmt)?;
        }
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<(), RuntimeError> {
        match self.exec_stmt_flow(stmt)? {
            Flow::Normal => Ok(()),
            Flow::Return(_) => Err(RuntimeError::new("`return` used outside of a function or [unsafe] block")),
        }
    }

    fn exec_stmt_flow(&mut self, stmt: &Stmt) -> Result<Flow, RuntimeError> {
        match stmt {
            Stmt::Assign { name, value, mutable } => {
                let v = self.eval(value)?;      

                if *mutable {
                    if self.in_unsafe == 0 {
                        return Err(RuntimeError::new("`mut` is only allowed inside `[unsafe]` blocks"));
                    }
                    // unconditionally scope to the current block, hiding outer references completely!
                    self.env_insert(name.clone(), Binding { value: v, mutable: true });
                    return Ok(Flow::Normal);
                }

                // ---------------------------------------------------------
                // FIX: Copy `in_unsafe` BEFORE taking the mutable borrow!
                // ---------------------------------------------------------
                let current_in_unsafe = self.in_unsafe; 

                if let Some(existing) = self.env_get_mut(name) {
                    if existing.mutable {
                        if current_in_unsafe == 0 { 
                            return Err(RuntimeError::new("reassignment is only allowed inside `[unsafe]` blocks"));
                        }
                        existing.value = v;
                    } else {
                        return Err(RuntimeError::new(format!("Error: Variable '{}' is immutable and cannot be changed.", name)));
                    }
                } else {
                    self.env_insert(name.clone(), Binding { value: v, mutable: false }); 
                }
                Ok(Flow::Normal)
            }

            Stmt::While { condition, body } => {
                if self.in_unsafe == 0 {
                    return Err(RuntimeError::new("`while` is only allowed inside `[unsafe]` blocks"));
                }
                loop {
                    let cond_val = self.eval(condition)?;
                    let is_true = match cond_val {
                        Value::Number(n) => n != 0.0,
                        other => return Err(RuntimeError::new(format!("condition in `while` must be a number, found {other}"))),
                    };
                    if !is_true { break; }

                    // Each iteration executes within its own block scope!
                    self.env.push(HashMap::new());
                    let flow = self.exec_body(body);
                    self.env.pop();

                    let flow = flow?;
                    if let Flow::Return(v) = flow {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal)
            }

            Stmt::Print(exprs) => {
                let mut out = String::new();
                for e in exprs {
                    let v = self.eval(e)?;
                    out.push_str(&v.to_string());
                }
                print!("{out}");
                use std::io::Write;
                std::io::stdout().flush().ok();
                Ok(Flow::Normal)
            }

            Stmt::FnDef { name, params, body } => {
                self.check_purity(body)?;
                self.functions.insert(name.clone(), FnDecl { params: params.clone(), body: body.clone() });
                Ok(Flow::Normal)
            }

            Stmt::Return(expr) => {
                let v = self.eval(expr)?;
                Ok(Flow::Return(v))
            }
        }
    }

    fn exec_body(&mut self, body: &[Stmt]) -> Result<Flow, RuntimeError> {
        for stmt in body {
            let flow = self.exec_stmt_flow(stmt)?;
            if matches!(flow, Flow::Return(_)) {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }

    fn eval(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Int(n) => Ok(Value::Number(*n as f64)),
            Expr::Float(n) => Ok(Value::Number(*n)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),

            Expr::Var(name) => self
                .env_get(name)
                .map(|b| b.value.clone())
                .ok_or_else(|| RuntimeError::new(format!("undefined variable `{name}`"))),

            Expr::Field { base, name } => {
                let base_val = self.eval(base)?;
                Self::subset_lookup(&base_val, name)
            }

            Expr::Subset { base, name } => {
                if let Expr::Var(class_name) = base.as_ref() {
                    if class_name == CLASS_T {
                        return self.construct_ticker(name);
                    }
                }
                let base_val = self.eval(base)?;
                Self::subset_lookup(&base_val, name)
            }

            Expr::Binary { op, lhs, rhs } => {
                let l = Self::as_number(&self.eval(lhs)?)?;
                let r = Self::as_number(&self.eval(rhs)?)?;
                let result = match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => {
                        if r == 0.0 { return Err(RuntimeError::new("division by zero")); }
                        l / r
                    }
                    BinOp::Power => l.powf(r), 
                    BinOp::Lt => if l < r { 1.0 } else { 0.0 },
                    BinOp::Gt => if l > r { 1.0 } else { 0.0 },
                    BinOp::Le => if l <= r { 1.0 } else { 0.0 },
                    BinOp::Ge => if l >= r { 1.0 } else { 0.0 },
                    BinOp::Eq => if l == r { 1.0 } else { 0.0 },
                    BinOp::Ne => if l != r { 1.0 } else { 0.0 },
                };
                Ok(Value::Number(result))
            }

            Expr::If { condition, then_branch, else_branch } => {
                let cond_val = self.eval(condition)?;
                let is_true = match cond_val {
                    Value::Number(n) => n != 0.0,
                    other => return Err(RuntimeError::new(format!("condition in `if` must be a number, found {other}"))),
                };
                if is_true {
                    self.eval(then_branch)
                } else {
                    self.eval(else_branch)
                }
            }

            Expr::Call { name, args } => self.call_function(name, args),

            Expr::Unsafe { body } => {
                self.in_unsafe += 1;
                let caller_env = self.env.clone(); // Full read-only snapshot
                
                self.env.push(HashMap::new()); // Give the unsafe block an explicit scope too
                let result = self.exec_body(body);
                
                self.env = caller_env; // Unconditionally restored
                self.in_unsafe -= 1;

                match result? {
                    Flow::Return(v) => Ok(v),
                    Flow::Normal => Err(RuntimeError::new("[unsafe] block must end with return")),
                }
            }
        }
    }

    fn as_number(v: &Value) -> Result<f64, RuntimeError> {
        match v {
            Value::Number(n) => Ok(*n),
            other => Err(RuntimeError::new(format!("expected a number, found {other}"))),
        }
    }

    /// Calls a user-defined function. This is what makes the function
    /// "pure": the body runs in a brand-new `HashMap` containing ONLY its
    /// parameters, swapped in for the duration of the call and then
    /// swapped back out. The function has no reference to the caller's
    /// variables at all -- not read-only access, no access. Its only
    /// possible inputs are its arguments, and (because `check_purity`
    /// forbids `print` and `t<...>` in its body) its only possible output
    /// is its return value. Same inputs in, same output out, always.
    fn call_function(&mut self, name: &str, args: &[Expr]) -> Result<Value, RuntimeError> {
        // Native (Rust-implemented) functions are checked first, so a
        // registration like `dcf` always wins over any same-named user
        // `fn` -- and, since a fn pointer is `Copy`, this doesn't need to
        // hold a borrow of `self.natives` across the argument evaluation
        // below (which needs `&mut self`).
        if let Some(native) = self.natives.get(name).copied() {
            let mut arg_values = Vec::with_capacity(args.len());
            for arg in args {
                arg_values.push(self.eval(arg)?);
            }
            return native(&arg_values);
        }

        let decl = self
            .functions
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::new(format!("undefined function `{name}`")))?;

        if args.len() != decl.params.len() {
            return Err(RuntimeError::new(format!(
                "function `{name}` expects {} argument(s), got {}",
                decl.params.len(), args.len()
            )));
        }

        // Arguments are evaluated in the CALLER's environment (that's the
        // one place the outside world gets in) -- the results are plain
        // values by the time the function sees them.
        let mut arg_values = Vec::with_capacity(args.len());
        for arg in args {
            arg_values.push(self.eval(arg)?);
        }

        let mut local_env = HashMap::new();
        for (param, value) in decl.params.iter().zip(arg_values) {
            local_env.insert(param.clone(), Binding { value, mutable: false });
        }

        // Swap the whole environment out, run the body, always swap the
        // caller's environment back -- even if the call below errors.
        let caller_env = std::mem::replace(&mut self.env, local_env);
        let caller_env = std::mem::replace(&mut self.env, vec![local_env]);
        let result = self.run_function_body(&decl.body);
        self.env = caller_env;

        result
    }

    /// Runs a function body statement-by-statement in whatever `self.env`
    /// currently is (the caller sets that up). Stops and returns as soon
    /// as it hits a `return`, wherever that happens to be -- so early
    /// returns work even though the parser only requires the *last*
    /// statement to be one.
    fn run_function_body(&mut self, body: &[Stmt]) -> Result<Value, RuntimeError> {
        match self.exec_body(body)? {
            Flow::Return(v) => Ok(v),
            Flow::Normal => Err(RuntimeError::new("function body did not return a value")),
        }
        // Unreachable for any function that made it through parsing,
        // since `parse_fn_def` requires the body to end with `Return`.
        Err(RuntimeError::new(
            "function body did not return a value",
        ))
    }

    /// Enforces "no side effects" for a `fn` body. The only two
    /// side-effecting things this language can do are `print` (writes to
    /// stdout) and `t<TICKER>` (shells out to Python and hits the
    /// filesystem/network) -- both are rejected, at any depth, anywhere
    /// in the body (including inside `if` branches).
    fn check_purity(&self, body: &[Stmt]) -> Result<(), RuntimeError> {
        for stmt in body {
            match stmt {
                Stmt::Print(_) => {
                    return Err(RuntimeError::new(
                        "`print` is a side effect and isn't allowed inside a function body or [unsafe] block",
                    ))
                }
                Stmt::FnDef { .. } => {
                    return Err(RuntimeError::new(
                        "nested function definitions aren't supported",
                    ))
                }
                Stmt::Assign { value, .. } => self.check_expr_purity(value)?,
                Stmt::Return(expr) => self.check_expr_purity(expr)?,
                Stmt::While { condition, body } => {
                    self.check_expr_purity(condition)?;
                    self.check_purity(body)?;
                }
            }
        }
        Ok(())
    }

    fn check_expr_purity(&self, expr: &Expr) -> Result<(), RuntimeError> {
        match expr {
            Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Var(_) => Ok(()),
            Expr::Field { base, .. } => self.check_expr_purity(base),
            Expr::Subset { base, .. } => {
                if let Expr::Var(class_name) = base.as_ref() {
                    if class_name == CLASS_T {
                        return Err(RuntimeError::new(
                            "`t<...>` fetches external data and isn't allowed inside a function body or [unsafe] block",
                        ));
                    }
                }
                self.check_expr_purity(base)
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.check_expr_purity(lhs)?;
                self.check_expr_purity(rhs)
            }
            Expr::If { condition, then_branch, else_branch } => {
                self.check_expr_purity(condition)?;
                self.check_expr_purity(then_branch)?;
                self.check_expr_purity(else_branch)
            }
            Expr::Call { args, .. } => {
                // Calling another function is fine -- that function had to
                // pass this same purity check when it was defined.
                for arg in args {
                    self.check_expr_purity(arg)?;
                }
                Ok(())
            }
            Expr::Unsafe { body } => self.check_purity(body),
        }
    }

    /// Look up `key` inside a `Value::Set`. This is what both `.field` and
    /// `<subset>` do at runtime -- they're the same operation with
    /// different spelling. A JSON leaf (number/string) becomes a scalar
    /// `Value`; a nested object stays a `Value::Set` so it can be indexed
    /// again (e.g. `APPL.earnings` then `<ttm>`).
    fn subset_lookup(base: &Value, key: &str) -> Result<Value, RuntimeError> {
        match base {
            Value::Set(json) => {
                let obj = json
                    .as_object()
                    .ok_or_else(|| RuntimeError::new(format!("no subsets to look `{key}` up in")))?;
                let found = obj
                    .get(key)
                    .ok_or_else(|| RuntimeError::new(format!("no subset named `{key}`")))?;
                Ok(Self::json_to_value(found))
            }
            other => Err(RuntimeError::new(format!(
                "cannot look up `{key}` on {other} -- it isn't a set of subsets"
            ))),
        }
    }

    fn json_to_value(json: &Json) -> Value {
        match json {
            Json::Number(n) => Value::Number(n.as_f64().unwrap_or(f64::NAN)),
            Json::String(s) => Value::Str(s.clone()),
            other => Value::Set(other.clone()),
        }
    }

    fn construct_ticker(&mut self, ticker: &str) -> Result<Value, RuntimeError> {
        std::fs::create_dir_all(&self.data_dir)
            .map_err(|e| RuntimeError::new(format!("could not create '{}': {e}", self.data_dir.display())))?;

        let json_path = self.data_dir.join(format!("{ticker}.json"));

        // If a prefetch pass (see `collect_tickers` + `main.rs`) already
        // fetched this ticker before the program started running, skip
        // shelling out again -- but only trust `self.prefetched`, not
        // "does the file exist," so a stale file left over from some
        // earlier, unrelated run never gets silently reused. A prefetch
        // failure for this ticker just falls through to the normal path
        // below, so the script still gets a real, accurate error here if
        // it still fails.
        if !self.prefetched.contains(ticker) || !json_path.is_file() {
            let status = Command::new(&self.python_bin)
                .arg(&self.generator_script)
                .arg(ticker)
                .arg(&self.data_dir)
                .status()
                .map_err(|e| {
                    RuntimeError::new(format!(
                        "failed to run `{} {}`: {e} (is python3 installed and on PATH?)",
                        self.python_bin, self.generator_script.display()
                    ))
                })?;

            if !status.success() {
                return Err(RuntimeError::new(format!(
                    "generator script exited with {status} for ticker `{ticker}`"
                )));
            }
        }

        let value = self.load_json(&json_path)?;

        // The ticker itself becomes a variable, independent of whatever
        // name it was assigned to (`company`, in the example program).
        self.env_insert(ticker.to_string(), Binding { value: value.clone(), mutable: false });
        Ok(value)
    }

    fn load_json(&self, path: &Path) -> Result<Value, RuntimeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| RuntimeError::new(format!("could not read '{}': {e}", path.display())))?;
        let json: Json = serde_json::from_str(&content)
            .map_err(|e| RuntimeError::new(format!("invalid JSON in '{}': {e}", path.display())))?;
        Ok(Value::Set(json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;
    use std::io::Write;

    /// Interpreter tests write a tiny throwaway "generator" that just
    /// echoes a fixed JSON payload, so they don't depend on python or
    /// real randomness.
    fn interpreter_with_fixture(dir: &Path, json_body: &str) -> Interpreter {
        let script_path = dir.join("fixture_generator.py");
        let mut f = std::fs::File::create(&script_path).unwrap();
        writeln!(f, "import sys, pathlib\n... [snip python fixture stub] ...").unwrap();
        Interpreter::new(dir.join("data"), script_path)
    }

    fn python_available() -> bool {
        Command::new("python3")
            .arg("--version")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn arithmetic_and_variables() {
        let src = "a = 2; b = 3; c = a + b * 2;";
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        let mut interp = Interpreter::new("unused_data", "unused_script.py");
        interp.run(&program).unwrap();
        match interp.env.get("c").unwrap() {
            Value::Number(n) => assert_eq!(*n, 8.0),
            other => panic!("expected number, got {other:?}"),
        }
    }

    #[test]
    fn undefined_variable_errors() {
        let src = "a = b;";
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        let mut interp = Interpreter::new("unused_data", "unused_script.py");
        let err = interp.run(&program).unwrap_err();
        assert!(err.message.contains("undefined variable"));
    }

    #[test]
    fn division_by_zero_errors() {
        let src = "a = 1 / 0;";
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        let mut interp = Interpreter::new("unused_data", "unused_script.py");
        let err = interp.run(&program).unwrap_err();
        assert!(err.message.contains("division by zero"));
    }

    #[test]
    fn reassigning_a_variable_is_an_error() {
        let src = "a = 1; a = 2;";
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        let mut interp = Interpreter::new("unused_data", "unused_script.py");
        let err = interp.run(&program).unwrap_err();
        assert!(err.message.contains("immutable"));
    }

    #[test]
    fn ticker_construction_and_subset_access() {
        if !python_available() {
            eprintln!("skipping: python3 not available in this environment");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("fin_lang_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let json_body = r#"{"earnings": {"last": 2.0, "ttm": 8.0}, "price": {"last": 4.0, "ttm": 3.0}}"#;
        let mut interp = interpreter_with_fixture(&tmp, json_body);

        let src = "company = t<APPL>;\n\
                    earnings = APPL.earnings <ttm>;\n\
                    price = APPL.price <last>;\n\
                    ratio = earnings/price;\n";
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        interp.run(&program).unwrap();

        match interp.env.get("ratio").unwrap() {
            Value::Number(n) => assert_eq!(*n, 2.0), // 8.0 / 4.0
            other => panic!("expected number, got {other:?}"),
        }
        // `APPL` should be bound as its own variable, independent of `company`.
        assert!(matches!(interp.env.get("APPL"), Some(Value::Set(_))));
        assert!(matches!(interp.env.get("company"), Some(Value::Set(_))));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ---- functions ----------------------------------------------------

    fn run(src: &str) -> Result<Interpreter, RuntimeError> {
        let program = Parser::new(tokenize(src).unwrap()).parse_program().unwrap();
        let mut interp = Interpreter::new("unused_data", "unused_script.py");
        interp.run(&program)?;
        Ok(interp)
    }

    #[test]
    fn pure_function_call() {
        let interp = run("fn square(x) { return x * x; } result = square(5);").unwrap();
        match interp.env.get("result").unwrap() {
            Value::Number(n) => assert_eq!(*n, 25.0),
            other => panic!("expected number, got {other:?}"),
        }
    }

    #[test]
    fn function_with_multiple_params_and_an_if() {
        let src = "fn max(a, b) { return if a > b { a } else { b }; } result = max(3, 7);";
        let interp = run(src).unwrap();
        match interp.env.get("result").unwrap() {
            Value::Number(n) => assert_eq!(*n, 7.0),
            other => panic!("expected number, got {other:?}"),
        }
    }

    #[test]
    fn function_cannot_see_caller_variables() {
        // `secret` is not a parameter of `leak`, so it should be
        // "undefined" from inside the function body -- functions only
        // see their own parameters, never the caller's environment.
        let src = "secret = 42; fn leak() { return secret; } x = leak();";
        let err = run(src).unwrap_err();
        assert!(err.message.contains("undefined variable"));
    }

    #[test]
    fn print_inside_function_is_rejected_at_definition_time() {
        let src = "fn noisy(x) { print(x); return x; }";
        let err = run(src).unwrap_err();
        assert!(err.message.contains("side effect"));
    }

    #[test]
    fn ticker_construction_inside_function_is_rejected() {
        let src = "fn make(sym) { return t<sym>; }";
        let err = run(src).unwrap_err();
        assert!(err.message.contains("external data"));
    }

    #[test]
    fn wrong_argument_count_is_an_error() {
        let src = "fn add(a, b) { return a + b; } x = add(1);";
        let err = run(src).unwrap_err();
        assert!(err.message.contains("expects 2 argument"));
    }

    #[test]
    fn calling_undefined_function_is_an_error() {
        let src = "x = mystery(1);";
        let err = run(src).unwrap_err();
        assert!(err.message.contains("undefined function"));
    }

    #[test]
    fn function_can_call_another_function() {
        let src = "fn double(x) { return x * 2; } \
                    fn quadruple(x) { return double(double(x)); } \
                    result = quadruple(3);";
        let interp = run(src).unwrap();
        match interp.env.get("result").unwrap() {
            Value::Number(n) => assert_eq!(*n, 12.0),
            other => panic!("expected number, got {other:?}"),
        }
    }
}