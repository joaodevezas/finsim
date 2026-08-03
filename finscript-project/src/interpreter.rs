//! Interpreter: walks the AST from `parser.rs` and actually runs the
//! program.

use crate::parser::{BinOp, Expr, Program, Stmt};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) const CLASS_T: &str = "t";

#[derive(Debug, Clone)]
struct FnDecl {
    params: Vec<String>,
    body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Str(String),
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
        RuntimeError { message: message.into() }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime error: {}", self.message)
    }
}
impl std::error::Error for RuntimeError {}

pub type NativeFn = fn(&[Value]) -> Result<Value, RuntimeError>;

#[derive(Debug)]
pub struct Interpreter {
    // True lexical block scopes, searching from the end (innermost) backwards.
    env: Vec<HashMap<String, Binding>>,
    functions: HashMap<String, FnDecl>,
    natives: HashMap<String, NativeFn>,
    data_dir: PathBuf,
    generator_script: PathBuf,
    python_bin: String,
    prefetched: std::collections::HashSet<String>,
    in_unsafe: usize,
    /// Raw `--name=value` overrides supplied on the command line, used to
    /// fill in `param()` slots (see `resolve_param` and DAG files in
    /// `dag.rs`). Keyed by variable name, exactly as written in the source.
    params: HashMap<String, String>,
}

#[derive(Debug, Clone)]
enum Flow {
    Normal,
    Return(Value),
}

/// Parse a raw `--name=value` string (from the command line) into a
/// `Value`: numbers parse as `Value::Number`, everything else is kept as
/// a `Value::Str` verbatim (no quoting needed on the CLI).
fn parse_param_value(raw: &str) -> Value {
    match raw.parse::<f64>() {
        Ok(n) => Value::Number(n),
        Err(_) => Value::Str(raw.to_string()),
    }
}

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
            params: HashMap::new(),
        }
    }

    /// Supply `--name=value` overrides (as parsed from the command line)
    /// to be used for any `param()` slots the program contains -- see the
    /// DAG-file format in `dag.rs`.
    pub fn with_params(mut self, params: HashMap<String, String>) -> Self {
        self.params = params;
        self
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

    pub fn mark_prefetched(&mut self, ticker: impl Into<String>) {
        self.prefetched.insert(ticker.into());
    }

    pub fn data_dir(&self) -> &Path { &self.data_dir }
    pub fn python_bin(&self) -> &str { &self.python_bin }

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
                // `name = param();` / `name = param(default);` is a
                // pseudo-call: a DAG-file parameter slot, resolved
                // against `--name=value` on the command line (or against
                // the optional default expression) rather than evaluated
                // like an ordinary function call. See `dag.rs`.
                let v = if let Expr::Call { name: fname, args } = value {
                    if fname.as_str() == "param" {
                        self.resolve_param(name, args)?
                    } else {
                        self.eval(value)?
                    }
                } else {
                    self.eval(value)?
                };

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

    /// Resolve a `param()` slot for the top-level variable `var_name`.
    ///
    /// - If `--var_name=value` was supplied on the command line, use it
    ///   (parsed as a number when possible, otherwise kept as a string).
    /// - Otherwise, if `param(default)` was written with a default
    ///   expression, evaluate and use that.
    /// - Otherwise, this is a required parameter that wasn't supplied:
    ///   error out with a message telling the user how to supply it.
    fn resolve_param(&mut self, var_name: &str, args: &[Expr]) -> Result<Value, RuntimeError> {
        if let Some(raw) = self.params.get(var_name).cloned() {
            return Ok(parse_param_value(&raw));
        }
        match args.len() {
            0 => Err(RuntimeError::new(format!(
                "missing required parameter `{var_name}`; supply it with --{var_name}=<value>"
            ))),
            1 => self.eval(&args[0]),
            n => Err(RuntimeError::new(format!(
                "`param()` takes at most one argument (a default value), got {n}"
            ))),
        }
    }

    /// Build the dependency DAG for `program` and write it out as both a
    /// `.fic` DAG file and a `.dot` Graphviz file, in this interpreter's
    /// data directory, named after `source_stem` (e.g. `"test"` ->
    /// `data/test.fic` + `data/test.dot`). Returns (fic_path, dot_path).
    ///
    /// The `.dot` file isn't runnable by anything in this crate -- it's
    /// for visualizing the graph with an external Graphviz renderer, e.g.:
    ///   `dot -Tpng data/test.dot -o data/test.png`
    /// or by pasting its contents into an online viewer such as
    /// https://dreampuf.github.io/GraphvizOnline/.
    pub fn save_dag_file(
        &self,
        program: &Program,
        source_stem: &str,
    ) -> Result<(PathBuf, PathBuf), RuntimeError> {
        let dag = crate::dag::build_dag(program)
            .map_err(|e| RuntimeError::new(format!("could not build DAG: {e}")))?;
        let rendered = crate::dag::render_dag_file(program, &dag);
        let dot = dag.to_dot();

        std::fs::create_dir_all(&self.data_dir)
            .map_err(|e| RuntimeError::new(format!("could not create '{}': {e}", self.data_dir.display())))?;

        let fic_path = self.data_dir.join(format!("{source_stem}.fic"));
        std::fs::write(&fic_path, rendered)
            .map_err(|e| RuntimeError::new(format!("could not write '{}': {e}", fic_path.display())))?;

        let dot_path = self.data_dir.join(format!("{source_stem}.dot"));
        std::fs::write(&dot_path, dot)
            .map_err(|e| RuntimeError::new(format!("could not write '{}': {e}", dot_path.display())))?;

        Ok((fic_path, dot_path))
    }

    fn call_function(&mut self, name: &str, args: &[Expr]) -> Result<Value, RuntimeError> {
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

        let mut arg_values = Vec::with_capacity(args.len());
        for arg in args {
            arg_values.push(self.eval(arg)?);
        }

        let mut local_env = HashMap::new();
        for (param, value) in decl.params.iter().zip(arg_values) {
            local_env.insert(param.clone(), Binding { value, mutable: false });
        }

        let caller_env = std::mem::replace(&mut self.env, vec![local_env]);
        let result = self.run_function_body(&decl.body);
        self.env = caller_env;

        result
    }

    fn run_function_body(&mut self, body: &[Stmt]) -> Result<Value, RuntimeError> {
        match self.exec_body(body)? {
            Flow::Return(v) => Ok(v),
            Flow::Normal => Err(RuntimeError::new("function body did not return a value")),
        }
    }

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
                for arg in args {
                    self.check_expr_purity(arg)?;
                }
                Ok(())
            }
            Expr::Unsafe { body } => self.check_purity(body),
        }
    }

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

    fn interpreter_with_fixture(dir: &Path, json_body: &str) -> Interpreter {
        let script_path = dir.join("fixture_generator.py");
        let mut f = std::fs::File::create(&script_path).unwrap();
        writeln!(f, "import sys, pathlib\n... [snip python fixture stub] ...").unwrap();
        Interpreter::new(dir.join("data"), script_path)
    }
    
    // ... all other existing tests (unchanged, just remember to use .env_get() inside tests)
}