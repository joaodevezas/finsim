//! Prewritten "built-in" functions, written in plain Rust, registered
//! into an `Interpreter` so DSL source can call them exactly as if they
//! were part of the language -- e.g. `dcf(...)`.
//!
//! To add a new one:
//!   1. Write it in its own file here (copy `dcf.rs`'s shape: a function
//!      `fn(&[Value]) -> Result<Value, RuntimeError>` that validates its
//!      own arg count/types).
//!   2. Add `mod your_file;` below.
//!   3. Register it in `register_all`.
//!
//! That's the entire integration surface -- no lexer or parser changes
//! are needed, since `name(args)` already parses as `Expr::Call` for any
//! identifier, whether or not a user `fn` of that name exists.

mod dcf;

use crate::interpreter::Interpreter;

/// Registers every prewritten function. Call this once, right after
/// constructing the `Interpreter` and before `run`:
///
/// ```ignore
/// let mut interp = Interpreter::new(data_dir, generator_script);
/// stdlib::register_all(&mut interp);
/// interp.run(&program)?;
/// ```
pub fn register_all(interp: &mut Interpreter) {
    interp.register_native("dcf", dcf::dcf);
}
