mod interpreter;
mod lexer;
mod parser;
mod stdlib;

use interpreter::Interpreter;
use lexer::tokenize;
use parser::Parser;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let program_name = args.next().unwrap_or_else(|| "fin".to_string());

    let path_arg = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: {program_name} <file.fi>");
            return ExitCode::FAILURE;
        }
    };

    let path = Path::new(&path_arg);
    if path.extension().and_then(|e| e.to_str()) != Some("fi") {
        eprintln!("warning: '{path_arg}' does not have a .fi extension, reading it anyway");
    }

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read '{path_arg}': {e}");
            return ExitCode::FAILURE;
        }
    };

    let tokens = match tokenize(&source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path_arg}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let program = match Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{path_arg}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let generator_script = match find_generator_script() {
        Some(p) => p,
        None => {
            eprintln!(
                "error: could not find scripts/generate_data.py \
                 (looked in the current directory and next to the `fin` executable; \
                 set FIN_GENERATOR_SCRIPT to point at it explicitly)"
            );
            return ExitCode::FAILURE;
        }
    };

    let mut interp = Interpreter::new("data", generator_script);
    stdlib::register_all(&mut interp);
    if let Err(e) = interp.run(&program) {
        eprintln!("{path_arg}: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Locate scripts/generate_data.py. Checked, in order:
/// 1. `FIN_GENERATOR_SCRIPT` env var, if set.
/// 2. `scripts/generate_data.py` relative to the current directory
///    (the normal case: running `fin` from the project root).
/// 3. A couple of locations relative to the `fin` executable itself, so
///    `cargo run` and a built binary both find it without extra setup.
fn find_generator_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FIN_GENERATOR_SCRIPT") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }

    let cwd_candidate = PathBuf::from("scripts/generate_data.py");
    if cwd_candidate.is_file() {
        return Some(cwd_candidate);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // target/debug/fin -> ../../scripts/generate_data.py
            let from_target = exe_dir.join("../../scripts/generate_data.py");
            if from_target.is_file() {
                return Some(from_target);
            }
            let alongside = exe_dir.join("scripts/generate_data.py");
            if alongside.is_file() {
                return Some(alongside);
            }
        }
    }

    None
}