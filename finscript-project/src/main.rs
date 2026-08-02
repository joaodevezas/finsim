mod interpreter;
mod lexer;
mod parser;
mod stdlib;

use interpreter::Interpreter;
use lexer::tokenize;
use parser::Parser;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::mpsc;
use std::thread;

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

    let mut interp = Interpreter::new("data", generator_script.clone());
    stdlib::register_all(&mut interp);

    let tickers = interpreter::collect_tickers(&program);
    if !tickers.is_empty() {
        eprintln!(
            "prefetching {} ticker(s): {}",
            tickers.len(),
            tickers.iter().cloned().collect::<Vec<_>>().join(", ")
        );
        let fetched = prefetch_tickers(&tickers, &generator_script, interp.data_dir(), interp.python_bin());
        for ticker in fetched {
            interp.mark_prefetched(ticker);
        }
    }

    if let Err(e) = interp.run(&program) {
        eprintln!("{path_arg}: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

const PREFETCH_CONCURRENCY: usize = 6;

fn prefetch_tickers(
    tickers: &std::collections::HashSet<String>,
    generator_script: &Path,
    data_dir: &Path,
    python_bin: &str,
) -> Vec<String> {
    if let Err(e) = std::fs::create_dir_all(data_dir) {
        eprintln!("warning: could not create '{}': {e} (prefetch skipped)", data_dir.display());
        return Vec::new();
    }

    let (tx, rx) = mpsc::channel::<String>();
    for ticker in tickers { tx.send(ticker.clone()).ok(); }
    drop(tx);

    let rx = std::sync::Mutex::new(rx);
    let worker_count = PREFETCH_CONCURRENCY.min(tickers.len().max(1));

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let rx = &rx;
            handles.push(scope.spawn(move || {
                let mut succeeded = Vec::new();
                loop {
                    let ticker = {
                        let rx = rx.lock().unwrap();
                        rx.recv()
                    };
                    let Ok(ticker) = ticker else { break };

                    let status = Command::new(python_bin)
                        .arg(generator_script)
                        .arg(&ticker)
                        .arg(data_dir)
                        .status();

                    match status {
                        Ok(s) if s.success() => succeeded.push(ticker),
                        Ok(s) => eprintln!("warning: prefetch of `{ticker}` exited with {s}"),
                        Err(e) => eprintln!("warning: could not prefetch `{ticker}`: {e}"),
                    }
                }
                succeeded
            }));
        }
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

fn find_generator_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FIN_GENERATOR_SCRIPT") {
        let p = PathBuf::from(p);
        if p.is_file() { return Some(p); }
    }

    let cwd_candidate = PathBuf::from("scripts/generate_data.py");
    if cwd_candidate.is_file() { return Some(cwd_candidate); }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let from_target = exe_dir.join("../../scripts/generate_data.py");
            if from_target.is_file() { return Some(from_target); }
            let alongside = exe_dir.join("scripts/generate_data.py");
            if alongside.is_file() { return Some(alongside); }
        }
    }
    None
}