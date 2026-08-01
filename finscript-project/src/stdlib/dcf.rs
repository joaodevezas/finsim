//! Built-in `dcf(...)`: a standard discounted-cash-flow valuation,
//! callable from DSL source exactly like a user-defined `fn`:
//!
//! ```text
//! value = dcf(fcf, growth_rate, discount_rate, terminal_growth_rate, years);
//! ```
//!
//! - `fcf`: most recent free cash flow.
//! - `growth_rate`: annual growth applied to `fcf` during the explicit
//!   projection window (e.g. `0.08` for 8%).
//! - `discount_rate`: the discount/required rate of return (e.g. `0.10`).
//! - `terminal_growth_rate`: perpetual growth rate applied after the
//!   projection window (e.g. `0.02`). Must be strictly less than
//!   `discount_rate`, or the terminal value formula blows up.
//! - `years`: length of the explicit projection window, as a whole number.
//!
//! This is the template to copy for any other prewritten ("native")
//! function: a plain Rust `fn(&[Value]) -> Result<Value, RuntimeError>`
//! that validates its own argument count and types, then does normal
//! Rust math. See `mod.rs` for how this gets registered so `dcf(...)`
//! resolves as if it were part of the language.

use crate::interpreter::{RuntimeError, Value};

pub fn dcf(args: &[Value]) -> Result<Value, RuntimeError> {
    if args.len() != 5 {
        return Err(RuntimeError::new(format!(
            "dcf(fcf, growth_rate, discount_rate, terminal_growth_rate, years) \
             expects 5 arguments, got {}",
            args.len()
        )));
    }

    let fcf = as_number(&args[0], "fcf")?;
    let growth_rate = as_number(&args[1], "growth_rate")?;
    let discount_rate = as_number(&args[2], "discount_rate")?;
    let terminal_growth_rate = as_number(&args[3], "terminal_growth_rate")?;
    let years_raw = as_number(&args[4], "years")?;

    if discount_rate <= terminal_growth_rate {
        return Err(RuntimeError::new(
            "dcf: discount_rate must be greater than terminal_growth_rate \
             (otherwise the terminal value is undefined)",
        ));
    }

    let years = years_raw.round() as i64;
    if years <= 0 {
        return Err(RuntimeError::new(format!(
            "dcf: years must be a positive whole number, got {years_raw}"
        )));
    }

    // Explicit projection window: grow fcf year over year, discount each
    // year's cash flow back to present value, and sum.
    let mut pv_sum = 0.0;
    let mut projected_fcf = fcf;
    for year in 1..=years {
        projected_fcf *= 1.0 + growth_rate;
        pv_sum += projected_fcf / (1.0 + discount_rate).powi(year as i32);
    }

    // Terminal value: one more year of growth at the perpetual rate,
    // capitalized via the Gordon growth formula, then discounted back
    // from the end of the projection window to today.
    let terminal_fcf = projected_fcf * (1.0 + terminal_growth_rate);
    let terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate);
    let pv_terminal = terminal_value / (1.0 + discount_rate).powi(years as i32);

    Ok(Value::Number(pv_sum + pv_terminal))
}

fn as_number(v: &Value, arg_name: &str) -> Result<f64, RuntimeError> {
    match v {
        Value::Number(n) => Ok(*n),
        other => Err(RuntimeError::new(format!(
            "dcf: argument `{arg_name}` must be a number, found {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_produce_a_positive_valuation() {
        let args = [
            Value::Number(100.0), // fcf
            Value::Number(0.05),  // growth_rate
            Value::Number(0.10),  // discount_rate
            Value::Number(0.02),  // terminal_growth_rate
            Value::Number(5.0),   // years
        ];
        let result = dcf(&args).unwrap();
        match result {
            Value::Number(n) => assert!(n > 0.0, "expected a positive valuation, got {n}"),
            other => panic!("expected a number, got {other:?}"),
        }
    }

    #[test]
    fn wrong_argument_count_is_an_error() {
        let args = [Value::Number(100.0)];
        let err = dcf(&args).unwrap_err();
        assert!(err.message.contains("expects 5 arguments"));
    }

    #[test]
    fn non_number_argument_is_an_error() {
        let args = [
            Value::Str("not a number".into()),
            Value::Number(0.05),
            Value::Number(0.10),
            Value::Number(0.02),
            Value::Number(5.0),
        ];
        let err = dcf(&args).unwrap_err();
        assert!(err.message.contains("fcf"));
    }

    #[test]
    fn discount_rate_below_terminal_growth_is_an_error() {
        let args = [
            Value::Number(100.0),
            Value::Number(0.05),
            Value::Number(0.01), // discount_rate
            Value::Number(0.02), // terminal_growth_rate -- >= discount_rate
            Value::Number(5.0),
        ];
        let err = dcf(&args).unwrap_err();
        assert!(err.message.contains("discount_rate must be greater"));
    }

    #[test]
    fn zero_years_is_an_error() {
        let args = [
            Value::Number(100.0),
            Value::Number(0.05),
            Value::Number(0.10),
            Value::Number(0.02),
            Value::Number(0.0),
        ];
        let err = dcf(&args).unwrap_err();
        assert!(err.message.contains("positive whole number"));
    }
}
