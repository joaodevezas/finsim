//! A small recursive-descent parser sitting on top of `lexer.rs`.
//!
//! Grammar (informal):
//!
//! ```text
//! program    := stmt* EOF
//! stmt       := "print" "(" expr ")" ";"
//!             | "fn" IDENT "(" (IDENT ("," IDENT)*)? ")" "{" fn_body "}"
//!             | "return" expr ";"
//!             | "mut" IDENT "=" expr ";"
//!             | IDENT "=" expr ";"
//!             | "while" expr "{" stmt* "}"
//! fn_body    := stmt*                 -- must end with a "return" stmt
//!
//! expr       := comparison
//! comparison := term (("+" | "-" | "<" | ">" | "<=" | ">=" | "==" | "!=") term)*
//! term       := unary (("*" | "/") unary)*
//! unary      := "-" unary | postfix
//! postfix    := primary ( "." IDENT | "<" IDENT ">" )*
//! primary    := INT | FLOAT | STRING | IDENT
//!             | IDENT "(" (expr ("," expr)*)? ")"   -- function call
//!             | "if" expr "{" expr "}" "else" "{" expr "}"
//!             | "[" "unsafe" "]" "{" stmt* "}"      -- unsafe block, ends in return
//!             | "(" expr ")"
//! ```
//!
//! Two things changed from the original design worth calling out:
//!
//! 1. **Statements now end at `;`, not at a newline.** The lexer no longer
//!    treats newlines as significant at all (see `lexer.rs`), so a
//!    statement -- a function body, for instance -- can freely span
//!    multiple lines. The one exception: a `fn ... { ... }` definition
//!    does NOT need a trailing `;` after its closing `}`, since the brace
//!    already unambiguously ends it (see `stmt_needs_semicolon`).
//!
//! 2. **`print` now uses call syntax**, `print(expr)`, instead of
//!    `print <expr>`. This isn't a functional requirement, but it makes
//!    `print` look like what it now sits next to -- real function calls
//!    (`square(x)`) -- instead of looking like a `<subset>` access, which
//!    it never actually was.
//!
//! `postfix` is still the interesting rule for data access -- it's what
//! turns `t<APPL>`, `APPL.earnings`, and `APPL.earnings <ttm>` into the
//! same shape of tree (a chain of "field" and "subset" accesses on a base
//! expression). The parser does NOT decide that `t<APPL>` means "construct
//! a ticker" -- that's a runtime decision made by the interpreter. See
//! `interpreter.rs`.

use crate::lexer::{SpannedToken, Token};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Lt, Gt, Le, Ge, Eq, Ne, Power,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Power => "^",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    /// A bare identifier: `ratio`, `price`, `t`, ...
    Var(String),
    /// `base.name` -- dot access into a "set".
    Field { base: Box<Expr>, name: String },
    /// `base<name>` -- angle-bracket access into a "set" (or, when `base`
    /// is the bare identifier `t`, a request to construct a new ticker --
    /// see `interpreter.rs::eval`).
    Subset { base: Box<Expr>, name: String },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    If { condition: Box<Expr>, then_branch: Box<Expr>, else_branch: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
    Unsafe { body: Vec<Stmt> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign { name: String, value: Expr, mutable: bool },
    Print(Vec<Expr>),
    FnDef { name: String, params: Vec<String>, body: Vec<Stmt> },
    While { condition: Expr, body: Vec<Stmt> },
    Return(Expr),
}

pub type Program = Vec<Stmt>;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}:{}: {}", self.line, self.col, self.message)
    }
}
impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Convenience: tokenize-free entry point isn't provided here on purpose
    /// -- construct with `Parser::new(lexer::tokenize(src)?)` from the caller
    /// so lex errors and parse errors stay clearly separate.
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut stmts = Vec::new();
        while !self.at_eof() {
            let stmt = self.parse_stmt()?;
            self.expect_stmt_end(&stmt)?;
            stmts.push(stmt);
        }
        Ok(stmts)
    }

    // ---- statements ----------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        if let Token::Ident(name) = &self.current().token {
            let name = name.clone();

            if name == "print" {
                self.advance();
                self.expect(&Token::LParen)?;
                let mut args = Vec::new();
                if self.current().token != Token::RParen {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.current().token == Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RParen)?;
                return Ok(Stmt::Print(args));
            }

            if name == "fn" {
                return self.parse_fn_def();
            }

            if name == "return" {
                self.advance();
                let expr = self.parse_expr()?;
                return Ok(Stmt::Return(expr));
            }

            if name == "while" {
                self.advance();
                let condition = self.parse_expr()?;
                self.expect(&Token::LBrace)?;
                let mut body = Vec::new();
                while self.current().token != Token::RBrace {
                    let stmt = self.parse_stmt()?;
                    self.expect_stmt_end(&stmt)?;
                    body.push(stmt);
                }
                self.expect(&Token::RBrace)?;
                return Ok(Stmt::While { condition, body });
            }

            if name == "mut" {
                self.advance();
                let var_name = self.expect_ident()?;
                self.expect(&Token::Equals)?;
                let value = self.parse_expr()?;
                return Ok(Stmt::Assign { name: var_name, value, mutable: true });
            }
        }

        let name = self.expect_ident()?;
        self.expect(&Token::Equals)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Assign { name, value, mutable: false })
    }

    /// `fn name(p1, p2) { stmt* }` -- the body is parsed exactly like a
    /// program (a list of `;`-terminated statements), except it must end
    /// with a `return`, which the parser enforces here rather than leaving
    /// it as a runtime surprise.
    fn parse_fn_def(&mut self) -> Result<Stmt, ParseError> {
        self.advance(); // consume 'fn'
        let name = self.expect_ident()?;

        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        if self.current().token != Token::RParen {
            loop {
                params.push(self.expect_ident()?);
                if self.current().token == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&Token::RParen)?;

        self.expect(&Token::LBrace)?;
        let mut body = Vec::new();
        while self.current().token != Token::RBrace {
            let stmt = self.parse_stmt()?;
            self.expect_stmt_end(&stmt)?;
            body.push(stmt);
        }
        self.expect(&Token::RBrace)?;

        if !matches!(body.last(), Some(Stmt::Return(_))) {
            return Err(self.error(format!("function `{name}` must end with a `return` statement")));
        }

        Ok(Stmt::FnDef { name, params, body })
    }

    // ---- expressions (precedence climbing) ------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.current().token {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                Token::LAngle => BinOp::Lt,
                Token::RAngle => BinOp::Gt,
                Token::LessEq => BinOp::Le,
                Token::GreaterEq => BinOp::Ge,
                Token::EqEq => BinOp::Eq,
                Token::NotEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_term()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.current().token {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.current().token == Token::Minus {
            self.advance();
            let operand = self.parse_unary()?;
            return Ok(Expr::Binary {
                op: BinOp::Sub,
                lhs: Box::new(Expr::Int(0)),
                rhs: Box::new(operand),
            });
        }
        self.parse_power() 
    }

    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_postfix()?;
        
        if self.current().token == Token::Caret {
            self.advance();
            let rhs = self.parse_power()?;
            lhs = Expr::Binary { op: BinOp::Power, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        
        Ok(lhs)
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.current().token {
                Token::Dot => {
                    self.advance();
                    let name = self.expect_ident()?;
                    expr = Expr::Field { base: Box::new(expr), name };
                }
                Token::LAngle => {
                    // Peek ahead to see if it's a subset `<ttm>` / `<CS.PA>`
                    // or a less-than `< 300`. Tickers with an exchange
                    // suffix (e.g. `CS.PA` for AXA on Euronext Paris,
                    // `600519.SS` for a Shanghai listing) lex as a DOTTED
                    // CHAIN of identifiers (`Ident Dot Ident ...`), not one
                    // token -- `.` is its own `Token::Dot` -- so the
                    // lookahead has to walk the whole chain, not just peek
                    // one identifier ahead.
                    if self.peek_is_subset_name() {
                        self.advance(); // consume '<'
                        let mut name = self.expect_ident()?;
                        while self.current().token == Token::Dot {
                            self.advance();
                            let part = self.expect_ident()?;
                            name.push('.');
                            name.push_str(&part);
                        }
                        self.expect(&Token::RAngle)?;
                        expr = Expr::Subset { base: Box::new(expr), name };
                    } else {
                        break; 
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Looks ahead from the current `<` (not yet consumed) to decide
    /// whether this is a subset access -- `<ttm>`, or a dotted chain like
    /// `<CS.PA>` / `<600519.SS>` -- as opposed to a `<` used as the
    /// less-than operator. Matches `IDENT ('.' IDENT)* '>'` immediately
    /// after the `<`, without consuming anything.
    fn peek_is_subset_name(&self) -> bool {
        let mut i = self.pos + 1;
        match self.tokens.get(i).map(|t| &t.token) {
            Some(Token::Ident(_)) => i += 1,
            _ => return false,
        }
        loop {
            match self.tokens.get(i).map(|t| &t.token) {
                Some(Token::RAngle) => return true,
                Some(Token::Dot) => {
                    i += 1;
                    match self.tokens.get(i).map(|t| &t.token) {
                        Some(Token::Ident(_)) => i += 1,
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let tok = self.current().token.clone();
        match tok {
            Token::Int(n) => { self.advance(); Ok(Expr::Int(n)) }
            Token::Float(n) => { self.advance(); Ok(Expr::Float(n)) }
            Token::Str(s) => { self.advance(); Ok(Expr::Str(s)) }
            Token::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Token::LBracket => {
                self.advance();
                let ident = self.expect_ident()?;
                if ident != "unsafe" {
                    return Err(self.error("expected 'unsafe' inside brackets"));
                }
                self.expect(&Token::RBracket)?;
                self.expect(&Token::LBrace)?;
                let mut body = Vec::new();
                while self.current().token != Token::RBrace {
                    let stmt = self.parse_stmt()?;
                    self.expect_stmt_end(&stmt)?;
                    body.push(stmt);
                }
                self.expect(&Token::RBrace)?;
                if !matches!(body.last(), Some(Stmt::Return(_))) {
                    return Err(self.error("[unsafe] block must end with a `return` statement"));
                }
                Ok(Expr::Unsafe { body })
            }
            Token::Ident(name) => {
                if name == "if" {
                    // 1. Parse the condition (e.g., price < 300)
                    self.advance(); 
                    let condition = self.parse_expr()?;
        
                    // 2. Parse the 'then' block
                    
                    self.expect(&Token::LBrace)?;
                    let then_branch = self.parse_expr()?;
                    self.expect(&Token::RBrace)?;
        
                    // 3. Expect the word "else" (The lazy way!)
                    let next_tok = self.advance();
                    if next_tok.token != Token::Ident("else".to_string()) {
                        return Err(self.error("expected 'else' after if block"));
                    }
        
                    // 4. Parse the 'else' block
                    self.expect(&Token::LBrace)?;
                    let else_branch = self.parse_expr()?;
                    self.expect(&Token::RBrace)?;
        
                    return Ok(Expr::If {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    });
                }

                self.advance();
                if self.current().token == Token::LParen {
                    self.advance(); 
                    let mut args = Vec::new();
                    if self.current().token != Token::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.current().token == Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::Call { name, args });
                }

                Ok(Expr::Var(name))
            }
            other => Err(self.error(format!("expected an expression, found {other}"))),
        }
    }

    // ---- token-stream helpers -------------------------------------------

    fn current(&self) -> &SpannedToken {
        // `tokenize` always ends with exactly one Eof, so this never runs
        // past the end as long as callers don't advance past it.
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> SpannedToken {
        let tok = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn at_eof(&self) -> bool {
        self.current().token == Token::Eof
    }

    fn expect(&mut self, want: &Token) -> Result<SpannedToken, ParseError> {
        if &self.current().token == want {
            Ok(self.advance())
        } else {
            let found = self.current().token.clone();
            Err(self.error(format!("expected {want}, found {found}")))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.current().token.clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.error(format!("expected an identifier, found {other}"))),
        }
    }

    /// A statement ends at `;` -- except a `fn ... { ... }` definition,
    /// whose closing `}` already unambiguously ends it (writing a `;`
    /// after it too would just be visual noise, so it's not required).
    fn expect_stmt_end(&mut self, stmt: &Stmt) -> Result<(), ParseError> {
        if matches!(stmt, Stmt::FnDef { .. } | Stmt::While { .. }) {
            return Ok(());
        }
        self.expect(&Token::Semicolon)?;
        Ok(())
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        let tok = self.current();
        ParseError {
            message: message.into(),
            line: tok.line,
            col: tok.col,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    fn parse(src: &str) -> Program {
        Parser::new(tokenize(src).unwrap()).parse_program().unwrap()
    }

    #[test]
    fn simple_assignment() {
        let prog = parse("x = 5;");
        assert_eq!(
            prog,
            vec![Stmt::Assign {
                name: "x".into(),
                value: Expr::Int(5)
            }]
        );
    }

    #[test]
    fn ticker_construction_shape() {
        let prog = parse("company = t<APPL>;");
        assert_eq!(
            prog,
            vec![Stmt::Assign {
                name: "company".into(),
                value: Expr::Subset {
                    base: Box::new(Expr::Var("t".into())),
                    name: "APPL".into(),
                }
            }]
        );
    }

    #[test]
    fn field_then_subset() {
        let prog = parse("earnings = APPL.earnings <ttm>;");
        assert_eq!(
            prog,
            vec![Stmt::Assign {
                name: "earnings".into(),
                value: Expr::Subset {
                    base: Box::new(Expr::Field {
                        base: Box::new(Expr::Var("APPL".into())),
                        name: "earnings".into(),
                    }),
                    name: "ttm".into(),
                }
            }]
        );
    }

    #[test]
    fn division() {
        let prog = parse("ratio = earnings/price;");
        assert_eq!(
            prog,
            vec![Stmt::Assign {
                name: "ratio".into(),
                value: Expr::Binary {
                    op: BinOp::Div,
                    lhs: Box::new(Expr::Var("earnings".into())),
                    rhs: Box::new(Expr::Var("price".into())),
                }
            }]
        );
    }

    #[test]
    fn print_stmt() {
        let prog = parse("print(ratio);");
        assert_eq!(prog, vec![Stmt::Print(vec![Expr::Var("ratio".into())])]);
    }

    #[test]
    fn precedence_mul_before_add() {
        let prog = parse("x = 1 + 2 * 3;");
        assert_eq!(
            prog,
            vec![Stmt::Assign {
                name: "x".into(),
                value: Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::Int(1)),
                    rhs: Box::new(Expr::Binary {
                        op: BinOp::Mul,
                        lhs: Box::new(Expr::Int(2)),
                        rhs: Box::new(Expr::Int(3)),
                    }),
                }
            }]
        );
    }

    #[test]
    fn full_example_parses() {
        // Same program as the original design, just with `;` endings and
        // `print(...)` instead of `print <...>`. Note it's now also legal
        // to spread a single logical statement across lines, since
        // newlines aren't significant anymore -- not exercised here, but
        // see `semicolons_allow_multiline_statements` below.
        let src = "company = t<APPL>;\n\
                    earnings = APPL.earnings <ttm>;\n\
                    price = APPL.price <last>;\n\
                    ratio = earnings/price;\n\
                    print(ratio);\n";
        let prog = parse(src);
        assert_eq!(prog.len(), 5);
    }

    #[test]
    fn semicolons_allow_multiline_statements() {
        // A single statement's expression can now span multiple lines,
        // since a newline is just whitespace -- only `;` ends a statement.
        let prog = parse(
            "x = 1
                 + 2
                 + 3;",
        );
        assert_eq!(prog.len(), 1);
    }

    #[test]
    fn missing_semicolon_is_an_error() {
        let tokens = tokenize("x = 5").unwrap(); // no trailing `;`
        let err = Parser::new(tokens).parse_program().unwrap_err();
        assert!(err.message.contains(";"));
    }

    #[test]
    fn missing_equals_is_an_error() {
        let tokens = tokenize("x 5;").unwrap();
        let err = Parser::new(tokens).parse_program().unwrap_err();
        assert!(err.message.contains("expected"));
    }

    // ---- functions --------------------------------------------------

    #[test]
    fn function_def_and_call_shape() {
        let prog = parse("fn square(x) { return x * x; } y = square(5);");
        assert_eq!(
            prog,
            vec![
                Stmt::FnDef {
                    name: "square".into(),
                    params: vec!["x".into()],
                    body: vec![Stmt::Return(Expr::Binary {
                        op: BinOp::Mul,
                        lhs: Box::new(Expr::Var("x".into())),
                        rhs: Box::new(Expr::Var("x".into())),
                    })],
                },
                Stmt::Assign {
                    name: "y".into(),
                    value: Expr::Call {
                        name: "square".into(),
                        args: vec![Expr::Int(5)],
                    },
                },
            ]
        );
    }
}