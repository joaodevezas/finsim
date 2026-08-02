//! A small hand-written lexer for the DSL.
//!
//! Handles input like:
//!
//! ```text
//! company = t<APPL>;
//! earnings = APPL.earnings <ttm>;
//! price = APPL.price <last>;
//! ratio = earnings/price;
//! print(ratio);
//! ```
//!
//! Design notes:
//! - Statements end at `;`, not at end-of-line. A newline is treated as
//!   ordinary whitespace (like a space or tab), so a statement -- e.g. a
//!   function body -- can freely span multiple lines.
//! - `<` and `>` are emitted as plain punctuation tokens. The lexer does not
//!   try to guess whether `<...>` means "type parameter", "subset access",
//!   or something else -- that's the parser's job, since it has context.
//! - Identifiers are just identifiers. Nothing (e.g. `print`, `t`, `if`,
//!   `fn`, `return`) is special-cased into a keyword here; see the comment
//!   on `Token::Ident` for why, and how to add real keywords later.

use std::fmt;
use std::iter::Peekable;
use std::str::CharIndices;

/// The kinds of tokens the lexer can produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // --- Literals ---
    /// Any identifier: variable names, class-like names (`t`), field
    /// names (`earnings`), etc. Keywords are *not* split out yet -- see
    /// note above `Lexer::read_ident`.
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),

    // --- Punctuation / operators ---
    Equals,   // =
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Dot,      // .
    Comma,    // ,
    LAngle,   // <
    RAngle,   // >
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]
    Semicolon, // ;
    EqEq,      // ==
    NotEq,     // !=
    LessEq,    // <=
    GreaterEq, // >=
    Caret,

    // --- Structural ---
    /// End of input. The lexer always emits exactly one of these, at the end.
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "identifier `{s}`"),
            Token::Int(n) => write!(f, "int `{n}`"),
            Token::Float(n) => write!(f, "float `{n}`"),
            Token::Str(s) => write!(f, "string {s:?}"),
            Token::Equals => write!(f, "`=`"),
            Token::Plus => write!(f, "`+`"),
            Token::Minus => write!(f, "`-`"),
            Token::Star => write!(f, "`*`"),
            Token::Slash => write!(f, "`/`"),
            Token::Dot => write!(f, "`.`"),
            Token::Comma => write!(f, "`,`"),
            Token::LAngle => write!(f, "`<`"),
            Token::RAngle => write!(f, "`>`"),
            Token::LParen => write!(f, "`(`"),
            Token::RParen => write!(f, "`)`"),
            Token::LBrace => write!(f, "`{{`"),
            Token::RBrace => write!(f, "`}}`"),
            Token::LBracket => write!(f, "`[`"),
            Token::RBracket => write!(f, "`]`"),
            Token::Semicolon => write!(f, "`;`"),
            Token::Eof => write!(f, "end of input"),
            Token::EqEq => write!(f, "`==`"),
            Token::NotEq => write!(f, "`!=`"),
            Token::LessEq => write!(f, "`<=`"),
            Token::GreaterEq => write!(f, "`>=`"),
            Token::Caret => write!(f, "`^`"),
        }
    }
}

/// A token together with the position it started at (1-indexed line/col),
/// so the parser can produce good error messages later.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub line: usize,
    pub col: usize,
}

/// A lexer error: an unexpected character, an unterminated string, or a
/// malformed number, along with where it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error at {}:{}: {}", self.line, self.col, self.message)
    }
}
impl std::error::Error for LexError {}

pub struct Lexer<'a> {
    chars: Peekable<CharIndices<'a>>,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            chars: source.char_indices().peekable(),
            line: 1,
            col: 1,
        }
    }

    /// Tokenize the whole input at once. This is the main entry point you'll
    /// call from the parser: `Lexer::new(src).tokenize()`.
    pub fn tokenize(mut self) -> Result<Vec<SpannedToken>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.token == Token::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    /// Peek at the next char without consuming it.
    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|&(_, c)| c)
    }

    /// Consume and return the next char, updating line/col bookkeeping.
    fn bump(&mut self) -> Option<char> {
        let (_, c) = self.chars.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn make_error(&self, message: impl Into<String>) -> LexError {
        LexError {
            message: message.into(),
            line: self.line,
            col: self.col,
        }
    }

    /// Skip spaces, tabs, newlines, carriage returns, and `#` line comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek_char() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.bump();
                }
                Some('#') => {
                    // Line comment: consume until newline (exclusive)
                    while let Some(c) = self.peek_char() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => break,
            }
        }
    }

    fn next_token(&mut self) -> Result<SpannedToken, LexError> {
        self.skip_whitespace_and_comments();

        let (start_line, start_col) = (self.line, self.col);

        let c = match self.peek_char() {
            Some(c) => c,
            None => {
                return Ok(SpannedToken {
                    token: Token::Eof,
                    line: start_line,
                    col: start_col,
                })
            }
        };

        let token = match c {
            '^' => {
                self.bump();
                Token::Caret
            }
            ';' => {
                self.bump();
                Token::Semicolon
            }
            '+' => {
                self.bump();
                Token::Plus
            }
            '-' => {
                self.bump();
                Token::Minus
            }
            '*' => {
                self.bump();
                Token::Star
            }
            '/' => {
                self.bump();
                Token::Slash
            }
            '.' => {
                self.bump();
                Token::Dot
            }
            ',' => {
                self.bump();
                Token::Comma
            }
            '=' => {
                self.bump();
                if self.peek_char() == Some('=') {
                    self.bump(); // consume the second '='
                    Token::EqEq
                } else {
                    Token::Equals // it's just a normal assignment
                }
            }
            '<' => {
                self.bump();
                if self.peek_char() == Some('=') {
                    self.bump();
                    Token::LessEq
                } else {
                    Token::LAngle
                }
            }
            '>' => {
                self.bump();
                if self.peek_char() == Some('=') {
                    self.bump();
                    Token::GreaterEq
                } else {
                    Token::RAngle
                }
            }
            '!' => {
                self.bump();
                if self.peek_char() == Some('=') {
                    self.bump();
                    Token::NotEq
                } else {
                    return Err(self.make_error("expected '=' after '!'"));
                }
            }
            '{' => {
                 self.bump(); 
                 Token::LBrace 
                }
            '}' => {
                 self.bump(); 
                 Token::RBrace 
                }
            '[' => {
                 self.bump(); 
                 Token::LBracket
                }
            ']' => {
                 self.bump(); 
                 Token::RBracket
                }
            '(' => {
                self.bump();
                Token::LParen
            }
            ')' => {
                self.bump();
                Token::RParen
            }
            '"' => self.read_string()?,
            c if c.is_ascii_digit() => self.read_number()?,
            c if is_ident_start(c) => self.read_ident(),
            other => {
                self.bump();
                return Err(self.make_error(format!("unexpected character '{other}'")));
            }
        };

        Ok(SpannedToken {
            token,
            line: start_line,
            col: start_col,
        })
    }

    fn read_ident(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.peek_char() {
            if is_ident_continue(c) {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Token::Ident(s)
    }

    fn read_number(&mut self) -> Result<Token, LexError> {
        let (start_line, start_col) = (self.line, self.col);
        let mut s = String::new();

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }

        // Only treat `.` as a decimal point if it's followed by a digit.
        let mut is_float = false;
        if self.peek_char() == Some('.') {
            let mut lookahead = self.chars.clone();
            lookahead.next(); // consume the '.'
            if let Some(&(_, next_c)) = lookahead.peek() {
                if next_c.is_ascii_digit() {
                    is_float = true;
                    s.push('.');
                    self.bump(); // consume '.'
                    while let Some(c) = self.peek_char() {
                        if c.is_ascii_digit() {
                            s.push(c);
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        if is_float {
            s.parse::<f64>()
                .map(Token::Float)
                .map_err(|e| LexError {
                    message: format!("invalid float literal '{s}': {e}"),
                    line: start_line,
                    col: start_col,
                })
        } else {
            s.parse::<i64>()
                .map(Token::Int)
                .map_err(|e| LexError {
                    message: format!("invalid int literal '{s}': {e}"),
                    line: start_line,
                    col: start_col,
                })
        }
    }

    fn read_string(&mut self) -> Result<Token, LexError> {
        let (start_line, start_col) = (self.line, self.col);
        self.bump(); // consume opening quote

        let mut s = String::new();
        loop {
            match self.bump() {
                None => {
                    return Err(LexError {
                        message: "unterminated string literal".to_string(),
                        line: start_line,
                        col: start_col,
                    })
                }
                Some('"') => break,
                Some('\\') => match self.bump() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some(other) => {
                        return Err(self.make_error(format!(
                            "unknown escape sequence '\\{other}' in string literal"
                        )))
                    }
                    None => {
                        return Err(LexError {
                            message: "unterminated string literal".to_string(),
                            line: start_line,
                            col: start_col,
                        })
                    }
                },
                Some(c) => s.push(c),
            }
        }

        Ok(Token::Str(s))
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn tokenize(source: &str) -> Result<Vec<SpannedToken>, LexError> {
    Lexer::new(source).tokenize()
}

// Ensure the lexer tests exist exactly as they did before, with standard checks.
#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Token> {
        tokenize(src)
            .unwrap()
            .into_iter()
            .map(|st| st.token)
            .collect()
    }

    #[test]
    fn brackets_lex() {
        assert_eq!(toks("[unsafe]"), vec![Token::LBracket, Token::Ident("unsafe".into()), Token::RBracket, Token::Eof]);
    }
}