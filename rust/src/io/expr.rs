//! Linear-expression parser for EDF channel selection.
//!
//! Port of the channel-derivation grammar in MATLAB
//! `DYNAM-O_dev/toolbox/helper_functions/EDF_toolbox/read_EDF.m:55–110`.
//!
//! Supports:
//!   - bare label: `"C3"`
//!   - rereference: `"C3-A1"`
//!   - mean of N channels: `"mean(A1, A2, A3)"`
//!   - arbitrary linear combination: `"(1/3)*C1 - 4*(C2-C3)/7"`
//!   - named reference assignment: `"M = mean(A1, A2)"` (consumed by [`parse_named`])
//!
//! **Brackets `[...]`** are grouping operators — exact aliases for `(`
//! and `)`. Use them anywhere you would use parens (the composer UI
//! emits `mean([C3-A2 - B], [C4-A1 - A])` which parses identically to
//! `mean((C3-A2 - B), (C4-A1 - A))`).
//!
//! **Dollar-escapes `$label$`** match a literal EDF channel name that
//! contains operator characters (`-`, spaces). Use `$C3-A2$` to mean
//! "the channel literally labelled `C3-A2`" rather than the expression
//! `C3 - A2`. This mirrors `read_EDF.m:133–135`.
//!
//! **Linearity rule** (matches `read_EDF.m:73–79`): each term in the
//! expression contains at most one signal-valued factor — any combination
//! of `signal * signal`, `signal / signal`, or `scalar / signal` is
//! rejected as nonlinear at parse time. Scalars (numeric literals) may be
//! multiplied or divided freely.
//!
//! AST is a flat sum of terms `coeff · signal_ref` where `signal_ref` is
//! either a file-channel name or a named-ref name. A scalar-only term
//! would have `signal_ref = None` but, like MATLAB, we reject scalars
//! mixed into a signal sum at parse time — see [`ExprError::ScalarTerm`].

use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug)]
pub enum ExprError {
    UnexpectedToken { pos: usize, found: String },
    UnexpectedEof,
    NonLinearMul { a: String, b: String },
    NonLinearDiv { a: String, b: String },
    ScalarOverSignal { b: String },
    ScalarTerm,
    UnknownSignal { name: String },
    CircularRef { name: String },
    Expected { expected: String, found: String, pos: usize },
    BadNamedRef { name: String },
    BadRefName { name: String },
    Empty,
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedToken { pos, found } => write!(f, "unexpected token at position {pos}: {found}"),
            Self::UnexpectedEof => write!(f, "unexpected end of expression"),
            Self::NonLinearMul { a, b } => write!(f, "nonlinear: cannot multiply two signals ({a} * {b})"),
            Self::NonLinearDiv { a, b } => write!(f, "nonlinear: cannot divide signal by signal ({a} / {b})"),
            Self::ScalarOverSignal { b } => write!(f, "nonlinear: cannot divide scalar by signal (1 / {b})"),
            Self::ScalarTerm => write!(f, "scalar-only term mixed into signal expression — expression must be a linear combination"),
            Self::UnknownSignal { name } => write!(f, "unknown signal: {name}"),
            Self::CircularRef { name } => write!(f, "circular reference detected involving '{name}'"),
            Self::Expected { expected, found, pos } => write!(f, "expected '{expected}' but got '{found}' at position {pos}"),
            Self::BadNamedRef { name } => write!(f, "named reference '{name}' must contain '=' between name and expression"),
            Self::BadRefName { name } => write!(f, "named reference name '{name}' is not a valid identifier"),
            Self::Empty => write!(f, "empty expression"),
        }
    }
}
impl std::error::Error for ExprError {}

/// One linear-combination term: `coeff · signal`. Scalar-only terms are
/// represented with `signal = None`, but they are rejected by [`parse`]
/// — they only exist transiently inside the parser.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub coeff: f64,
    pub signal: Option<SignalRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalRef {
    /// Reference to a literal EDF channel label.
    Leaf(String),
    /// Reference to a named derivation (resolved through [`resolve_references`]).
    Named(String),
}

impl SignalRef {
    pub fn name(&self) -> &str {
        match self { SignalRef::Leaf(s) | SignalRef::Named(s) => s }
    }
}

/// Parsed expression: a flat sum of terms. The order is not significant;
/// the evaluator sums them all.
#[derive(Debug, Clone)]
pub struct ExprAst {
    pub terms: Vec<Term>,
}

impl ExprAst {
    /// All distinct signal names appearing in the AST.
    pub fn signals(&self) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for t in &self.terms {
            if let Some(s) = &t.signal {
                let n = s.name();
                if seen.insert(n.to_string()) { out.push(n.to_string()); }
            }
        }
        out
    }
}

/// One named-reference assignment, e.g. `("M", parse("mean(A1, A2)")?)`.
pub type NamedRef = (String, ExprAst);

// ────────────────────────────── tokeniser ──────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Number(f64),
    Plus, Minus, Star, Slash,
    LParen, RParen,
    Comma,
    Eq,
}

fn tokenise(s: &str) -> Result<Vec<(usize, Tok)>, ExprError> {
    let bytes = s.as_bytes();
    let mut out: Vec<(usize, Tok)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        // Whitespace.
        if c.is_ascii_whitespace() { i += 1; continue; }
        // Single-character operators / punctuation. `[` and `]` are
        // aliases for `(` and `)` so the composer can wrap grouped
        // expressions in brackets (`mean([C3-A2 - B], …)`).
        let tok = match c {
            b'+' => Some(Tok::Plus),
            b'-' => Some(Tok::Minus),
            b'*' => Some(Tok::Star),
            b'/' => Some(Tok::Slash),
            b'(' | b'[' => Some(Tok::LParen),
            b')' | b']' => Some(Tok::RParen),
            b',' => Some(Tok::Comma),
            b'=' => Some(Tok::Eq),
            _ => None,
        };
        if let Some(t) = tok { out.push((i, t)); i += 1; continue; }
        // Number: digits, optional fractional, optional exponent.
        if c.is_ascii_digit() || (c == b'.' && i + 1 < bytes.len() && bytes[i+1].is_ascii_digit()) {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') { i += 1; }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') { i += 1; }
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            }
            let lit = &s[start..i];
            let v: f64 = lit.parse().map_err(|_| ExprError::UnexpectedToken {
                pos: start, found: lit.to_string(),
            })?;
            out.push((start, Tok::Number(v)));
            continue;
        }
        // `$LABEL$` escape (read_EDF.m:133–135) — literal channel name
        // that may contain operator characters like `-` or spaces. The
        // contents are taken verbatim (after interior trim) and emitted
        // as a single Ident token. Brackets `[...]` are NOT a literal
        // escape — they're grouping (handled above as LParen/RParen).
        if c == b'$' {
            let start = i + 1;
            i += 1;
            while i < bytes.len() && bytes[i] != b'$' { i += 1; }
            if i >= bytes.len() {
                return Err(ExprError::UnexpectedToken {
                    pos: start - 1, found: "$ without closing $".to_string(),
                });
            }
            let lit = s[start..i].trim();
            i += 1; // skip closing $
            out.push((start - 1, Tok::Ident(lit.to_string())));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let lit = &s[start..i];
            out.push((start, Tok::Ident(lit.to_string())));
            continue;
        }
        return Err(ExprError::UnexpectedToken {
            pos: i, found: (c as char).to_string(),
        });
    }
    Ok(out)
}

// ─────────────────────────────── parser ───────────────────────────────

struct Parser<'a> {
    toks: &'a [(usize, Tok)],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos).map(|(_, t)| t) }
    fn cur_pos(&self) -> usize { self.toks.get(self.pos).map(|(p, _)| *p).unwrap_or(usize::MAX) }
    fn bump(&mut self) -> Option<&Tok> {
        let t = self.toks.get(self.pos).map(|(_, t)| t);
        self.pos += 1;
        t
    }
    fn eat(&mut self, want: &Tok) -> Result<(), ExprError> {
        let got_pos = self.cur_pos();
        match self.peek() {
            Some(t) if std::mem::discriminant(t) == std::mem::discriminant(want) => {
                self.pos += 1;
                Ok(())
            }
            Some(other) => Err(ExprError::Expected {
                expected: format!("{:?}", want),
                found: format!("{:?}", other),
                pos: got_pos,
            }),
            None => Err(ExprError::UnexpectedEof),
        }
    }

    /// expr = term (('+'|'-') term)*
    fn expr(&mut self) -> Result<Vec<Term>, ExprError> {
        let mut terms = self.term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus)  => { self.bump(); let mut t = self.term()?; terms.append(&mut t); }
                Some(Tok::Minus) => { self.bump(); let mut t = self.term()?; for x in &mut t { x.coeff = -x.coeff; } terms.append(&mut t); }
                _ => break,
            }
        }
        Ok(terms)
    }

    /// term = factor (('*'|'/') factor)*
    fn term(&mut self) -> Result<Vec<Term>, ExprError> {
        let mut acc = self.factor()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.bump();
                    let rhs = self.factor()?;
                    acc = mul_terms(&acc, &rhs)?;
                }
                Some(Tok::Slash) => {
                    self.bump();
                    let rhs = self.factor()?;
                    acc = div_terms(&acc, &rhs)?;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    /// factor = number | unary-minus factor | ident | 'mean' '(' arglist ')' | '(' expr ')'
    fn factor(&mut self) -> Result<Vec<Term>, ExprError> {
        match self.peek().cloned() {
            Some(Tok::Number(v)) => { self.bump(); Ok(vec![Term { coeff: v, signal: None }]) }
            Some(Tok::Minus) => { self.bump(); let mut inner = self.factor()?; for t in &mut inner { t.coeff = -t.coeff; } Ok(inner) }
            Some(Tok::Plus)  => { self.bump(); self.factor() }
            Some(Tok::LParen) => {
                self.bump();
                let inner = self.expr()?;
                self.eat(&Tok::RParen)?;
                Ok(inner)
            }
            Some(Tok::Ident(name)) => {
                self.bump();
                if name.eq_ignore_ascii_case("mean") && matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump(); // (
                    let mut args: Vec<Vec<Term>> = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.expr()?);
                        while matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                            args.push(self.expr()?);
                        }
                    }
                    self.eat(&Tok::RParen)?;
                    if args.is_empty() {
                        return Err(ExprError::Empty);
                    }
                    // mean(a, b, c) = (a + b + c) / N
                    let n = args.len() as f64;
                    let mut out: Vec<Term> = Vec::new();
                    for a in args { for mut t in a { t.coeff /= n; out.push(t); } }
                    Ok(out)
                } else {
                    Ok(vec![Term { coeff: 1.0, signal: Some(SignalRef::Leaf(name)) }])
                }
            }
            Some(other) => Err(ExprError::UnexpectedToken {
                pos: self.cur_pos(), found: format!("{:?}", other),
            }),
            None => Err(ExprError::UnexpectedEof),
        }
    }
}

/// Multiply two linear-combinations. At most one side may carry a signal
/// in any pair of terms; otherwise the result would be nonlinear.
fn mul_terms(a: &[Term], b: &[Term]) -> Result<Vec<Term>, ExprError> {
    let mut out: Vec<Term> = Vec::new();
    for ta in a {
        for tb in b {
            match (&ta.signal, &tb.signal) {
                (Some(sa), Some(sb)) => {
                    return Err(ExprError::NonLinearMul {
                        a: sa.name().to_string(), b: sb.name().to_string(),
                    });
                }
                (Some(_), None) => out.push(Term { coeff: ta.coeff * tb.coeff, signal: ta.signal.clone() }),
                (None, Some(_)) => out.push(Term { coeff: ta.coeff * tb.coeff, signal: tb.signal.clone() }),
                (None, None)    => out.push(Term { coeff: ta.coeff * tb.coeff, signal: None }),
            }
        }
    }
    Ok(out)
}

/// Divide two linear-combinations. RHS must be scalar; LHS may be any
/// linear combination.
fn div_terms(a: &[Term], b: &[Term]) -> Result<Vec<Term>, ExprError> {
    // RHS must be a single scalar term.
    if b.len() != 1 || b[0].signal.is_some() {
        let rhs_name = b.first().and_then(|t| t.signal.as_ref()).map(|s| s.name().to_string()).unwrap_or_else(|| "<expr>".to_string());
        // If LHS contained any signal — flag scalar/signal; else generic.
        if a.iter().any(|t| t.signal.is_some()) {
            return Err(ExprError::NonLinearDiv { a: "<expr>".to_string(), b: rhs_name });
        } else {
            return Err(ExprError::ScalarOverSignal { b: rhs_name });
        }
    }
    let denom = b[0].coeff;
    if denom == 0.0 {
        return Err(ExprError::UnexpectedToken { pos: 0, found: "divide by zero".to_string() });
    }
    let mut out: Vec<Term> = Vec::with_capacity(a.len());
    for ta in a {
        out.push(Term { coeff: ta.coeff / denom, signal: ta.signal.clone() });
    }
    Ok(out)
}

/// Parse a channel expression.
pub fn parse(s: &str) -> Result<ExprAst, ExprError> {
    let s = s.trim();
    if s.is_empty() { return Err(ExprError::Empty); }
    let toks = tokenise(s)?;
    if toks.is_empty() { return Err(ExprError::Empty); }
    let mut p = Parser { toks: &toks, pos: 0 };
    let terms = p.expr()?;
    if p.pos != toks.len() {
        return Err(ExprError::UnexpectedToken {
            pos: p.cur_pos(),
            found: format!("{:?}", p.peek().unwrap()),
        });
    }
    // Linearity: combine duplicate signals. Also: if any term has a
    // signal AND there is any other signal-bearing term, scalar-only
    // terms mixed in are rejected (matches read_EDF.m).
    let has_signal = terms.iter().any(|t| t.signal.is_some());
    if has_signal {
        for t in &terms {
            if t.signal.is_none() && t.coeff != 0.0 {
                return Err(ExprError::ScalarTerm);
            }
        }
    }
    // Coalesce duplicates: sum coeffs for matching (signal name + kind).
    let mut combined: Vec<Term> = Vec::new();
    for t in terms {
        if let Some(idx) = combined.iter().position(|x| x.signal == t.signal) {
            combined[idx].coeff += t.coeff;
        } else {
            combined.push(t);
        }
    }
    // Drop zero-coefficient terms.
    combined.retain(|t| t.coeff != 0.0 || t.signal.is_none());
    Ok(ExprAst { terms: combined })
}

/// Parse a `"NAME = expr"` line into a [`NamedRef`].
pub fn parse_named(s: &str) -> Result<NamedRef, ExprError> {
    let s = s.trim();
    let Some(eq_idx) = s.find('=') else {
        return Err(ExprError::BadNamedRef { name: s.to_string() });
    };
    let name = s[..eq_idx].trim();
    let body = s[eq_idx + 1..].trim();
    if name.is_empty() {
        return Err(ExprError::BadRefName { name: String::new() });
    }
    // Identifier rule: alpha[alphanumeric_]*. Allow `$x$` escape.
    let is_ident = name.bytes().enumerate().all(|(i, b)| {
        if i == 0 { b.is_ascii_alphabetic() || b == b'_' }
        else { b.is_ascii_alphanumeric() || b == b'_' }
    });
    if !is_ident {
        return Err(ExprError::BadRefName { name: name.to_string() });
    }
    let ast = parse(body)?;
    Ok((name.to_string(), ast))
}

/// Topologically sort a list of named refs so each ref's dependencies
/// resolve before it. Detects cycles. The output preserves stable
/// ordering for independent refs.
pub fn resolve_references(refs: &[NamedRef]) -> Result<Vec<NamedRef>, ExprError> {
    let names: HashSet<String> = refs.iter().map(|(n, _)| n.clone()).collect();
    // Build dependency graph: each ref → set of named-ref deps.
    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    for (n, ast) in refs {
        let mut d: Vec<String> = Vec::new();
        for sig in ast.signals() {
            if names.contains(&sig) { d.push(sig); }
        }
        deps.insert(n.clone(), d);
    }
    // DFS topo sort.
    let mut state: HashMap<String, u8> = HashMap::new();  // 0=unseen 1=stack 2=done
    let mut order: Vec<String> = Vec::new();
    fn dfs(
        name: &str,
        deps: &HashMap<String, Vec<String>>,
        state: &mut HashMap<String, u8>,
        order: &mut Vec<String>,
    ) -> Result<(), ExprError> {
        match state.get(name) {
            Some(&2) => return Ok(()),
            Some(&1) => return Err(ExprError::CircularRef { name: name.to_string() }),
            _ => {}
        }
        state.insert(name.to_string(), 1);
        if let Some(ds) = deps.get(name) {
            for d in ds { dfs(d, deps, state, order)?; }
        }
        state.insert(name.to_string(), 2);
        order.push(name.to_string());
        Ok(())
    }
    for (n, _) in refs {
        dfs(n, &deps, &mut state, &mut order)?;
    }
    // Map name→ast for output.
    let map: HashMap<String, ExprAst> = refs.iter().cloned().collect();
    Ok(order.into_iter().filter_map(|n| map.get(&n).map(|a| (n.clone(), a.clone()))).collect())
}

/// Evaluate the AST against a signal table. Each signal slice must have
/// the same length; the output is that length. Returns
/// `ExprError::UnknownSignal` if any referenced name is missing.
pub fn evaluate(ast: &ExprAst, signals: &HashMap<String, Vec<f64>>) -> Result<Vec<f64>, ExprError> {
    // Determine output length and consistency.
    let mut n_out: Option<usize> = None;
    for t in &ast.terms {
        if let Some(sig) = &t.signal {
            let name = sig.name();
            let data = signals.get(name).ok_or_else(|| ExprError::UnknownSignal { name: name.to_string() })?;
            match n_out {
                None => n_out = Some(data.len()),
                Some(n) if n != data.len() => {
                    return Err(ExprError::UnknownSignal {
                        name: format!("{} has length {} but expected {}", name, data.len(), n),
                    });
                }
                _ => {}
            }
        }
    }
    let n = n_out.unwrap_or(0);
    let mut out: Vec<f64> = vec![0.0; n];
    for t in &ast.terms {
        if let Some(sig) = &t.signal {
            let data = signals.get(sig.name()).unwrap();
            for (o, &x) in out.iter_mut().zip(data.iter()) { *o += t.coeff * x; }
        }
        // scalar-only terms are not reachable here (parse rejects them
        // when mixed with signals; pure-scalar expressions don't
        // produce a signal vector at all).
    }
    Ok(out)
}

// ─────────────────────────────── tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn one_sig(coeff: f64, name: &str) -> Term {
        Term { coeff, signal: Some(SignalRef::Leaf(name.to_string())) }
    }

    #[test]
    fn parses_bare_label() {
        let ast = parse("C3").unwrap();
        assert_eq!(ast.terms, vec![one_sig(1.0, "C3")]);
    }
    #[test]
    fn parses_diff() {
        let ast = parse("A - B").unwrap();
        assert_eq!(ast.terms, vec![one_sig(1.0, "A"), one_sig(-1.0, "B")]);
    }
    #[test]
    fn parses_mean3() {
        let ast = parse("mean(A, B, C)").unwrap();
        for name in &["A", "B", "C"] {
            let t = ast.terms.iter().find(|t| t.signal.as_ref().map(|s| s.name()) == Some(name)).unwrap();
            assert!((t.coeff - 1.0/3.0).abs() < 1e-12);
        }
    }
    #[test]
    fn parses_linear_combo() {
        let ast = parse("(1/3)*C1 - 4*(C2-C3)/7").unwrap();
        let c1 = ast.terms.iter().find(|t| t.signal.as_ref().map(|s| s.name()) == Some("C1")).unwrap();
        let c2 = ast.terms.iter().find(|t| t.signal.as_ref().map(|s| s.name()) == Some("C2")).unwrap();
        let c3 = ast.terms.iter().find(|t| t.signal.as_ref().map(|s| s.name()) == Some("C3")).unwrap();
        assert!((c1.coeff - 1.0/3.0).abs() < 1e-12);
        assert!((c2.coeff + 4.0/7.0).abs() < 1e-12);
        assert!((c3.coeff - 4.0/7.0).abs() < 1e-12);
    }
    #[test]
    fn rejects_signal_times_signal() {
        assert!(matches!(parse("A * B"), Err(ExprError::NonLinearMul { .. })));
    }
    #[test]
    fn rejects_signal_div_signal() {
        assert!(matches!(parse("A / B"), Err(ExprError::NonLinearDiv { .. })));
    }
    #[test]
    fn rejects_scalar_over_signal() {
        assert!(matches!(parse("1 / A"), Err(ExprError::ScalarOverSignal { .. })));
    }
    #[test]
    fn rejects_scalar_mixed_into_signal_sum() {
        assert!(matches!(parse("A + 1"), Err(ExprError::ScalarTerm)));
    }
    #[test]
    fn parses_named_ref() {
        let (name, ast) = parse_named("M = mean(A1, A2)").unwrap();
        assert_eq!(name, "M");
        let a1 = ast.terms.iter().find(|t| t.signal.as_ref().map(|s| s.name()) == Some("A1")).unwrap();
        assert!((a1.coeff - 0.5).abs() < 1e-12);
    }
    #[test]
    fn topo_sorts_refs() {
        let r1 = ("M".to_string(), parse("mean(A1, A2)").unwrap());
        let r2 = ("D".to_string(), parse("C3 - M").unwrap());
        let sorted = resolve_references(&[r2.clone(), r1.clone()]).unwrap();
        // M must come before D.
        let pos_m = sorted.iter().position(|(n, _)| n == "M").unwrap();
        let pos_d = sorted.iter().position(|(n, _)| n == "D").unwrap();
        assert!(pos_m < pos_d);
    }
    #[test]
    fn detects_cycle() {
        let r1 = ("A".to_string(), parse("B").unwrap());
        let r2 = ("B".to_string(), parse("A").unwrap());
        assert!(matches!(resolve_references(&[r1, r2]), Err(ExprError::CircularRef { .. })));
    }
    #[test]
    fn evaluates_diff() {
        let ast = parse("A - B").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("A".into(), vec![10.0, 20.0, 30.0]);
        sigs.insert("B".into(), vec![1.0, 2.0, 3.0]);
        let out = evaluate(&ast, &sigs).unwrap();
        assert_eq!(out, vec![9.0, 18.0, 27.0]);
    }
    #[test]
    fn brackets_are_grouping() {
        // `[expr]` is an alias for `(expr)` — contents are parsed as a
        // sub-expression, not as a literal label.
        let ast = parse("[C3 - A2]").unwrap();
        assert_eq!(ast.terms, vec![one_sig(1.0, "C3"), one_sig(-1.0, "A2")]);
    }
    #[test]
    fn dollar_escapes_literal_label() {
        // Use `$label$` to refer to a channel whose literal name
        // contains '-' or other operator characters.
        let ast = parse("$C3-A2$").unwrap();
        assert_eq!(ast.terms, vec![one_sig(1.0, "C3-A2")]);
    }
    #[test]
    fn parses_users_grouped_mean() {
        // The user's example: brackets group three-channel differences
        // inside each mean argument.
        let ast = parse("mean([C3 - A2 - B], [C4 - A1 - A])").unwrap();
        // Expected: mean(C3 - A2 - B, C4 - A1 - A)
        //   = 1/2 C3  - 1/2 A2  - 1/2 B  + 1/2 C4  - 1/2 A1  - 1/2 A
        let expected: std::collections::HashMap<&str, f64> = [
            ("C3", 0.5), ("A2", -0.5), ("B", -0.5),
            ("C4", 0.5), ("A1", -0.5), ("A", -0.5),
        ].into_iter().collect();
        for t in &ast.terms {
            let n = t.signal.as_ref().unwrap().name();
            let e = expected.get(n).copied().unwrap_or(0.0);
            assert!((t.coeff - e).abs() < 1e-12, "{} coeff {} != {}", n, t.coeff, e);
        }
        assert_eq!(ast.terms.len(), 6);
    }
    #[test]
    fn parses_mean_of_dollar_escaped_labels() {
        let ast = parse("mean($C3-A2$, $C4-A1$)").unwrap();
        let names: Vec<&str> = ast.terms.iter()
            .filter_map(|t| t.signal.as_ref().map(|s| s.name())).collect();
        assert!(names.contains(&"C3-A2"));
        assert!(names.contains(&"C4-A1"));
        for t in &ast.terms {
            assert!((t.coeff - 0.5).abs() < 1e-12);
        }
    }
    #[test]
    fn parses_dollar_label_with_spaces() {
        let ast = parse("$ C3-A2 $").unwrap();
        assert_eq!(ast.terms, vec![one_sig(1.0, "C3-A2")]);
    }
    #[test]
    fn evaluates_mean() {
        let ast = parse("mean(A, B, C)").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("A".into(), vec![3.0, 6.0]);
        sigs.insert("B".into(), vec![6.0, 12.0]);
        sigs.insert("C".into(), vec![9.0, 18.0]);
        let out = evaluate(&ast, &sigs).unwrap();
        let e0 = (3.0 + 6.0 + 9.0) / 3.0;
        let e1 = (6.0 + 12.0 + 18.0) / 3.0;
        assert!((out[0] - e0).abs() < 1e-12);
        assert!((out[1] - e1).abs() < 1e-12);
    }
}
