//! Linear-expression parser for EDF channel selection.
//!
//! Port of the channel-derivation grammar in MATLAB
//! `DYNAM-O/toolbox/helper_functions/EDF_toolbox/read_EDF.m:55–110`.
//!
//! Supports:
//!   - bare label: `"C3"`
//!   - rereference: `"C3-A1"`
//!   - mean of N channels: `"mean(A1, A2, A3)"`
//!   - arbitrary linear combination: `"(1/3)*C1 - 4*(C2-C3)/7"`
//!   - named reference assignment: `"M = mean(A1, A2)"` (consumed by [`parse_named`])
//!
//! **`mean(...)` averages the channels that are available.** Cohorts often
//! name the same montage differently across files, so a reference like
//! `M = mean(C3-A2, [C3-A2 - B])` lists every spelling; per file, the mean
//! is taken over the arguments whose signals that file actually has
//! (renormalized `1/M` over the M available ones) and fails only when none
//! are. An argument counts as available only when every signal inside it
//! is. Availability is resolved at evaluation, so the mean survives
//! parsing as a structural [`SignalRef::Mean`] node; when all N arguments
//! are present the result is identical to the plain `(a + … )/N`
//! desugaring, bit for bit.
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

#[derive(Debug, Clone, PartialEq)]
pub enum SignalRef {
    /// Reference to a literal EDF channel label.
    Leaf(String),
    /// Reference to a named derivation (resolved through [`resolve_references`]).
    Named(String),
    /// `mean(arg, …)` kept structural so evaluation can average over the
    /// arguments actually available in the file (see module doc). Each
    /// argument is a full sub-expression.
    Mean(Vec<ExprAst>),
}

impl SignalRef {
    pub fn name(&self) -> &str {
        match self {
            SignalRef::Leaf(s) | SignalRef::Named(s) => s,
            // Only ever surfaces in error text — a mean has no single name;
            // its constituent signals are reported via `signals()`.
            SignalRef::Mean(_) => "mean(…)",
        }
    }
}

/// Parsed expression: a flat sum of terms. The order is not significant;
/// the evaluator sums them all.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprAst {
    pub terms: Vec<Term>,
}

impl ExprAst {
    /// All distinct signal names appearing in the AST, mean arguments
    /// included (they must be loaded/declared even though a file may
    /// satisfy only a subset of them).
    pub fn signals(&self) -> Vec<String> {
        fn walk(terms: &[Term], seen: &mut HashSet<String>, out: &mut Vec<String>) {
            for t in terms {
                match &t.signal {
                    Some(SignalRef::Mean(args)) => {
                        for a in args { walk(&a.terms, seen, out); }
                    }
                    Some(s) => {
                        let n = s.name();
                        if seen.insert(n.to_string()) { out.push(n.to_string()); }
                    }
                    None => {}
                }
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        walk(&self.terms, &mut seen, &mut out);
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
                    // All-scalar mean folds to a number; a mean over
                    // signals stays a structural node so evaluation can
                    // average the AVAILABLE arguments (module doc). An
                    // argument mixing a signal with a bare scalar is
                    // rejected like any other mixed sum.
                    if args.iter().all(|a| a.iter().all(|t| t.signal.is_none())) {
                        let n = args.len() as f64;
                        let sum: f64 = args.iter().flatten().map(|t| t.coeff).sum();
                        return Ok(vec![Term { coeff: sum / n, signal: None }]);
                    }
                    for a in &args {
                        let has_signal = a.iter().any(|t| t.signal.is_some());
                        if !has_signal || a.iter().any(|t| t.signal.is_none() && t.coeff != 0.0) {
                            return Err(ExprError::ScalarTerm);
                        }
                    }
                    let args: Vec<ExprAst> =
                        args.into_iter().map(|t| ExprAst { terms: t }).collect();
                    Ok(vec![Term { coeff: 1.0, signal: Some(SignalRef::Mean(args)) }])
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
    Ok(ExprAst { terms: coalesce(terms) })
}

/// Coalesce duplicates (sum coeffs for matching signals, first-occurrence
/// order) and drop zero-coefficient signal terms. Shared by [`parse`] and
/// [`flatten`] so the availability-resolved term list matches what the
/// old parse-time desugaring produced, term for term.
fn coalesce(terms: Vec<Term>) -> Vec<Term> {
    let mut combined: Vec<Term> = Vec::new();
    for t in terms {
        if let Some(idx) = combined.iter().position(|x| x.signal == t.signal) {
            combined[idx].coeff += t.coeff;
        } else {
            combined.push(t);
        }
    }
    combined.retain(|t| t.coeff != 0.0 || t.signal.is_none());
    combined
}

/// Resolve every `mean(...)` in `ast` against the signals actually
/// available, producing a flat `Leaf`/`Named`-only term list.
///
/// Each mean argument is kept iff every signal inside it (recursively) is
/// in `available`; the kept arguments are renormalized `1/M`. A mean with
/// no available argument is the only error. Signals OUTSIDE a mean are
/// not checked here — [`evaluate`] reports those as `UnknownSignal`, so
/// legacy behavior for non-mean expressions is unchanged.
///
/// When all N arguments are available the output is exactly the old
/// parse-time desugaring — same term order, same `coeff / N` arithmetic —
/// so previously-working selections stay bit-identical.
pub fn flatten(ast: &ExprAst, available: &HashSet<String>) -> Result<Vec<Term>, ExprError> {
    fn all_available(terms: &[Term], available: &HashSet<String>) -> bool {
        terms.iter().all(|t| match &t.signal {
            Some(s) => available.contains(s.name()),
            None => true,
        })
    }
    fn walk(terms: &[Term], available: &HashSet<String>, out: &mut Vec<Term>) -> Result<(), ExprError> {
        for t in terms {
            match &t.signal {
                Some(SignalRef::Mean(args)) => {
                    let mut kept: Vec<Vec<Term>> = Vec::new();
                    let mut wanted: Vec<String> = Vec::new();
                    for a in args {
                        wanted.extend(a.signals());
                        let mut f: Vec<Term> = Vec::new();
                        if walk(&a.terms, available, &mut f).is_ok() && all_available(&f, available) {
                            kept.push(f);
                        }
                    }
                    if kept.is_empty() {
                        return Err(ExprError::UnknownSignal {
                            name: format!(
                                "mean(): none of its channels are available (wanted any of: {})",
                                wanted.join(", ")
                            ),
                        });
                    }
                    let m = kept.len() as f64;
                    for f in kept {
                        for ft in f {
                            // Ordering matters for bit-parity with the old
                            // desugar: divide by M first, then apply the
                            // outer coefficient.
                            out.push(Term { coeff: (ft.coeff / m) * t.coeff, signal: ft.signal });
                        }
                    }
                }
                _ => out.push(t.clone()),
            }
        }
        Ok(())
    }
    let mut out: Vec<Term> = Vec::new();
    walk(&ast.terms, available, &mut out)?;
    Ok(coalesce(out))
}

/// Can `ast` be evaluated against these signal names? Mirrors
/// [`evaluate`]'s availability rules: every signal outside a mean must be
/// present, and each mean needs at least one fully-available argument.
pub fn satisfiable(ast: &ExprAst, available: &HashSet<String>) -> bool {
    match flatten(ast, available) {
        Ok(flat) => flat.iter().all(|t| match &t.signal {
            Some(s) => available.contains(s.name()),
            None => true,
        }),
        Err(_) => false,
    }
}

/// Whether a channel expression can be evaluated against a file exposing
/// `labels`, resolving named references first: a reference joins the
/// available set iff its own expression is satisfiable (in dependency
/// order — pass the [`resolve_references`] output), so a mean over an
/// unavailable reference simply skips it. This is the shared
/// availability rule for wizard coverage counts and expression
/// validators; it must match what [`evaluate`] accepts at run time.
pub fn channel_satisfiable(
    channel: &ExprAst,
    ordered_refs: &[NamedRef],
    labels: &HashSet<String>,
) -> bool {
    let mut avail = labels.clone();
    for (name, ast) in ordered_refs {
        if satisfiable(ast, &avail) {
            avail.insert(name.clone());
        }
    }
    satisfiable(channel, &avail)
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
/// Walk a list of channel expressions and named references, collect every
/// **EDF leaf label** that needs to be loaded from disk to evaluate them.
///
/// Channel expressions like `[C3 - LM]` look like they just need `C3` and
/// `LM`, but `LM` is a named reference (e.g. `LM = mean(M1, M2)`) — so
/// `M1` and `M2` are the real leaves. The dispatcher needs to know every
/// underlying EDF channel up front, or the partial-load `read_edf_signals`
/// call won't bring `M1` / `M2` into memory and `select_channel_with_refs`
/// later fails with `unknown signal: M1`.
///
/// `channels` are the cohort's channel expressions (raw user input — not
/// the output aliases). `references` are the named-reference entries in
/// the `"name = expression"` form persisted to `batch_settings.json`.
///
/// On parse error in any reference or channel the function returns the
/// error wrapped with which entry was being parsed. Circular references
/// are detected by [`resolve_references`] and propagate as a parse error.
///
/// Returns leaves in stable order (first-seen wins), so downstream callers
/// that use the result as `wanted_labels` get deterministic load order.
pub fn collect_leaf_labels(
    channels: &[&str],
    references: &[&str],
) -> Result<Vec<String>, ExprError> {
    use std::collections::HashSet;

    // Parse all references → name -> AST map, plus a name set so the
    // walker can tell a Leaf-typed name that's actually a Named ref
    // from a real EDF leaf.
    let mut ref_asts: HashMap<String, ExprAst> = HashMap::new();
    for r in references {
        let (name, ast) = parse_named(r)?;
        ref_asts.insert(name, ast);
    }
    let ref_names: HashSet<String> = ref_asts.keys().cloned().collect();

    // Recursively walk an AST, collecting leaves. Tracks a `visiting`
    // set so a cycle (already prevented by resolve_references on the
    // graph level) doesn't recurse forever if a caller passes weird
    // input. Identifiers that match a reference name expand via
    // ref_asts; everything else is a real EDF leaf.
    fn walk(
        ast: &ExprAst,
        ref_asts: &HashMap<String, ExprAst>,
        ref_names: &HashSet<String>,
        out: &mut Vec<String>,
        seen: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
    ) {
        for term in &ast.terms {
            let Some(sig) = &term.signal else { continue };
            // A mean has no name of its own — its arguments carry the
            // leaves (all of them: a partial file loads what it has).
            if let SignalRef::Mean(args) = sig {
                for a in args {
                    walk(a, ref_asts, ref_names, out, seen, visiting);
                }
                continue;
            }
            let name = sig.name().to_string();
            if ref_names.contains(&name) {
                if !visiting.insert(name.clone()) {
                    // cycle — bail (the leaves collected so far are
                    // still valid for the non-cyclic siblings)
                    continue;
                }
                if let Some(inner) = ref_asts.get(&name) {
                    walk(inner, ref_asts, ref_names, out, seen, visiting);
                }
                visiting.remove(&name);
            } else if seen.insert(name.clone()) {
                out.push(name);
            }
        }
    }

    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut visiting: HashSet<String> = HashSet::new();

    for ch in channels {
        match parse(ch) {
            Ok(ast) => walk(&ast, &ref_asts, &ref_names, &mut out, &mut seen, &mut visiting),
            // A channel that doesn't parse as an expression is treated as a
            // single literal EDF label — e.g. a referential-montage name with
            // a space (`"O2 A1"`, `"C3 A2"`) or a bracketed label. Loading the
            // raw string as a leaf lets `select_channel_with_refs`'s exact-
            // label fast-path resolve it; a genuine typo simply fails later
            // with a clear `unknown signal` at evaluation instead of blocking
            // the whole batch here at parse time.
            Err(_) => {
                let name = ch.trim().to_string();
                if !name.is_empty() && seen.insert(name.clone()) {
                    out.push(name);
                }
            }
        }
    }
    // Also include leaves from references that aren't reached by any
    // channel — defensive: lets the user define a reference now and a
    // channel that uses it later without surprising load behaviour.
    for ast in ref_asts.values() {
        walk(ast, &ref_asts, &ref_names, &mut out, &mut seen, &mut visiting);
    }

    Ok(out)
}

pub fn evaluate(ast: &ExprAst, signals: &HashMap<String, Vec<f64>>) -> Result<Vec<f64>, ExprError> {
    // Resolve mean(...) nodes against what this signal table actually
    // holds — the mean of the available arguments (module doc). After
    // this, every term is a plain Leaf/Named reference.
    let available: HashSet<String> = signals.keys().cloned().collect();
    let flat = flatten(ast, &available)?;
    // Determine output length and consistency.
    let mut n_out: Option<usize> = None;
    for t in &flat {
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
    for t in &flat {
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

    /// Flatten with every signal in the AST available — the all-present
    /// view, which must match the historical parse-time desugaring.
    fn flat_all(ast: &ExprAst) -> Vec<Term> {
        let avail: HashSet<String> = ast.signals().into_iter().collect();
        flatten(ast, &avail).unwrap()
    }

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
            let t = flat_all(&ast).into_iter().find(|t| t.signal.as_ref().map(|s| s.name().to_string()) == Some(name.to_string())).unwrap();
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
        let a1 = flat_all(&ast).into_iter().find(|t| t.signal.as_ref().map(|s| s.name().to_string()) == Some("A1".to_string())).unwrap();
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
        // Expected (all channels available): mean(C3 - A2 - B, C4 - A1 - A)
        //   = 1/2 C3  - 1/2 A2  - 1/2 B  + 1/2 C4  - 1/2 A1  - 1/2 A
        let expected: std::collections::HashMap<&str, f64> = [
            ("C3", 0.5), ("A2", -0.5), ("B", -0.5),
            ("C4", 0.5), ("A1", -0.5), ("A", -0.5),
        ].into_iter().collect();
        let flat = flat_all(&ast);
        for t in &flat {
            let n = t.signal.as_ref().unwrap().name().to_string();
            let e = expected.get(n.as_str()).copied().unwrap_or(0.0);
            assert!((t.coeff - e).abs() < 1e-12, "{} coeff {} != {}", n, t.coeff, e);
        }
        assert_eq!(flat.len(), 6);
    }
    #[test]
    fn parses_mean_of_dollar_escaped_labels() {
        let ast = parse("mean($C3-A2$, $C4-A1$)").unwrap();
        let names = ast.signals();
        assert!(names.contains(&"C3-A2".to_string()));
        assert!(names.contains(&"C4-A1".to_string()));
        for t in flat_all(&ast) {
            assert!((t.coeff - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn mean_averages_only_available_channels() {
        // The heterogeneous-cohort case: a mean listing every spelling of
        // a montage averages the ones this file actually has.
        let ast = parse("mean(A, B, C, D)").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("A".into(), vec![2.0, 4.0]);
        sigs.insert("C".into(), vec![6.0, 8.0]);
        let v = evaluate(&ast, &sigs).unwrap();
        assert_eq!(v, vec![4.0, 6.0]); // (A + C) / 2
    }

    #[test]
    fn mean_argument_needs_all_its_signals() {
        // An argument is available only when EVERY signal inside it is:
        // `C3-A2` with A2 missing must not contribute a half-referenced C3.
        let ast = parse("mean(C3 - A2, O1)").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("C3".into(), vec![10.0]);
        sigs.insert("O1".into(), vec![4.0]);
        let v = evaluate(&ast, &sigs).unwrap();
        assert_eq!(v, vec![4.0]); // only O1 qualifies
    }

    #[test]
    fn mean_with_no_available_channels_errors() {
        let ast = parse("mean(A, B)").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("X".into(), vec![1.0]);
        let e = evaluate(&ast, &sigs).unwrap_err();
        assert!(e.to_string().contains("none of its channels are available"), "{e}");
    }

    #[test]
    fn mean_all_present_is_bitwise_the_desugared_form() {
        let mean_ast = parse("mean(A, B, C)").unwrap();
        let sum_ast = parse("(A + B + C)/3").unwrap();
        let mut sigs: HashMap<String, Vec<f64>> = HashMap::new();
        sigs.insert("A".into(), vec![0.1, 1.0e-17, 7.3]);
        sigs.insert("B".into(), vec![0.2, 2.0e+13, -1.1]);
        sigs.insert("C".into(), vec![0.3, 3.7e-5, 0.0]);
        let a = evaluate(&mean_ast, &sigs).unwrap();
        let b = evaluate(&sum_ast, &sigs).unwrap();
        assert_eq!(a, b); // exact — same coefficients, same summation order
    }

    #[test]
    fn scalar_only_mean_folds_to_a_number() {
        let ast = parse("mean(1, 2)").unwrap();
        assert_eq!(ast.terms.len(), 1);
        assert!(ast.terms[0].signal.is_none());
        assert!((ast.terms[0].coeff - 1.5).abs() < 1e-12);
    }

    #[test]
    fn mean_mixing_signal_and_scalar_arg_is_rejected() {
        assert!(matches!(parse("mean(A, 2)"), Err(ExprError::ScalarTerm)));
    }

    #[test]
    fn channel_satisfiable_resolves_refs_adaptively() {
        // M lists every montage spelling; a file carrying just one of
        // them still satisfies both M and the channel that rereferences
        // against it. A file with none satisfies neither.
        let m = parse_named("M = mean($C3-A2$, $[C3-A2 - B]$)").unwrap();
        let refs = resolve_references(&[m]).unwrap();
        let ch = parse("$C3-A2$ - M").unwrap();
        let have: HashSet<String> = ["C3-A2".to_string()].into_iter().collect();
        assert!(channel_satisfiable(&ch, &refs, &have));
        let none: HashSet<String> = ["Fpz".to_string()].into_iter().collect();
        assert!(!channel_satisfiable(&ch, &refs, &none));
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

    #[test]
    fn collect_leaf_labels_expands_named_references() {
        // The user's CFS bug: LM = mean(M1, M2); channels reference LM.
        // The dispatcher needs to load M1 and M2 in addition to C3/C4.
        let channels = ["[C3 - LM]", "[C4 - LM]", "-LM"];
        let references = ["LM = mean(M1, M2)"];
        let leaves = collect_leaf_labels(&channels, &references).unwrap();
        // Order: C3 (from channel 1), then M1, M2 (from LM expansion),
        // then C4 (from channel 2). LM itself is NOT a leaf — it's a
        // named ref. -LM contributes no new leaves (M1/M2 already in).
        assert!(leaves.contains(&"C3".to_string()));
        assert!(leaves.contains(&"C4".to_string()));
        assert!(leaves.contains(&"M1".to_string()));
        assert!(leaves.contains(&"M2".to_string()));
        assert!(!leaves.contains(&"LM".to_string()));
    }

    #[test]
    fn collect_leaf_labels_handles_nested_references() {
        // LM = mean(M1, M2); BIG = mean(LM, C3); channel [C3 - BIG].
        // BIG's leaves resolve through LM to {M1, M2, C3}.
        let channels = ["[C3 - BIG]"];
        let references = ["LM = mean(M1, M2)", "BIG = mean(LM, C3)"];
        let leaves = collect_leaf_labels(&channels, &references).unwrap();
        assert!(leaves.contains(&"C3".to_string()));
        assert!(leaves.contains(&"M1".to_string()));
        assert!(leaves.contains(&"M2".to_string()));
        assert!(!leaves.contains(&"LM".to_string()));
        assert!(!leaves.contains(&"BIG".to_string()));
    }

    #[test]
    fn collect_leaf_labels_dedupes_repeated_leaves() {
        let channels = ["C3", "[C3 - LM]", "-LM"];
        let references = ["LM = mean(M1, M2)"];
        let leaves = collect_leaf_labels(&channels, &references).unwrap();
        // C3 appears in two channel expressions but only once in the
        // leaf list.
        let c3_count = leaves.iter().filter(|s| *s == "C3").count();
        assert_eq!(c3_count, 1);
    }

    #[test]
    fn collect_leaf_labels_bare_literal_channels_unchanged() {
        // No references defined — the leaves are literally the channel
        // expressions.
        let channels = ["C3", "C4", "O1"];
        let references: [&str; 0] = [];
        let leaves = collect_leaf_labels(&channels, &references).unwrap();
        assert_eq!(leaves, vec!["C3", "C4", "O1"]);
    }

    #[test]
    fn collect_leaf_labels_treats_unparseable_channel_as_literal_label() {
        // Referential-montage labels with a space (`O2 A1`, `C3 A2`) don't
        // parse as expressions (two idents, no operator) — they must be
        // collected verbatim as literal leaves so the partial EDF load brings
        // the real signal into memory and the eval fast-path resolves it.
        let channels = ["O2 A1", "C3 A2"];
        let references: [&str; 0] = [];
        let leaves = collect_leaf_labels(&channels, &references).unwrap();
        assert_eq!(leaves, vec!["O2 A1", "C3 A2"]);
    }
}
