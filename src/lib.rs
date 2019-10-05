//! THIS IS SUPER UNSTABLE
//! THIS IS PRIMARIALY INTENDED AS AN EXPERIMENT

#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]

extern crate proc_macro;

use std::cell::RefCell;
use std::cmp;
use std::fmt;
use std::iter;
use std::ops::RangeBounds;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::rc::Rc;
use std::vec;

use proc_macro::{Delimiter, Spacing, LineColumn};
use proc_macro::bridge::{server, TokenTree};

mod lexer;

type TokenTreeT = TokenTree<Group, Punct, Ident, Literal>;

fn set_tt_span(tt: &mut TokenTreeT, span: Span) {
    match tt {
        TokenTree::Group(t) => t.set_span(span),
        TokenTree::Literal(t) => t.set_span(span),
        TokenTree::Ident(t) => t.set_span(span),
        TokenTree::Punct(t) => t.set_span(span),
    }
}

pub struct TokenStreamBuilder {
    inner: Vec<TokenTreeT>,
}

#[derive(Clone)]
pub struct TokenStream {
    inner: Vec<TokenTreeT>,
}

impl TokenStream {
    pub fn new() -> TokenStream {
        TokenStream { inner: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
}

fn add_to_source_map(src: &str) -> u32 {
    // Create a dummy file & add it to the source map
    SOURCE_MAP.with(|cm| {
        let mut cm = cm.borrow_mut();
        let name = format!("<parsed string {}>", cm.files.len());
        let span = cm.add_file(&name, src);
        span.lo
    })
}

impl FromStr for TokenStream {
    type Err = lexer::LexError;

    fn from_str(src: &str) -> Result<TokenStream, lexer::LexError> {
        let start_off = add_to_source_map(src);
        lexer::lex_stream(src, start_off)
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut joint = false;
        for (i, tt) in self.inner.iter().enumerate() {
            if i != 0 && !joint {
                write!(f, " ")?;
            }
            joint = false;
            match *tt {
                TokenTree::Group(ref tt) => {
                    let (start, end) = match tt.delimiter() {
                        Delimiter::Parenthesis => ("(", ")"),
                        Delimiter::Brace => ("{", "}"),
                        Delimiter::Bracket => ("[", "]"),
                        Delimiter::None => ("", ""),
                    };
                    if tt.stream().into_iter().next().is_none() {
                        write!(f, "{} {}", start, end)?
                    } else {
                        write!(f, "{} {} {}", start, tt.stream(), end)?
                    }
                }
                TokenTree::Ident(ref tt) => write!(f, "{}", tt)?,
                TokenTree::Punct(ref tt) => {
                    write!(f, "{}", tt.as_char())?;
                    match tt.spacing() {
                        Spacing::Alone => {}
                        Spacing::Joint => joint = true,
                    }
                }
                TokenTree::Literal(ref tt) => write!(f, "{}", tt)?,
            }
        }

        Ok(())
    }
}

impl From<TokenTreeT> for TokenStream {
    fn from(tree: TokenTreeT) -> TokenStream {
        TokenStream { inner: vec![tree] }
    }
}

impl iter::FromIterator<TokenTreeT> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTreeT>>(streams: I) -> Self {
        let mut v = Vec::new();

        for token in streams.into_iter() {
            v.push(token);
        }

        TokenStream { inner: v }
    }
}

impl iter::FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        let mut v = Vec::new();

        for stream in streams.into_iter() {
            v.extend(stream.inner);
        }

        TokenStream { inner: v }
    }
}

impl Extend<TokenTreeT> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenTreeT>>(&mut self, streams: I) {
        self.inner.extend(streams);
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
        self.inner
            .extend(streams.into_iter().flat_map(|stream| stream));
    }
}

pub type TokenStreamIter = vec::IntoIter<TokenTreeT>;

impl IntoIterator for TokenStream {
    type Item = TokenTreeT;
    type IntoIter = TokenStreamIter;

    fn into_iter(self) -> TokenStreamIter {
        self.inner.into_iter()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct SourceFile {
    path: PathBuf,
}

impl SourceFile {
    /// Get the path to this source file as a string.
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn is_real(&self) -> bool {
        // XXX(nika): Support real files in the future?
        false
    }
}

impl fmt::Debug for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SourceFile")
            .field("path", &self.path())
            .field("is_real", &self.is_real())
            .finish()
    }
}

thread_local! {
    static SOURCE_MAP: RefCell<SourceMap> = RefCell::new(SourceMap {
        // NOTE: We start with a single dummy file which all call_site() and
        // def_site() spans reference.
        files: vec![{
            {
                FileInfo {
                    name: "<unspecified>".to_owned(),
                    span: Span { lo: 0, hi: 0 },
                    lines: vec![0],
                }
            }
        }],
    });
}

struct FileInfo {
    name: String,
    span: Span,
    lines: Vec<usize>,
}

impl FileInfo {
    fn offset_line_column(&self, offset: usize) -> LineColumn {
        assert!(self.span_within(Span {
            lo: offset as u32,
            hi: offset as u32
        }));
        let offset = offset - self.span.lo as usize;
        match self.lines.binary_search(&offset) {
            Ok(found) => LineColumn {
                line: found + 1,
                column: 0,
            },
            Err(idx) => LineColumn {
                line: idx,
                column: offset - self.lines[idx - 1],
            },
        }
    }

    fn span_within(&self, span: Span) -> bool {
        span.lo >= self.span.lo && span.hi <= self.span.hi
    }
}

/// Computesthe offsets of each line in the given source string.
fn lines_offsets(s: &str) -> Vec<usize> {
    let mut lines = vec![0];
    let mut prev = 0;
    while let Some(len) = s[prev..].find('\n') {
        prev += len + 1;
        lines.push(prev);
    }
    lines
}

struct SourceMap {
    files: Vec<FileInfo>,
}

impl SourceMap {
    fn next_start_pos(&self) -> u32 {
        // Add 1 so there's always space between files.
        //
        // We'll always have at least 1 file, as we initialize our files list
        // with a dummy file.
        self.files.last().unwrap().span.hi + 1
    }

    fn add_file(&mut self, name: &str, src: &str) -> Span {
        let lines = lines_offsets(src);
        let lo = self.next_start_pos();
        // XXX(nika): Shouild we bother doing a checked cast or checked add here?
        let span = Span {
            lo,
            hi: lo + (src.len() as u32),
        };

        self.files.push(FileInfo {
            name: name.to_owned(),
            span,
            lines,
        });

        span
    }

    fn fileinfo(&self, span: Span) -> &FileInfo {
        for file in &self.files {
            if file.span_within(span) {
                return file;
            }
        }
        panic!("Invalid span with no related FileInfo!");
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Span {
    lo: u32,
    hi: u32,
}

impl Span {
    pub fn call_site() -> Span {
        Span { lo: 0, hi: 0 }
    }

    pub fn def_site() -> Span {
        Span::call_site()
    }

    pub fn resolved_at(&self, _other: Span) -> Span {
        // Stable spans consist only of line/column information, so
        // `resolved_at` and `located_at` only select which span the
        // caller wants line/column information from.
        *self
    }

    pub fn located_at(&self, other: Span) -> Span {
        other
    }

    pub fn source_file(&self) -> SourceFile {
        SOURCE_MAP.with(|cm| {
            let cm = cm.borrow();
            let fi = cm.fileinfo(*self);
            SourceFile {
                path: Path::new(&fi.name).to_owned(),
            }
        })
    }

    pub fn start(&self) -> LineColumn {
        SOURCE_MAP.with(|cm| {
            let cm = cm.borrow();
            let fi = cm.fileinfo(*self);
            fi.offset_line_column(self.lo as usize)
        })
    }

    pub fn end(&self) -> LineColumn {
        SOURCE_MAP.with(|cm| {
            let cm = cm.borrow();
            let fi = cm.fileinfo(*self);
            fi.offset_line_column(self.hi as usize)
        })
    }

    pub fn join(&self, other: Span) -> Option<Span> {
        SOURCE_MAP.with(|cm| {
            let cm = cm.borrow();
            // If `other` is not within the same FileInfo as us, return None.
            if !cm.fileinfo(*self).span_within(other) {
                return None;
            }
            Some(Span {
                lo: cmp::min(self.lo, other.lo),
                hi: cmp::max(self.hi, other.hi),
            })
        })
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bytes({}..{})", self.lo, self.hi)
    }
}

#[derive(Clone)]
pub struct Group {
    delimiter: Delimiter,
    stream: TokenStream,
    span: Span,
}

impl Group {
    pub fn new(delimiter: Delimiter, stream: TokenStream) -> Group {
        Group {
            delimiter,
            stream,
            span: Span::call_site(),
        }
    }

    pub fn delimiter(&self) -> Delimiter {
        self.delimiter
    }

    pub fn stream(&self) -> TokenStream {
        self.stream.clone()
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn span_open(&self) -> Span {
        self.span
    }

    pub fn span_close(&self) -> Span {
        self.span
    }

    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (left, right) = match self.delimiter {
            Delimiter::Parenthesis => ("(", ")"),
            Delimiter::Brace => ("{", "}"),
            Delimiter::Bracket => ("[", "]"),
            Delimiter::None => ("", ""),
        };

        f.write_str(left)?;
        self.stream.fmt(f)?;
        f.write_str(right)?;

        Ok(())
    }
}

/// An `Punct` is an single punctuation character like `+`, `-` or `#`.
///
/// Multicharacter operators like `+=` are represented as two instances of
/// `Punct` with different forms of `Spacing` returned.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Punct {
    op: char,
    joint: bool, // Spacing is not `Hash`
    span: Span,
}

impl Punct {
    /// Creates a new `Punct` from the given character and spacing.
    ///
    /// The `ch` argument must be a valid punctuation character permitted by the
    /// language, otherwise the function will panic.
    ///
    /// The returned `Punct` will have the default span of `Span::call_site()`
    /// which can be further configured with the `set_span` method below.
    pub fn new(op: char, spacing: Spacing) -> Punct {
        Punct {
            op,
            joint: spacing == Spacing::Joint,
            span: Span::call_site(),
        }
    }

    /// Returns the value of this punctuation character as `char`.
    pub fn as_char(&self) -> char {
        self.op
    }

    /// Returns the spacing of this punctuation character, indicating whether
    /// it's immediately followed by another `Punct` in the token stream, so
    /// they can potentially be combined into a multicharacter operator
    /// (`Joint`), or it's followed by some other token or whitespace (`Alone`)
    /// so the operator has certainly ended.
    pub fn spacing(&self) -> Spacing {
        if self.joint {
            Spacing::Joint
        } else {
            Spacing::Alone
        }
    }

    /// Returns the span for this punctuation character.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Configure the span for this punctuation character.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

/// Prints the punctuation character as a string that should be losslessly
/// convertible back into the same character.
impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op.fmt(f)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Ident {
    // sym: String,
    sym: u32,
    span: Span,
    raw: bool,
}

impl Ident {
    fn _new(string: &str, raw: bool, span: Span) -> Ident {
        lexer::validate_ident(string);

        Ident {
            // sym: string.to_owned(),
            sym: 0,
            span,
            raw,
        }
    }

    pub fn new(string: &str, span: Span) -> Ident {
        Ident::_new(string, false, span)
    }

    pub fn new_raw(string: &str, span: Span) -> Ident {
        Ident::_new(string, true, span)
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.raw {
            "r#".fmt(f)?;
        }
        self.sym.fmt(f)
    }
}

#[derive(Clone)]
pub struct Literal {
    text: String,
    span: Span,
}

macro_rules! suffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(format!(concat!("{}", stringify!($kind)), n))
        }
    )*)
}

macro_rules! unsuffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Literal {
            Literal::_new(n.to_string())
        }
    )*)
}

impl Literal {
    fn _new(text: String) -> Literal {
        Literal {
            text,
            span: Span::call_site(),
        }
    }

    suffixed_numbers! {
        u8_suffixed => u8,
        u16_suffixed => u16,
        u32_suffixed => u32,
        u64_suffixed => u64,
        u128_suffixed => u128,
        usize_suffixed => usize,
        i8_suffixed => i8,
        i16_suffixed => i16,
        i32_suffixed => i32,
        i64_suffixed => i64,
        i128_suffixed => i128,
        isize_suffixed => isize,

        f32_suffixed => f32,
        f64_suffixed => f64,
    }

    unsuffixed_numbers! {
        u8_unsuffixed => u8,
        u16_unsuffixed => u16,
        u32_unsuffixed => u32,
        u64_unsuffixed => u64,
        u128_unsuffixed => u128,
        usize_unsuffixed => usize,
        i8_unsuffixed => i8,
        i16_unsuffixed => i16,
        i32_unsuffixed => i32,
        i64_unsuffixed => i64,
        i128_unsuffixed => i128,
        isize_unsuffixed => isize,
    }

    pub fn f32_unsuffixed(f: f32) -> Literal {
        let mut s = f.to_string();
        if !s.contains(".") {
            s.push_str(".0");
        }
        Literal::_new(s)
    }

    pub fn f64_unsuffixed(f: f64) -> Literal {
        let mut s = f.to_string();
        if !s.contains(".") {
            s.push_str(".0");
        }
        Literal::_new(s)
    }

    pub fn string(t: &str) -> Literal {
        let mut text = String::with_capacity(t.len() + 2);
        text.push('"');
        for c in t.chars() {
            if c == '\'' {
                // escape_default turns this into "\'" which is unnecessary.
                text.push(c);
            } else {
                text.extend(c.escape_default());
            }
        }
        text.push('"');
        Literal::_new(text)
    }

    pub fn character(t: char) -> Literal {
        let mut text = String::new();
        text.push('\'');
        if t == '"' {
            // escape_default turns this into '\"' which is unnecessary.
            text.push(t);
        } else {
            text.extend(t.escape_default());
        }
        text.push('\'');
        Literal::_new(text)
    }

    pub fn byte_string(bytes: &[u8]) -> Literal {
        let mut escaped = "b\"".to_string();
        for b in bytes {
            match *b {
                b'\0' => escaped.push_str(r"\0"),
                b'\t' => escaped.push_str(r"\t"),
                b'\n' => escaped.push_str(r"\n"),
                b'\r' => escaped.push_str(r"\r"),
                b'"' => escaped.push_str("\\\""),
                b'\\' => escaped.push_str("\\\\"),
                b'\x20'..=b'\x7E' => escaped.push(*b as char),
                _ => escaped.push_str(&format!("\\x{:02X}", b)),
            }
        }
        escaped.push('"');
        Literal::_new(escaped)
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }

    pub fn subspan<R: RangeBounds<usize>>(&self, _range: R) -> Option<Span> {
        None
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.text.fmt(f)
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Literal")
            .field("lit", &format_args!("{}", self.text))
            .field("span", &self.span)
            .finish()
    }
}

struct Diagnostic {}

// The fallback server
struct Server {}

impl server::Types for Server {
    type TokenStream = TokenStream;
    type TokenStreamBuilder = TokenStreamBuilder;
    type TokenStreamIter = TokenStreamIter;
    type Group = Group;
    type Punct = Punct;
    type Ident = Ident;
    type Literal = Literal;
    type SourceFile = Rc<SourceFile>;
    type MultiSpan = Vec<Span>;
    type Diagnostic = Diagnostic;
    type Span = Span;
}

#[test]
fn whee() {
    
}

