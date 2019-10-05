//! THIS IS SUPER UNSTABLE
//! THIS IS PRIMARIALY INTENDED AS AN EXPERIMENT

#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use std::cell::RefCell;
use std::cmp;
use std::fmt;
use std::iter;
use std::ops::{RangeBounds, Bound};
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::rc::Rc;
use std::vec;

use proc_macro::{Delimiter, Spacing, LineColumn, Level};
use proc_macro::bridge::{server, client, TokenTree};

mod lexer;

fn my_client_impl(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}
const CLIENT: client::Client<fn(proc_macro::TokenStream) -> proc_macro::TokenStream> = client::Client::expand1(my_client_impl);

pub fn run_server<F: FnOnce()>() -> Result<(), ()> {
    let dummy_span = Span { lo: 0, hi: 0 };
    let server = Server {
        source_map: SourceMap {
            files: vec![Rc::new(FileInfo {
                name: "<unspecified>".to_owned(),
                span: dummy_span,
                lines: vec![0],
            })],
        },
        def_site: dummy_span,
        call_site: dummy_span,
        mixed_site: dummy_span,
    };

    match CLIENT.run(&server::SameThread, server, TokenStream::new()) {
        Ok(_) => Ok(()),
        Err(_) => Err(()),
    }
}

// The canonical `Server` type used by this crate.
struct Server {
    source_map: SourceMap,
    def_site: Span,
    call_site: Span,
    mixed_site: Span,
}

pub fn new_server() {
    unimplemented!()
}

impl server::Types for Server {
    type TokenStream = TokenStream;
    type TokenStreamBuilder = TokenStreamBuilder;
    type TokenStreamIter = TokenStreamIter;
    type Group = Group;
    type Punct = Punct;
    type Ident = Ident;
    type Literal = Literal;
    type SourceFile = Rc<FileInfo>;
    type MultiSpan = Vec<Span>;
    type Diagnostic = Diagnostic;
    type Span = Span;
}

impl server::TokenStream for Server {
    fn new(&mut self) -> Self::TokenStream {
        TokenStream::new()
    }

    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }

    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        let name = format!("<parsed string {}>", self.source_map.files.len());
        let span = self.source_map.add_file(&name, src);
        lexer::lex_stream(src, span.lo)
            .expect("error while lexing")
    }

    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }

    fn from_token_tree(
        &mut self,
        tree: TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>,
    ) -> Self::TokenStream {
        TokenStream { inner: vec![tree] }
    }

    fn into_iter(&mut self, stream: Self::TokenStream) -> Self::TokenStreamIter {
        stream.inner.into_iter()
    }
}

impl server::TokenStreamBuilder for Server {
    fn new(&mut self) -> Self::TokenStreamBuilder {
        TokenStreamBuilder { inner: Vec::new() }
    }

    fn push(&mut self, builder: &mut Self::TokenStreamBuilder, stream: Self::TokenStream) {
        builder.inner.extend(stream.inner)
    }

    fn build(&mut self, builder: Self::TokenStreamBuilder) -> Self::TokenStream {
        TokenStream { inner: builder.inner }
    }
}

impl server::TokenStreamIter for Server {
    fn next(&mut self, iter: &mut Self::TokenStreamIter) -> Option<TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>> {
        iter.next()
    }
}

impl server::Group for Server {
    fn new(&mut self, delimiter: Delimiter, stream: Self::TokenStream) -> Self::Group {
        Group::new(delimiter, stream)
    }

    fn delimiter(&mut self, group: &Self::Group) -> Delimiter {
        group.delimiter
    }

    fn stream(&mut self, group: &Self::Group) -> Self::TokenStream {
        group.stream.clone()
    }

    fn span(&mut self, group: &Self::Group) -> Self::Span {
        group.span
    }

    fn span_open(&mut self, group: &Self::Group) -> Self::Span {
        group.span // FIXME: Generate correct span
    }

    fn span_close(&mut self, group: &Self::Group) -> Self::Span {
        group.span // FIXME: Generate correct span
    }

    fn set_span(&mut self, group: &mut Self::Group, span: Self::Span) {
        group.set_span(span);
    }
}

impl server::Punct for Server {
    fn new(&mut self, ch: char, spacing: Spacing) -> Self::Punct {
        Punct::new(ch, spacing)
    }

    fn as_char(&mut self, punct: Self::Punct) -> char {
        punct.op
    }

    fn spacing(&mut self, punct: Self::Punct) -> Spacing {
        punct.spacing()
    }

    fn span(&mut self, punct: Self::Punct) -> Self::Span {
        punct.span
    }

    fn with_span(&mut self, punct: Self::Punct, span: Self::Span) -> Self::Punct {
        Punct { span, ..punct }
    }
}

impl server::Ident for Server {
    fn new(&mut self, string: &str, span: Self::Span, is_raw: bool) -> Self::Ident {
        Ident::_new(string, is_raw, span)
    }

    fn span(&mut self, ident: Self::Ident) -> Self::Span {
        ident.span
    }

    fn with_span(&mut self, ident: Self::Ident, span: Self::Span) -> Self::Ident {
        Ident { span, ..ident }
    }
}

impl server::Literal for Server {
    fn debug(&mut self, literal: &Self::Literal) -> String {
        format!("{:?}", literal)
    }

    fn integer(&mut self, n: &str) -> Self::Literal {
        Literal::_new(format!("{}", n))
    }

    fn typed_integer(&mut self, n: &str, kind: &str) -> Self::Literal {
        Literal::_new(format!("{}{}", n, kind))
    }

    fn float(&mut self, n: &str) -> Self::Literal {
        let mut s = n.to_string();
        if !s.contains(".") {
            s.push_str(".9");
        }
        Literal::_new(s)
    }

    fn f32(&mut self, n: &str) -> Self::Literal {
        Literal::_new(format!("{}f32", n))
    }

    fn f64(&mut self, n: &str) -> Self::Literal {
        Literal::_new(format!("{}f64", n))
    }

    fn string(&mut self, string: &str) -> Self::Literal {
        let mut text = String::with_capacity(string.len() + 2);
        text.push('"');
        for c in string.chars() {
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

    fn character(&mut self, ch: char) -> Self::Literal {
        let mut text = String::new();
        text.push('\'');
        if ch == '"' {
            // escape_default turns this into '\"' which is unnecessary.
            text.push(ch);
        } else {
            text.extend(ch.escape_default());
        }
        text.push('\'');
        Literal::_new(text)
    }

    fn byte_string(&mut self, bytes: &[u8]) -> Self::Literal {
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

    fn span(&mut self, literal: &Self::Literal) -> Self::Span {
        literal.span
    }

    fn set_span(&mut self, literal: &mut Self::Literal, span: Self::Span) {
        literal.set_span(span)
    }

    fn subspan(&mut self, _literal: &Self::Literal, _start: Bound<usize>, _end: Bound<usize>) -> Option<Self::Span> {
        None // FIXME: Support this
    }
}

impl server::SourceFile for Server {
    fn eq(&mut self, file1: &Self::SourceFile, file2: &Self::SourceFile) -> bool {
        Rc::ptr_eq(file1, file2)
    }

    fn path(&mut self, file: &Self::SourceFile) -> String {
        file.name.clone()
    }

    fn is_real(&mut self, _file: &Self::SourceFile) -> bool {
        false
    }
}

impl server::MultiSpan for Server {
    fn new(&mut self) -> Self::MultiSpan {
        vec![]
    }

    fn push(&mut self, spans: &mut Self::MultiSpan, span: Self::Span) {
        spans.push(span)
    }
}

impl server::Diagnostic for Server {
    fn new(&mut self, _level: Level, _msg: &str, _spans: Self::MultiSpan) -> Self::Diagnostic {
        unimplemented!()
    }

    fn sub(&mut self, _diag: &mut Self::Diagnostic, _level: Level, _msg: &str, _spans: Self::MultiSpan) {
        unimplemented!()
    }

    fn emit(&mut self, _diag: Self::Diagnostic) {
        unimplemented!()
    }
}

impl server::Span for Server {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("bytes({}..{})", span.lo, span.hi)
    }
    fn def_site(&mut self) -> Self::Span {
        self.def_site
    }
    fn call_site(&mut self) -> Self::Span {
        self.call_site
    }
    /*
    fn mixed_site(&mut self) -> Self::Span {
        self.mixed_site
    }
     */
    fn source_file(&mut self, span: Self::Span) -> Self::SourceFile {
        self.source_map.fileinfo(span).clone()
    }
    fn parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        None
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        span
    }
    fn start(&mut self, span: Self::Span) -> LineColumn {
        let fi = self.source_map.fileinfo(span);
        fi.offset_line_column(span.lo as usize)
    }
    fn end(&mut self, span: Self::Span) -> LineColumn {
        let fi = self.source_map.fileinfo(span);
        fi.offset_line_column(span.hi as usize)
    }
    fn join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        // If `other` is not within the same `FileInfo` as us, return None.
        if !self.source_map.fileinfo(first).span_within(second) {
            return None;
        }
        Some(Span {
            lo: cmp::min(first.lo, second.lo),
            hi: cmp::max(first.hi, second.hi),
        })
    }
    fn resolved_at(&mut self, span: Self::Span, _at: Self::Span) -> Self::Span {
        span
    }
    fn source_text(&mut self, _span: Self::Span) -> Option<String> {
        // NOTE: Consider keeping track of source text we've been asked to parse?
        None
    }
}

type TokenTreeT = TokenTree<Group, Punct, Ident, Literal>;

fn set_tt_span(tt: &mut TokenTreeT, span: Span) {
    match tt {
        TokenTree::Group(t) => t.set_span(span),
        TokenTree::Literal(t) => t.set_span(span),
        TokenTree::Ident(t) => t.set_span(span),
        TokenTree::Punct(t) => t.set_span(span),
    }
}

struct TokenStreamBuilder {
    inner: Vec<TokenTreeT>,
}

#[derive(Clone)]
struct TokenStream {
    inner: Vec<TokenTreeT>,
}

impl TokenStream {
    fn new() -> TokenStream {
        TokenStream { inner: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.inner.len() == 0
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
                    if tt.stream.is_empty() {
                        write!(f, "{} {}", start, end)?
                    } else {
                        write!(f, "{} {} {}", start, tt.stream, end)?
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

type TokenStreamIter = vec::IntoIter<TokenTreeT>;

#[derive(Clone, PartialEq, Eq)]
struct SourceFile {
    path: PathBuf,
}

impl SourceFile {
    /// Get the path to this source file as a string.
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    fn is_real(&self) -> bool {
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
    files: Vec<Rc<FileInfo>>,
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

        self.files.push(Rc::new(FileInfo {
            name: name.to_owned(),
            span,
            lines,
        }));

        span
    }

    fn fileinfo(&self, span: Span) -> &Rc<FileInfo> {
        for file in &self.files {
            if file.span_within(span) {
                return file;
            }
        }
        panic!("Invalid span with no related FileInfo!");
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct Span {
    lo: u32,
    hi: u32,
}

impl Span {
    fn call_site() -> Span {
        Span { lo: 0, hi: 0 }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bytes({}..{})", self.lo, self.hi)
    }
}

#[derive(Clone)]
struct Group {
    delimiter: Delimiter,
    stream: TokenStream,
    span: Span,
}

impl Group {
    fn new(delimiter: Delimiter, stream: TokenStream) -> Group {
        Group {
            delimiter,
            stream,
            span: Span::call_site(),
        }
    }

    fn delimiter(&self) -> Delimiter {
        self.delimiter
    }

    fn stream(&self) -> TokenStream {
        self.stream.clone()
    }

    fn span(&self) -> Span {
        self.span
    }

    fn span_open(&self) -> Span {
        self.span
    }

    fn span_close(&self) -> Span {
        self.span
    }

    fn set_span(&mut self, span: Span) {
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
struct Punct {
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
    fn new(op: char, spacing: Spacing) -> Punct {
        Punct {
            op,
            joint: spacing == Spacing::Joint,
            span: Span::call_site(),
        }
    }

    /// Returns the value of this punctuation character as `char`.
    fn as_char(&self) -> char {
        self.op
    }

    /// Returns the spacing of this punctuation character, indicating whether
    /// it's immediately followed by another `Punct` in the token stream, so
    /// they can potentially be combined into a multicharacter operator
    /// (`Joint`), or it's followed by some other token or whitespace (`Alone`)
    /// so the operator has certainly ended.
    fn spacing(&self) -> Spacing {
        if self.joint {
            Spacing::Joint
        } else {
            Spacing::Alone
        }
    }

    /// Returns the span for this punctuation character.
    fn span(&self) -> Span {
        self.span
    }

    /// Configure the span for this punctuation character.
    fn set_span(&mut self, span: Span) {
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
struct Ident {
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

    fn new(string: &str, span: Span) -> Ident {
        Ident::_new(string, false, span)
    }

    fn new_raw(string: &str, span: Span) -> Ident {
        Ident::_new(string, true, span)
    }

    fn span(&self) -> Span {
        self.span
    }

    fn set_span(&mut self, span: Span) {
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
struct Literal {
    text: String,
    span: Span,
}

macro_rules! suffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        fn $name(n: $kind) -> Literal {
            Literal::_new(format!(concat!("{}", stringify!($kind)), n))
        }
    )*)
}

macro_rules! unsuffixed_numbers {
    ($($name:ident => $kind:ident,)*) => ($(
        fn $name(n: $kind) -> Literal {
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

    fn f32_unsuffixed(f: f32) -> Literal {
        let mut s = f.to_string();
        if !s.contains(".") {
            s.push_str(".0");
        }
        Literal::_new(s)
    }

    fn f64_unsuffixed(f: f64) -> Literal {
        let mut s = f.to_string();
        if !s.contains(".") {
            s.push_str(".0");
        }
        Literal::_new(s)
    }

    fn string(t: &str) -> Literal {
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

    fn character(t: char) -> Literal {
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

    fn byte_string(bytes: &[u8]) -> Literal {
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

    fn span(&self) -> Span {
        self.span
    }

    fn set_span(&mut self, span: Span) {
        self.span = span;
    }

    fn subspan<R: RangeBounds<usize>>(&self, _range: R) -> Option<Span> {
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

