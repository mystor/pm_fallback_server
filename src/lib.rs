//! THIS IS SUPER UNSTABLE
//! THIS IS PRIMARIALY INTENDED AS AN EXPERIMENT

#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use std::cmp;
use std::fmt;
use std::ops::Bound;
use std::rc::Rc;
use std::vec;
use std::cell::Cell;
use std::panic;

use proc_macro::bridge::{client, server, TokenTree};
use proc_macro::{Delimiter, Level, LineColumn, Spacing};

mod lexer;

// Hacky thread local and client set-up to work around the limited internal API
// used by proc_macro.
thread_local! {
    static COMMAND: Cell<Option<Box<dyn FnOnce()>>> = Cell::new(None);
}

fn my_client_impl(stream: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Call the stashed closure
    let cmd = COMMAND.with(|cmd| cmd.replace(None));
    cmd.unwrap()();
    stream
}
const CLIENT: client::Client<fn(proc_macro::TokenStream) -> proc_macro::TokenStream> =
    client::Client::expand1(my_client_impl);

/// Run the command with a proc_macro server.
pub fn run_server<F: FnOnce() + 'static>(f: F) {
    let server = Server::new();

    let cmd = COMMAND.with(|cmd| cmd.replace(Some(Box::new(f))));
    assert!(cmd.is_none(), "Found pending command");

    let result = CLIENT.run(&server::SameThread, server, TokenStream::new());

    let cmd = COMMAND.with(|cmd| cmd.replace(None));
    assert!(cmd.is_none(), "Command was not processed");

    if let Err(err) = result {
        panic::resume_unwind(err.into());
    }
}

// The internal `Server` type.
struct Server {
    source_map: SourceMap,
    def_site: Span,
    call_site: Span,
    // mixed_site: Span,
}

impl Server {
    fn new() -> Self {
        Server {
            source_map: SourceMap {
                files: vec![Rc::new(FileInfo {
                    name: "<unspecified>".to_owned(),
                    span: DUMMY_SPAN,
                    lines: vec![0],
                })],
            },
            def_site: DUMMY_SPAN,
            call_site: DUMMY_SPAN,
            // mixed_site: DUMMY_SPAN,
        }
    }
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
        lexer::lex_stream(src, span.lo).expect("error while lexing")
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
        TokenStream {
            inner: builder.inner,
        }
    }
}

impl server::TokenStreamIter for Server {
    fn next(
        &mut self,
        iter: &mut Self::TokenStreamIter,
    ) -> Option<TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>> {
        iter.next()
    }
}

impl server::Group for Server {
    fn new(&mut self, delimiter: Delimiter, stream: Self::TokenStream) -> Self::Group {
        Group::new(delimiter, stream, self.call_site)
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
        Punct::new(ch, spacing, self.call_site)
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
        Ident::new(string, is_raw, span)
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
        format!(
            "Literal {{ lit: {}, span: {:?} }}",
            literal.text, literal.span
        )
    }

    fn integer(&mut self, n: &str) -> Self::Literal {
        Literal::new(format!("{}", n), self.call_site)
    }

    fn typed_integer(&mut self, n: &str, kind: &str) -> Self::Literal {
        Literal::new(format!("{}{}", n, kind), self.call_site)
    }

    fn float(&mut self, n: &str) -> Self::Literal {
        let mut s = n.to_string();
        if !s.contains(".") {
            s.push_str(".9");
        }
        Literal::new(s, self.call_site)
    }

    fn f32(&mut self, n: &str) -> Self::Literal {
        Literal::new(format!("{}f32", n), self.call_site)
    }

    fn f64(&mut self, n: &str) -> Self::Literal {
        Literal::new(format!("{}f64", n), self.call_site)
    }

    fn string(&mut self, string: &str) -> Self::Literal {
        Literal::string(string, self.call_site)
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
        Literal::new(text, self.call_site)
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
        Literal::new(escaped, self.call_site)
    }

    fn span(&mut self, literal: &Self::Literal) -> Self::Span {
        literal.span
    }

    fn set_span(&mut self, literal: &mut Self::Literal, span: Self::Span) {
        literal.set_span(span)
    }

    fn subspan(
        &mut self,
        _literal: &Self::Literal,
        _start: Bound<usize>,
        _end: Bound<usize>,
    ) -> Option<Self::Span> {
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
    fn new(&mut self, level: Level, msg: &str, _spans: Self::MultiSpan) -> Self::Diagnostic {
        Diagnostic { level, msg: msg.to_string(), children: Vec::new() }
    }

    fn sub(
        &mut self,
        diag: &mut Self::Diagnostic,
        level: Level,
        msg: &str,
        _spans: Self::MultiSpan,
    ) {
        diag.children.push(Diagnostic {
            level,
            msg: msg.to_string(),
            children: Vec::new(),
        });
    }

    fn emit(&mut self, diag: Self::Diagnostic) {
        eprintln!("{:?}: {}", diag.level, diag.msg);
        for child in diag.children {
            self.emit(child)
        }
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

    fn parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
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
        None  // NOTE: Consider keeping track of source text we've been asked to parse?
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
                TokenTree::Group(ref tt) => write!(f, "{}", tt)?,
                TokenTree::Ident(ref tt) => write!(f, "{}", tt)?,
                TokenTree::Punct(ref tt) => {
                    write!(f, "{}", tt.op)?;
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

const DUMMY_SPAN: Span = Span { lo: 0, hi: 0 };

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
    fn new(delimiter: Delimiter, stream: TokenStream, span: Span) -> Group {
        Group {
            delimiter,
            stream,
            span,
        }
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

        if self.stream.is_empty() {
            write!(f, "{} {}", left, right)
        } else {
            write!(f, "{} {} {}", left, self.stream, right)
        }
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
    fn new(op: char, spacing: Spacing, span: Span) -> Punct {
        Punct {
            op,
            joint: spacing == Spacing::Joint,
            span,
        }
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
    fn new(string: &str, raw: bool, span: Span) -> Ident {
        lexer::validate_ident(string);

        Ident {
            // sym: string.to_owned(),
            sym: 0,
            span,
            raw,
        }
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

impl Literal {
    fn new(text: String, span: Span) -> Literal {
        Literal { text, span }
    }

    fn string(t: &str, span: Span) -> Literal {
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
        Literal::new(text, span)
    }

    fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.text.fmt(f)
    }
}

struct Diagnostic {
    level: Level,
    msg: String,
    children: Vec<Diagnostic>,
}
