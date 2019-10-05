//! Adapted from `proc_macro2`

use std::str::{Bytes, CharIndices, Chars};
use unicode_xid::UnicodeXID;

use proc_macro::bridge::TokenTree;
use proc_macro::{Delimiter, Spacing};

use crate::{set_tt_span, Group, Ident, Literal, Punct, Span, TokenStream, DUMMY_SPAN};

type TokenTreeT = TokenTree<Group, Punct, Ident, Literal>;

#[derive(Debug)]
pub(crate) struct LexError;

pub(crate) fn validate_ident(string: &str) {
    let validate = string;
    if validate.is_empty() {
        panic!("Ident is not allowed to be empty; use Option<Ident>");
    }

    if validate.bytes().all(|digit| digit >= b'0' && digit <= b'9') {
        panic!("Ident cannot be a number; use Literal instead");
    }

    fn ident_ok(string: &str) -> bool {
        let mut chars = string.chars();
        let first = chars.next().unwrap();
        if !is_ident_start(first) {
            return false;
        }
        for ch in chars {
            if !is_ident_continue(ch) {
                return false;
            }
        }
        true
    }

    if !ident_ok(validate) {
        panic!("{:?} is not a valid Ident", string);
    }
}

pub(crate) fn lex_stream(src: &str, off: u32) -> Result<TokenStream, LexError> {
    let cursor = Cursor { rest: src, off };
    match token_stream(cursor) {
        Ok((input, output)) => {
            if skip_whitespace(input).len() != 0 {
                Err(LexError)
            } else {
                Ok(output)
            }
        }
        Err(LexError) => Err(LexError),
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct Cursor<'a> {
    rest: &'a str,
    off: u32,
}

impl<'a> Cursor<'a> {
    fn advance(&self, amt: usize) -> Cursor<'a> {
        Cursor {
            rest: &self.rest[amt..],
            off: self.off + (amt as u32),
        }
    }

    fn find(&self, p: char) -> Option<usize> {
        self.rest.find(p)
    }

    fn starts_with(&self, s: &str) -> bool {
        self.rest.starts_with(s)
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    fn len(&self) -> usize {
        self.rest.len()
    }

    fn as_bytes(&self) -> &'a [u8] {
        self.rest.as_bytes()
    }

    fn bytes(&self) -> Bytes<'a> {
        self.rest.bytes()
    }

    fn chars(&self) -> Chars<'a> {
        self.rest.chars()
    }

    fn char_indices(&self) -> CharIndices<'a> {
        self.rest.char_indices()
    }
}

type PResult<'a, O> = Result<(Cursor<'a>, O), LexError>;

fn whitespace(input: Cursor) -> PResult<()> {
    if input.is_empty() {
        return Err(LexError);
    }

    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let s = input.advance(i);
        if bytes[i] == b'/' {
            if s.starts_with("//")
                && (!s.starts_with("///") || s.starts_with("////"))
                && !s.starts_with("//!")
            {
                if let Some(len) = s.find('\n') {
                    i += len + 1;
                    continue;
                }
                break;
            } else if s.starts_with("/**/") {
                i += 4;
                continue;
            } else if s.starts_with("/*")
                && (!s.starts_with("/**") || s.starts_with("/***"))
                && !s.starts_with("/*!")
            {
                let (_, com) = block_comment(s)?;
                i += com.len();
                continue;
            }
        }
        match bytes[i] {
            b' ' | 0x09..=0x0d => {
                i += 1;
                continue;
            }
            b if b <= 0x7f => {}
            _ => {
                let ch = s.chars().next().unwrap();
                if is_whitespace(ch) {
                    i += ch.len_utf8();
                    continue;
                }
            }
        }
        return if i > 0 { Ok((s, ())) } else { Err(LexError) };
    }
    Ok((input.advance(input.len()), ()))
}

fn block_comment(input: Cursor) -> PResult<&str> {
    if !input.starts_with("/*") {
        return Err(LexError);
    }

    let mut depth = 0;
    let bytes = input.as_bytes();
    let mut i = 0;
    let upper = bytes.len() - 1;
    while i < upper {
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            depth += 1;
            i += 1; // eat '*'
        } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
            depth -= 1;
            if depth == 0 {
                return Ok((input.advance(i + 2), &input.rest[..i + 2]));
            }
            i += 1; // eat '/'
        }
        i += 1;
    }
    Err(LexError)
}

fn skip_whitespace(input: Cursor) -> Cursor {
    match whitespace(input) {
        Ok((rest, _)) => rest,
        Err(LexError) => input,
    }
}

fn is_whitespace(ch: char) -> bool {
    // Rust treats left-to-right mark and right-to-left mark as whitespace
    ch.is_whitespace() || ch == '\u{200e}' || ch == '\u{200f}'
}

fn word_break(input: Cursor) -> PResult<()> {
    match input.chars().next() {
        Some(ch) if UnicodeXID::is_xid_continue(ch) => Err(LexError),
        Some(_) | None => Ok((input, ())),
    }
}

macro_rules! named {
    ($name:ident -> $o:ty, $submac:ident!( $($args:tt)* )) => {
        fn $name<'a>(i: Cursor<'a>) -> PResult<'a, $o> {
            $submac!(i, $($args)*)
        }
    };
}

macro_rules! alt {
    ($i:expr, $e:ident | $($rest:tt)*) => {
        alt!($i, call!($e) | $($rest)*)
    };

    ($i:expr, $subrule:ident!( $($args:tt)*) | $($rest:tt)*) => {
        match $subrule!($i, $($args)*) {
            res @ Ok(_) => res,
            _ => alt!($i, $($rest)*)
        }
    };

    ($i:expr, $subrule:ident!( $($args:tt)* ) => { $gen:expr } | $($rest:tt)+) => {
        match $subrule!($i, $($args)*) {
            Ok((i, o)) => Ok((i, $gen(o))),
            Err(LexError) => alt!($i, $($rest)*)
        }
    };

    ($i:expr, $e:ident => { $gen:expr } | $($rest:tt)*) => {
        alt!($i, call!($e) => { $gen } | $($rest)*)
    };

    ($i:expr, $e:ident => { $gen:expr }) => {
        alt!($i, call!($e) => { $gen })
    };

    ($i:expr, $subrule:ident!( $($args:tt)* ) => { $gen:expr }) => {
        match $subrule!($i, $($args)*) {
            Ok((i, o)) => Ok((i, $gen(o))),
            Err(LexError) => Err(LexError),
        }
    };

    ($i:expr, $e:ident) => {
        alt!($i, call!($e))
    };

    ($i:expr, $subrule:ident!( $($args:tt)*)) => {
        $subrule!($i, $($args)*)
    };
}

macro_rules! do_parse {
    ($i:expr, ( $($rest:expr),* )) => {
        Ok(($i, ( $($rest),* )))
    };

    ($i:expr, $e:ident >> $($rest:tt)*) => {
        do_parse!($i, call!($e) >> $($rest)*)
    };

    ($i:expr, $submac:ident!( $($args:tt)* ) >> $($rest:tt)*) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, _)) => do_parse!(i, $($rest)*),
        }
    };

    ($i:expr, $field:ident : $e:ident >> $($rest:tt)*) => {
        do_parse!($i, $field: call!($e) >> $($rest)*)
    };

    ($i:expr, $field:ident : $submac:ident!( $($args:tt)* ) >> $($rest:tt)*) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, o)) => {
                let $field = o;
                do_parse!(i, $($rest)*)
            },
        }
    };
}

macro_rules! peek {
    ($i:expr, $submac:ident!( $($args:tt)* )) => {
        match $submac!($i, $($args)*) {
            Ok((_, o)) => Ok(($i, o)),
            Err(LexError) => Err(LexError),
        }
    };
}

macro_rules! call {
    ($i:expr, $fun:expr $(, $args:expr)*) => {
        $fun($i $(, $args)*)
    };
}

macro_rules! option {
    ($i:expr, $f:expr) => {
        match $f($i) {
            Ok((i, o)) => Ok((i, Some(o))),
            Err(LexError) => Ok(($i, None)),
        }
    };
}

macro_rules! take_until_newline_or_eof {
    ($i:expr,) => {{
        if $i.len() == 0 {
            Ok(($i, ""))
        } else {
            match $i.find('\n') {
                Some(i) => Ok(($i.advance(i), &$i.rest[..i])),
                None => Ok(($i.advance($i.len()), &$i.rest[..$i.len()])),
            }
        }
    }};
}

macro_rules! tuple {
    ($i:expr, $($rest:tt)*) => {
        tuple_parser!($i, (), $($rest)*)
    };
}

/// Do not use directly. Use `tuple!`.
macro_rules! tuple_parser {
    ($i:expr, ($($parsed:tt),*), $e:ident, $($rest:tt)*) => {
        tuple_parser!($i, ($($parsed),*), call!($e), $($rest)*)
    };

    ($i:expr, (), $submac:ident!( $($args:tt)* ), $($rest:tt)*) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, o)) => tuple_parser!(i, (o), $($rest)*),
        }
    };

    ($i:expr, ($($parsed:tt)*), $submac:ident!( $($args:tt)* ), $($rest:tt)*) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, o)) => tuple_parser!(i, ($($parsed)* , o), $($rest)*),
        }
    };

    ($i:expr, ($($parsed:tt),*), $e:ident) => {
        tuple_parser!($i, ($($parsed),*), call!($e))
    };

    ($i:expr, (), $submac:ident!( $($args:tt)* )) => {
        $submac!($i, $($args)*)
    };

    ($i:expr, ($($parsed:expr),*), $submac:ident!( $($args:tt)* )) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, o)) => Ok((i, ($($parsed),*, o)))
        }
    };

    ($i:expr, ($($parsed:expr),*)) => {
        Ok(($i, ($($parsed),*)))
    };
}

macro_rules! not {
    ($i:expr, $submac:ident!( $($args:tt)* )) => {
        match $submac!($i, $($args)*) {
            Ok((_, _)) => Err(LexError),
            Err(LexError) => Ok(($i, ())),
        }
    };
}

macro_rules! tag {
    ($i:expr, $tag:expr) => {
        if $i.starts_with($tag) {
            Ok(($i.advance($tag.len()), &$i.rest[..$tag.len()]))
        } else {
            Err(LexError)
        }
    };
}

macro_rules! punct {
    ($i:expr, $punct:expr) => {
        punct($i, $punct)
    };
}

/// Do not use directly. Use `punct!`.
fn punct<'a>(input: Cursor<'a>, token: &'static str) -> PResult<'a, &'a str> {
    let input = skip_whitespace(input);
    if input.starts_with(token) {
        Ok((input.advance(token.len()), token))
    } else {
        Err(LexError)
    }
}

macro_rules! preceded {
    ($i:expr, $submac:ident!( $($args:tt)* ), $submac2:ident!( $($args2:tt)* )) => {
        match tuple!($i, $submac!($($args)*), $submac2!($($args2)*)) {
            Ok((remaining, (_, o))) => Ok((remaining, o)),
            Err(LexError) => Err(LexError),
        }
    };

    ($i:expr, $submac:ident!( $($args:tt)* ), $g:expr) => {
        preceded!($i, $submac!($($args)*), call!($g))
    };
}

macro_rules! delimited {
    ($i:expr, $submac:ident!( $($args:tt)* ), $($rest:tt)+) => {
        match tuple_parser!($i, (), $submac!($($args)*), $($rest)*) {
            Err(LexError) => Err(LexError),
            Ok((i1, (_, o, _))) => Ok((i1, o))
        }
    };
}

macro_rules! map {
    ($i:expr, $submac:ident!( $($args:tt)* ), $g:expr) => {
        match $submac!($i, $($args)*) {
            Err(LexError) => Err(LexError),
            Ok((i, o)) => Ok((i, call!(o, $g)))
        }
    };

    ($i:expr, $f:expr, $g:expr) => {
        map!($i, call!($f), $g)
    };
}

fn token_stream(mut input: Cursor) -> PResult<TokenStream> {
    let mut trees = Vec::new();
    loop {
        let input_no_ws = skip_whitespace(input);
        if input_no_ws.rest.len() == 0 {
            break;
        }
        if let Ok((a, tokens)) = doc_comment(input_no_ws) {
            input = a;
            trees.extend(tokens);
            continue;
        }

        let (a, tt) = match token_tree(input_no_ws) {
            Ok(p) => p,
            Err(_) => break,
        };
        trees.push(tt);
        input = a;
    }
    Ok((input, TokenStream { inner: trees }))
}

#[inline]
fn is_ident_start(c: char) -> bool {
    ('a' <= c && c <= 'z')
        || ('A' <= c && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && UnicodeXID::is_xid_start(c))
}

#[inline]
fn is_ident_continue(c: char) -> bool {
    ('a' <= c && c <= 'z')
        || ('A' <= c && c <= 'Z')
        || c == '_'
        || ('0' <= c && c <= '9')
        || (c > '\x7f' && UnicodeXID::is_xid_continue(c))
}

fn spanned<'a, T>(
    input: Cursor<'a>,
    f: fn(Cursor<'a>) -> PResult<'a, T>,
) -> PResult<'a, (T, Span)> {
    let input = skip_whitespace(input);
    let lo = input.off;
    let (a, b) = f(input)?;
    let hi = a.off;
    let span = Span { lo, hi };
    Ok((a, (b, span)))
}

fn token_tree(input: Cursor) -> PResult<TokenTreeT> {
    let (rest, (mut tt, span)) = spanned(input, token_kind)?;
    set_tt_span(&mut tt, span);
    Ok((rest, tt))
}

named!(token_kind -> TokenTreeT, alt!(
    map!(group, TokenTree::Group)
    |
    map!(literal, TokenTree::Literal) // must be before symbol
    |
    map!(op, TokenTree::Punct)
    |
    symbol_leading_ws
));

named!(group -> Group, alt!(
    delimited!(
        punct!("("),
        token_stream,
        punct!(")")
    ) => { |ts| Group::new(Delimiter::Parenthesis, ts, DUMMY_SPAN) }
    |
    delimited!(
        punct!("["),
        token_stream,
        punct!("]")
    ) => { |ts| Group::new(Delimiter::Bracket, ts, DUMMY_SPAN) }
    |
    delimited!(
        punct!("{"),
        token_stream,
        punct!("}")
    ) => { |ts| Group::new(Delimiter::Brace, ts, DUMMY_SPAN) }
));

fn symbol_leading_ws(input: Cursor) -> PResult<TokenTreeT> {
    symbol(skip_whitespace(input))
}

fn symbol(input: Cursor) -> PResult<TokenTreeT> {
    let raw = input.starts_with("r#");
    let rest = input.advance((raw as usize) << 1);

    let (rest, sym) = symbol_not_raw(rest)?;

    if !raw {
        let ident = Ident::new(sym, false, Span::call_site());
        return Ok((rest, TokenTree::Ident(ident)));
    }

    if sym == "_" {
        return Err(LexError);
    }

    let ident = Ident::new(sym, true, Span::call_site());
    Ok((rest, TokenTree::Ident(ident)))
}

fn symbol_not_raw(input: Cursor) -> PResult<&str> {
    let mut chars = input.char_indices();

    match chars.next() {
        Some((_, ch)) if is_ident_start(ch) => {}
        _ => return Err(LexError),
    }

    let mut end = input.len();
    for (i, ch) in chars {
        if !is_ident_continue(ch) {
            end = i;
            break;
        }
    }

    Ok((input.advance(end), &input.rest[..end]))
}

fn literal(input: Cursor) -> PResult<Literal> {
    let input_no_ws = skip_whitespace(input);

    match literal_nocapture(input_no_ws) {
        Ok((a, ())) => {
            let start = input.len() - input_no_ws.len();
            let len = input_no_ws.len() - a.len();
            let end = start + len;
            Ok((
                a,
                Literal::new(input.rest[start..end].to_string(), DUMMY_SPAN),
            ))
        }
        Err(LexError) => Err(LexError),
    }
}

named!(literal_nocapture -> (), alt!(
    string
    |
    byte_string
    |
    byte
    |
    character
    |
    float
    |
    int
));

named!(string -> (), alt!(
    quoted_string
    |
    preceded!(
        punct!("r"),
        raw_string
    ) => { |_| () }
));

named!(quoted_string -> (), do_parse!(
    punct!("\"") >>
    cooked_string >>
    tag!("\"") >>
    option!(symbol_not_raw) >>
    (())
));

fn cooked_string(input: Cursor) -> PResult<()> {
    let mut chars = input.char_indices().peekable();
    while let Some((byte_offset, ch)) = chars.next() {
        match ch {
            '"' => {
                return Ok((input.advance(byte_offset), ()));
            }
            '\r' => {
                if let Some((_, '\n')) = chars.next() {
                    // ...
                } else {
                    break;
                }
            }
            '\\' => match chars.next() {
                Some((_, 'x')) => {
                    if !backslash_x_char(&mut chars) {
                        break;
                    }
                }
                Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\'))
                | Some((_, '\'')) | Some((_, '"')) | Some((_, '0')) => {}
                Some((_, 'u')) => {
                    if !backslash_u(&mut chars) {
                        break;
                    }
                }
                Some((_, '\n')) | Some((_, '\r')) => {
                    while let Some(&(_, ch)) = chars.peek() {
                        if ch.is_whitespace() {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                _ => break,
            },
            _ch => {}
        }
    }
    Err(LexError)
}

named!(byte_string -> (), alt!(
    delimited!(
        punct!("b\""),
        cooked_byte_string,
        tag!("\"")
    ) => { |_| () }
    |
    preceded!(
        punct!("br"),
        raw_string
    ) => { |_| () }
));

fn cooked_byte_string(mut input: Cursor) -> PResult<()> {
    let mut bytes = input.bytes().enumerate();
    'outer: while let Some((offset, b)) = bytes.next() {
        match b {
            b'"' => {
                return Ok((input.advance(offset), ()));
            }
            b'\r' => {
                if let Some((_, b'\n')) = bytes.next() {
                    // ...
                } else {
                    break;
                }
            }
            b'\\' => match bytes.next() {
                Some((_, b'x')) => {
                    if !backslash_x_byte(&mut bytes) {
                        break;
                    }
                }
                Some((_, b'n')) | Some((_, b'r')) | Some((_, b't')) | Some((_, b'\\'))
                | Some((_, b'0')) | Some((_, b'\'')) | Some((_, b'"')) => {}
                Some((newline, b'\n')) | Some((newline, b'\r')) => {
                    let rest = input.advance(newline + 1);
                    for (offset, ch) in rest.char_indices() {
                        if !ch.is_whitespace() {
                            input = rest.advance(offset);
                            bytes = input.bytes().enumerate();
                            continue 'outer;
                        }
                    }
                    break;
                }
                _ => break,
            },
            b if b < 0x80 => {}
            _ => break,
        }
    }
    Err(LexError)
}

fn raw_string(input: Cursor) -> PResult<()> {
    let mut chars = input.char_indices();
    let mut n = 0;
    while let Some((byte_offset, ch)) = chars.next() {
        match ch {
            '"' => {
                n = byte_offset;
                break;
            }
            '#' => {}
            _ => return Err(LexError),
        }
    }
    for (byte_offset, ch) in chars {
        match ch {
            '"' if input.advance(byte_offset + 1).starts_with(&input.rest[..n]) => {
                let rest = input.advance(byte_offset + 1 + n);
                return Ok((rest, ()));
            }
            '\r' => {}
            _ => {}
        }
    }
    Err(LexError)
}

named!(byte -> (), do_parse!(
    punct!("b") >>
    tag!("'") >>
    cooked_byte >>
    tag!("'") >>
    (())
));

fn cooked_byte(input: Cursor) -> PResult<()> {
    let mut bytes = input.bytes().enumerate();
    let ok = match bytes.next().map(|(_, b)| b) {
        Some(b'\\') => match bytes.next().map(|(_, b)| b) {
            Some(b'x') => backslash_x_byte(&mut bytes),
            Some(b'n') | Some(b'r') | Some(b't') | Some(b'\\') | Some(b'0') | Some(b'\'')
            | Some(b'"') => true,
            _ => false,
        },
        b => b.is_some(),
    };
    if ok {
        match bytes.next() {
            Some((offset, _)) => {
                if input.chars().as_str().is_char_boundary(offset) {
                    Ok((input.advance(offset), ()))
                } else {
                    Err(LexError)
                }
            }
            None => Ok((input.advance(input.len()), ())),
        }
    } else {
        Err(LexError)
    }
}

named!(character -> (), do_parse!(
    punct!("'") >>
    cooked_char >>
    tag!("'") >>
    (())
));

fn cooked_char(input: Cursor) -> PResult<()> {
    let mut chars = input.char_indices();
    let ok = match chars.next().map(|(_, ch)| ch) {
        Some('\\') => match chars.next().map(|(_, ch)| ch) {
            Some('x') => backslash_x_char(&mut chars),
            Some('u') => backslash_u(&mut chars),
            Some('n') | Some('r') | Some('t') | Some('\\') | Some('0') | Some('\'') | Some('"') => {
                true
            }
            _ => false,
        },
        ch => ch.is_some(),
    };
    if ok {
        match chars.next() {
            Some((idx, _)) => Ok((input.advance(idx), ())),
            None => Ok((input.advance(input.len()), ())),
        }
    } else {
        Err(LexError)
    }
}

macro_rules! next_ch {
    ($chars:ident @ $pat:pat $(| $rest:pat)*) => {
        match $chars.next() {
            Some((_, ch)) => match ch {
                $pat $(| $rest)*  => ch,
                _ => return false,
            },
            None => return false
        }
    };
}

fn backslash_x_char<I>(chars: &mut I) -> bool
where
    I: Iterator<Item = (usize, char)>,
{
    next_ch!(chars @ '0'..='7');
    next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
    true
}

fn backslash_x_byte<I>(chars: &mut I) -> bool
where
    I: Iterator<Item = (usize, u8)>,
{
    next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
    next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
    true
}

fn backslash_u<I>(chars: &mut I) -> bool
where
    I: Iterator<Item = (usize, char)>,
{
    next_ch!(chars @ '{');
    next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
    loop {
        let c = next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F' | '_' | '}');
        if c == '}' {
            return true;
        }
    }
}

fn float(input: Cursor) -> PResult<()> {
    let (mut rest, ()) = float_digits(input)?;
    if let Some(ch) = rest.chars().next() {
        if is_ident_start(ch) {
            rest = symbol_not_raw(rest)?.0;
        }
    }
    word_break(rest)
}

fn float_digits(input: Cursor) -> PResult<()> {
    let mut chars = input.chars().peekable();
    match chars.next() {
        Some(ch) if ch >= '0' && ch <= '9' => {}
        _ => return Err(LexError),
    }

    let mut len = 1;
    let mut has_dot = false;
    let mut has_exp = false;
    while let Some(&ch) = chars.peek() {
        match ch {
            '0'..='9' | '_' => {
                chars.next();
                len += 1;
            }
            '.' => {
                if has_dot {
                    break;
                }
                chars.next();
                if chars
                    .peek()
                    .map(|&ch| ch == '.' || is_ident_start(ch))
                    .unwrap_or(false)
                {
                    return Err(LexError);
                }
                len += 1;
                has_dot = true;
            }
            'e' | 'E' => {
                chars.next();
                len += 1;
                has_exp = true;
                break;
            }
            _ => break,
        }
    }

    let rest = input.advance(len);
    if !(has_dot || has_exp || rest.starts_with("f32") || rest.starts_with("f64")) {
        return Err(LexError);
    }

    if has_exp {
        let mut has_exp_value = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '+' | '-' => {
                    if has_exp_value {
                        break;
                    }
                    chars.next();
                    len += 1;
                }
                '0'..='9' => {
                    chars.next();
                    len += 1;
                    has_exp_value = true;
                }
                '_' => {
                    chars.next();
                    len += 1;
                }
                _ => break,
            }
        }
        if !has_exp_value {
            return Err(LexError);
        }
    }

    Ok((input.advance(len), ()))
}

fn int(input: Cursor) -> PResult<()> {
    let (mut rest, ()) = digits(input)?;
    if let Some(ch) = rest.chars().next() {
        if is_ident_start(ch) {
            rest = symbol_not_raw(rest)?.0;
        }
    }
    word_break(rest)
}

fn digits(mut input: Cursor) -> PResult<()> {
    let base = if input.starts_with("0x") {
        input = input.advance(2);
        16
    } else if input.starts_with("0o") {
        input = input.advance(2);
        8
    } else if input.starts_with("0b") {
        input = input.advance(2);
        2
    } else {
        10
    };

    let mut len = 0;
    let mut empty = true;
    for b in input.bytes() {
        let digit = match b {
            b'0'..=b'9' => (b - b'0') as u64,
            b'a'..=b'f' => 10 + (b - b'a') as u64,
            b'A'..=b'F' => 10 + (b - b'A') as u64,
            b'_' => {
                if empty && base == 10 {
                    return Err(LexError);
                }
                len += 1;
                continue;
            }
            _ => break,
        };
        if digit >= base {
            return Err(LexError);
        }
        len += 1;
        empty = false;
    }
    if empty {
        Err(LexError)
    } else {
        Ok((input.advance(len), ()))
    }
}

fn op(input: Cursor) -> PResult<Punct> {
    let input = skip_whitespace(input);
    match op_char(input) {
        Ok((rest, '\'')) => {
            symbol(rest)?;
            Ok((rest, Punct::new('\'', Spacing::Joint, DUMMY_SPAN)))
        }
        Ok((rest, ch)) => {
            let kind = match op_char(rest) {
                Ok(_) => Spacing::Joint,
                Err(LexError) => Spacing::Alone,
            };
            Ok((rest, Punct::new(ch, kind, DUMMY_SPAN)))
        }
        Err(LexError) => Err(LexError),
    }
}

fn op_char(input: Cursor) -> PResult<char> {
    if input.starts_with("//") || input.starts_with("/*") {
        // Do not accept `/` of a comment as an op.
        return Err(LexError);
    }

    let mut chars = input.chars();
    let first = match chars.next() {
        Some(ch) => ch,
        None => {
            return Err(LexError);
        }
    };
    let recognized = "~!@#$%^&*-=+|;:,<.>/?'";
    if recognized.contains(first) {
        Ok((input.advance(first.len_utf8()), first))
    } else {
        Err(LexError)
    }
}

fn doc_comment(input: Cursor) -> PResult<Vec<TokenTreeT>> {
    let (rest, ((comment, inner), span)) = spanned(input, doc_comment_contents)?;

    let mut trees = Vec::new();
    trees.push(TokenTree::Punct(Punct::new('#', Spacing::Alone, span)));
    if inner {
        trees.push(TokenTree::Punct(Punct::new('!', Spacing::Alone, span)));
    }
    trees.push(TokenTree::Group(Group::new(
        Delimiter::Bracket,
        TokenStream {
            inner: vec![
                TokenTree::Ident(Ident::new("doc", false, span)),
                TokenTree::Punct(Punct::new('=', Spacing::Alone, span)),
                TokenTree::Literal(Literal::string(comment, span)),
            ],
        },
        span,
    )));
    Ok((rest, trees))
}

named!(doc_comment_contents -> (&str, bool), alt!(
    do_parse!(
        punct!("//!") >>
        s: take_until_newline_or_eof!() >>
        ((s, true))
    )
    |
    do_parse!(
        option!(whitespace) >>
        peek!(tag!("/*!")) >>
        s: block_comment >>
        ((s, true))
    )
    |
    do_parse!(
        punct!("///") >>
        not!(tag!("/")) >>
        s: take_until_newline_or_eof!() >>
        ((s, false))
    )
    |
    do_parse!(
        option!(whitespace) >>
        peek!(tuple!(tag!("/**"), not!(tag!("*")))) >>
        s: block_comment >>
        ((s, false))
    )
));
