use std::iter;
use std::mem;
use std::vec;

use crate::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use rustc_lexer::{first_token, is_id_continue, is_id_start, TokenKind};

fn tokenize(mut input: &str, mut lo: u32) -> impl Iterator<Item = (TokenKind, &str, Span)> + '_ {
    iter::from_fn(move || {
        if input.is_empty() {
            return None;
        }

        let token = first_token(input);
        let hi = lo + token.len as u32;
        let span = Span { lo, hi };
        lo = span.hi;
        let text = &input[..token.len];
        input = &input[token.len..];
        Some((token.kind, text, span))
    })
}

fn doc_attr(
    tts: &mut Vec<TokenTree<Group, Punct, Ident, Literal>>,
    doc: &str,
    inner: bool,
    span: Span,
) {
    tts.push(TokenTree::Punct(Punct::new('#', Spacing::Alone, span)));
    if inner {
        tts.push(TokenTree::Punct(Punct::new('!', Spacing::Alone, span)));
    }
    tts.push(TokenTree::Group(Group {
        delimiter: Delimiter::Brace,
        stream: TokenStream {
            inner: vec![
                TokenTree::Ident(Ident::new("doc", false, span)),
                TokenTree::Punct(Punct::new('=', Spacing::Alone, span)),
                TokenTree::Literal(Literal::string(doc, span)),
            ],
        },
        span,
    }))
}

pub(crate) fn lex_stream(src: &str, lo: u32) -> Result<TokenStream, ()> {
    let mut stack = Vec::new();
    let mut active = Vec::new();

    let mut iter = tokenize(src, lo).peekable();
    while let Some((kind, text, span)) = iter.next() {
        match kind {
            TokenKind::Whitespace => {}

            TokenKind::LineComment => {
                if text.starts_with("//!") {
                    doc_attr(&mut active, &text[3..], true, span);
                } else if text.starts_with("///") && !text.starts_with("////") {
                    doc_attr(&mut active, &text[3..], false, span);
                }
            }
            TokenKind::BlockComment { terminated: false } => return Err(()),
            TokenKind::BlockComment { terminated: true } => {
                // FIXME: This is wrong, but is good-enough for now. We don't
                // appear to handle block doc-comments correctly in any fallback
                // implementations, so this is about as correct as proc-macro2.
                if text.starts_with("/*!") {
                    doc_attr(&mut active, &text[3..text.len() - 2], true, span);
                } else if text.starts_with("/**") {
                    doc_attr(&mut active, &text[3..text.len() - 2], false, span);
                }
            }

            TokenKind::OpenParen | TokenKind::OpenBrace | TokenKind::OpenBracket => {
                let delimiter = match kind {
                    TokenKind::OpenParen => Delimiter::Parenthesis,
                    TokenKind::OpenBrace => Delimiter::Brace,
                    TokenKind::OpenBracket => Delimiter::Bracket,
                    _ => unreachable!(),
                };
                let parent = mem::replace(&mut active, Vec::new());
                stack.push((delimiter, parent));
            }

            TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket => {
                let expected = match kind {
                    TokenKind::CloseParen => Delimiter::Parenthesis,
                    TokenKind::CloseBrace => Delimiter::Brace,
                    TokenKind::CloseBracket => Delimiter::Bracket,
                    _ => unreachable!(),
                };
                let (delimiter, parent) = stack.pop().ok_or(())?;
                if delimiter != expected {
                    return Err(());
                }

                let inner = mem::replace(&mut active, parent);
                active.push(TokenTree::Group(Group {
                    delimiter,
                    span,
                    stream: TokenStream { inner },
                }));
            }

            TokenKind::Ident => {
                active.push(TokenTree::Ident(Ident::new(text, false, span)));
            }
            TokenKind::RawIdent => {
                active.push(TokenTree::Ident(Ident::new(&text[2..], true, span)));
            }
            TokenKind::Literal { .. } => {
                active.push(TokenTree::Literal(Literal::new(text.to_owned(), span)));
            }
            TokenKind::Lifetime { .. } => {
                active.push(TokenTree::Punct(Punct::new('\'', Spacing::Joint, span)));
                active.push(TokenTree::Ident(Ident::new(&text[1..], false, span)));
            }

            TokenKind::Semi
            | TokenKind::Comma
            | TokenKind::Dot
            | TokenKind::At
            | TokenKind::Pound
            | TokenKind::Tilde
            | TokenKind::Question
            | TokenKind::Colon
            | TokenKind::Dollar
            | TokenKind::Eq
            | TokenKind::Not
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::Minus
            | TokenKind::And
            | TokenKind::Or
            | TokenKind::Plus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Caret
            | TokenKind::Percent => {
                let peek = iter.peek().map(|t| t.0).unwrap_or(TokenKind::Unknown);
                let spacing = match peek {
                    TokenKind::Semi
                    | TokenKind::Comma
                    | TokenKind::Dot
                    | TokenKind::At
                    | TokenKind::Pound
                    | TokenKind::Tilde
                    | TokenKind::Question
                    | TokenKind::Colon
                    | TokenKind::Dollar
                    | TokenKind::Eq
                    | TokenKind::Not
                    | TokenKind::Lt
                    | TokenKind::Gt
                    | TokenKind::Minus
                    | TokenKind::And
                    | TokenKind::Or
                    | TokenKind::Plus
                    | TokenKind::Star
                    | TokenKind::Slash
                    | TokenKind::Caret
                    | TokenKind::Percent => Spacing::Joint,
                    _ => Spacing::Alone,
                };

                let ch = text.chars().next().unwrap();
                active.push(TokenTree::Punct(Punct::new(ch, spacing, span)));
            }

            TokenKind::Unknown => return Err(()),
        }
    }

    if !stack.is_empty() {
        return Err(());
    }

    Ok(TokenStream { inner: active })
}

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
        if !is_id_start(first) {
            return false;
        }
        for ch in chars {
            if !is_id_continue(ch) {
                return false;
            }
        }
        true
    }

    if !ident_ok(validate) {
        panic!("{:?} is not a valid Ident", string);
    }
}
