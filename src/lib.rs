//! THIS IS SUPER UNSTABLE
//! THIS IS PRIMARIALY INTENDED AS AN EXPERIMENT

#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]

extern crate proc_macro;

use std::rc::Rc;
use proc_macro::bridge::{server, TokenTree};

#[macro_use]
mod strnom;
mod fallback;

struct Diagnostic {}

use fallback::{TokenStream, TokenStreamBuilder, TokenStreamIter, Group, Punct, Ident, Literal, SourceFile, Span};

// The fallback server
struct Server {}

type TokenTreeT = TokenTree<Group, Punct, Ident, Literal>;

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

