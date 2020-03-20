use std::collections::HashMap;
use std::rc::Rc;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Symbol {
    index: usize,
}

pub(crate) struct Interner {
    // XXX: We shouldn't need a `Rc` here, but it keeps the code simpler.
    names: HashMap<Rc<str>, Symbol>,
    strings: Vec<Rc<str>>,
}

impl Interner {
    pub(crate) fn new() -> Self {
        Interner {
            names: HashMap::new(),
            strings: Vec::new(),
        }
    }

    pub(crate) fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        let string = <Rc<str>>::from(string);
        let name = Symbol {
            index: self.strings.len(),
        };
        self.strings.push(string.clone());
        self.names.insert(string, name);
        name
    }

    pub(crate) fn get(&self, symbol: Symbol) -> &str {
        &self.strings[symbol.index]
    }
}
