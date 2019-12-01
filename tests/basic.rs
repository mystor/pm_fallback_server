extern crate proc_macro;

use pm_fallback_server::run_server;

#[test]
fn basic() {
    run_server(|| {
        let ts = proc_macro::TokenStream::new();
        println!("{}", ts);

        let ts2: proc_macro::TokenStream = "hello world, this is Nika!".parse().unwrap();
        println!("{}", ts2);
    });
}
