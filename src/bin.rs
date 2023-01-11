use micrograd::engine::{ComputationGraph, Op};
fn main() {
    let mut cg = ComputationGraph::<f64>::new();
    let a = cg.new_value(2.0, "a");
    let b = cg.new_value(-3.0, "b");
    let c = cg.new_value(10.0, "c");

    let e = cg.new_computation(a, Some(b), Op::MUL, "e");
    let d = cg.new_computation(e, Some(c), Op::ADD, "d");

    let f = cg.new_value(-2.0, "f");
    let l = cg.new_computation(d, Some(f), Op::MUL, "L");

    cg.write_dag_to_dot("input.dot");
}
