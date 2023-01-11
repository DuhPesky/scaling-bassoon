use micrograd::engine::{ComputationGraph, Op};
fn main() {
    let mut cg = ComputationGraph::<f64>::new();
    // let x1 = cg.new_value(2.0, "x1");
    // let x2 = cg.new_value(0.0, "x2");
    //
    // let w1 = cg.new_value(-3.0, "w1");
    // let w2 = cg.new_value(1.0, "w2");
    //
    // let b = cg.new_value(6.8813735870195432, "b");
    //
    // let x1w1 = cg.new_computation(x1, Some(w1), Op::MUL, "x1*w1");
    // let x2w2 = cg.new_computation(x2, Some(w2), Op::MUL, "x2*w2");
    // let xw_sum = cg.new_computation(x1w1, Some(x2w2), Op::ADD, "x1*w1 + x2*w2");
    //
    // let n = cg.new_computation(xw_sum, Some(b), Op::ADD, "n");
    // let o = cg.new_computation(n, None, Op::TANH, "o");
    //
    // cg.set_node_grad(n, 0.5);
    // cg.set_node_grad(o, 1.0);
    // let a = cg.new_value(1.0, "a");
    // let b = cg.new_computation(a, Some(a), Op::ADD, "b");
    // cg.backward_one_level(b);
    // cg.backward_full_pass();

    let a = cg.new_value(-2.0, "a");
    let b = cg.new_value(3.0, "b");
    let d = cg.new_computation(a, Some(b), Op::MUL, "d");
    let e = cg.new_computation(a, Some(b), Op::ADD, "e");
    let f = cg.new_computation(d, Some(e), Op::MUL, "f");

    cg.backward_full_pass();
    cg.write_dag_to_dot("input.dot");
}
