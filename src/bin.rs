use micrograd::engine::{ComputationGraph, Op};
use micrograd::nn::MLP;

fn main() {
    let cg = ComputationGraph::<f64>::new();
    let x = [2.0, 3.0, -1.0];
    let n: MLP<f64, 1> = MLP::new(&[&[4, 4, 1]]);
    // cg.backward_full_pass();
    cg.write_dag_to_dot("input.dot");
}
