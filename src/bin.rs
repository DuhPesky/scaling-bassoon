use micrograd::engine::ComputationGraph;
use micrograd::nn::MLP;

fn main() {
    let mut cg = ComputationGraph::<f64>::new();

    const INPUT_LAYER: [f64; 3] = [2.0, 3.0, -1.0];
    // layer 1 has 4 neurons with 3 weights per neuron
    // if weights are greater than input size, then those weights are not used in calculation
    let layer_1 = [INPUT_LAYER.len(); 4];
    let layer_2 = [layer_1.len(); 4];
    let layer_3 = [layer_2.len(); 1];

    let n: MLP<3> = MLP::new(&[&layer_1, &layer_2, &layer_3], &mut cg);
    n.call(&INPUT_LAYER, &mut cg);
    // cg.backward_full_pass();
    cg.write_dag_to_dot("input.dot");
}
