use micrograd::engine::ComputationGraph;
use micrograd::nn::MLP;

fn main() {
    let mut cg = ComputationGraph::<f64>::new();

    const TRUTH: [f64; 4] = [1.0, -1.0, -1.0, 0.0];
    const INPUTS: [[f64; 3]; 4] = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    // layer 1 has 4 neurons with 3 weights per neuron
    // if weights are greater than input size, then those weights are not used in calculation
    let layer_1 = [INPUTS[0].len(); 4];
    let layer_2 = [layer_1.len(); 4];
    let layer_3 = [layer_2.len(); 1];

    let n: MLP<3> = MLP::new(&[&layer_1, &layer_2, &layer_3], &mut cg);
    let params = n.parameters();

    for step in 0..1 {
        // forward pass
        let mut ypred = Vec::new();
        for input in &INPUTS {
            ypred.push(n.call(input, &mut cg));
        }
        let loss = cg.loss_computation(&ypred, &TRUTH);
        cg.backward_full_pass();
        cg.gradient_descent_step(&params);
        println!("step: {}, loss: {}", step, cg.get_node_data(loss));
    }

    cg.write_dag_to_dot("input.dot");
}
