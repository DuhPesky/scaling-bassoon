use crate::engine::{ComputationGraph, Op};
use daggy::NodeIndex;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use std::{array, fmt};

// W = nin = # of weights
struct Neuron {
    weights: Vec<NodeIndex>,
    b: NodeIndex,
    non_lin: bool,
}

impl Neuron {
    fn new(num_weights: usize, non_lin: bool, cg: &mut ComputationGraph<f64>) -> Self {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = thread_rng();

        let mut weights = Vec::new();
        for _ in 0..num_weights {
            weights.push(cg.new_value(between.sample(&mut rng), "w"));
        }

        let b = between.sample(&mut rng);

        Self {
            weights,
            b: cg.new_value(b, "b"),
            non_lin,
        }
    }

    // x is inputs to neuron
    fn call(&self, x: &[NodeIndex], cg: &mut ComputationGraph<f64>) -> NodeIndex {
        let mut xw_sum = cg.new_value(0.0, "");

        for (w_i, x_i) in self.weights.iter().zip(x.iter()) {
            // multiply weight to input
            let xw_product = cg.new_computation(*x_i, Some(*w_i), Op::MUL, "xi*wi");
            // sum activation of that input to the last one
            xw_sum = cg.new_computation(xw_sum, Some(xw_product), Op::ADD, "");
        }

        let mut xw_bias_sum = cg.new_computation(xw_sum, Some(self.b), Op::ADD, "xw+b");

        if self.non_lin {
            xw_bias_sum = cg.new_computation(xw_bias_sum, None, Op::TANH, "")
        }

        xw_bias_sum
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let node_indices = self
            .weights
            .iter()
            .map(|x| x.index())
            .collect::<Vec<usize>>();

        write!(
            f,
            "Neuron {{ weights: {:?}, b: {:?}, non_lin: {} }}",
            node_indices,
            self.b.index(),
            self.non_lin
        )
    }
}

// N = nouts = # of neurons
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(weights: &[usize], cg: &mut ComputationGraph<f64>) -> Self {
        let mut neurons = Vec::new();

        for i in weights {
            neurons.push(Neuron::new(*i, true, cg));
        }

        Self { neurons }
    }

    fn call(&self, x: &[NodeIndex], cg: &mut ComputationGraph<f64>) -> Vec<NodeIndex> {
        let mut neuron_activations = Vec::new();

        for neuron in &self.neurons {
            neuron_activations.push(neuron.call(x, cg));
        }

        neuron_activations
    }

    fn neuron_count(&self) -> usize {
        self.neurons.len()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Layer {{ neurons: [")?;
        for neuron in &self.neurons {
            write!(f, "{}, ", neuron)?;
        }
        write!(f, "] }}")
    }
}

// L = nouts = # of layers
// layers is 2d array, where the outer array is the layer number and the inner array is the # of
// weights for that neuron in that layer
pub struct MLP<const L: usize> {
    layers: [Layer; L],
}

impl<const L: usize> MLP<L> {
    pub fn new(neuron_layer_count: &[&[usize]; L], cg: &mut ComputationGraph<f64>) -> Self {
        // Layer 0: W_0 W_1 W_2 .... W_N
        // Layer 1: W_0 W_1 W_2 .... W_N
        // Layer 2: W_0 W_1 W_2 .... W_N
        // Layer N: W_0 W_1 W_2 .... W_N
        let layers = array::from_fn(|i| Layer::new(neuron_layer_count[i], cg));
        for lay in &layers {
            println!("{}", lay);
        }
        Self { layers }
    }

    pub fn call(&self, x: &[f64], cg: &mut ComputationGraph<f64>) {
        // initialize with inputs
        let mut layer_activations = x
            .iter()
            .map(|x| cg.new_value(*x, "x"))
            .collect::<Vec<NodeIndex>>();

        for layer in &self.layers {
            println!("layer: {}", layer.neuron_count());
            layer_activations = layer.call(&layer_activations, cg);
        }
    }
}
