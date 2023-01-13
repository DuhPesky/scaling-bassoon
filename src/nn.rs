use crate::engine::Value;
use num_traits::Float;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use std::{array, fmt};

// W = nin = # of weights
struct Neuron<T>
where
    T: Float + fmt::Display,
{
    weights: Vec<Value<T>>,
    b: Value<T>,
    non_lin: bool,
}

impl<T> Neuron<T>
where
    T: Float + fmt::Display,
{
    fn new(num_weights: usize, non_lin: bool) -> Self {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = thread_rng();

        let mut weights = Vec::new();
        for _ in 0..num_weights {
            weights.push(Value::new(
                T::from(between.sample(&mut rng)).unwrap(),
                T::zero(),
                None,
                "",
            ));
        }

        let b = T::from(between.sample(&mut rng)).unwrap();

        Self {
            weights,
            b: Value::new(b, T::zero(), None, ""),
            non_lin,
        }
    }

    fn get_activation(&self, x: &[isize]) -> T {
        let activation = self
            .weights
            .iter()
            .zip(x.iter())
            .fold(T::zero(), |acc, (w_i, x_i)| {
                acc + w_i.data * T::from(*x_i).expect("Unable to convert x_i to T")
            })
            + self.b.data;

        match self.non_lin {
            true => activation.tanh(),
            false => activation,
        }
    }
}

// N = nouts = # of neurons
struct Layer<T>
where
    T: Float + fmt::Display,
{
    neurons: Vec<Neuron<T>>,
}

impl<T> Layer<T>
where
    T: Float + fmt::Display,
{
    fn new(num_weights: &[usize]) -> Self {
        let mut neurons = Vec::new();
        for i in num_weights {
            neurons.push(Neuron::new(num_weights[*i], true));
        }

        Self { neurons }
    }

    fn get_activations(&self, x: &[isize]) -> Vec<T> {
        let mut activations = Vec::new();
        for neuron in &self.neurons {
            activations.push(neuron.get_activation(x));
        }

        activations
    }
}

// L = nouts = # of layers
// layers is 2d array, where the outer array is the layer number and the inner array is the # of
// weights for that neuron in that layer
pub struct MLP<T, const L: usize>
where
    T: Float + fmt::Display,
{
    layers: [Layer<T>; L],
}

impl<T, const L: usize> MLP<T, L>
where
    T: Float + fmt::Display,
{
    pub fn new(neuron_layer_count: &[&[usize]; L]) -> Self {
        Self {
            layers: array::from_fn(|i| Layer::new(neuron_layer_count[i])),
        }
    }
}
