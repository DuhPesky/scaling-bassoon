#![allow(clippy::type_complexity)]
use daggy::{petgraph::Direction, Dag, EdgeIndex, NodeIndex, Walker};
use num_traits::Float;
use std::{fmt, fs::File, io::Write};

#[derive(Clone, Copy)]
pub struct Value<T>
where
    T: Float + fmt::Display,
{
    pub data: T,
    grad: T,
    op: Option<Op>,
    id: &'static str,
}

impl<T> Value<T>
where
    T: Float + fmt::Display,
{
    pub fn new(data: T, grad: T, op: Option<Op>, id: &'static str) -> Self {
        Self { data, grad, op, id }
    }
}

impl<T> fmt::Display for Value<T>
where
    T: Float + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Value {} {{ data: {:.4}, grad: {:.4} }}",
            self.id, self.data, self.grad
        )
    }
}

pub struct Empty;

impl fmt::Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

#[derive(Copy, Clone)]
pub enum Op {
    ADD,
    MUL,
    SUB,
    POW,
    TANH,
}

impl Op {
    fn as_str(&self) -> &'static str {
        match self {
            Op::ADD => "+",
            Op::MUL => "*",
            Op::SUB => "-",
            Op::POW => "^",
            Op::TANH => "tanh",
        }
    }
}

impl<T> GradFn<T> for Op
where
    T: Float + fmt::Display,
{
    fn backward(&self) -> Box<dyn Fn(&mut Dag<Value<T>, Empty>, NodeIndex)> {
        let op = |dag: &mut Dag<Value<T>, Empty>, out_idx: NodeIndex| {
            let out_node = dag
                .node_weight(out_idx)
                .expect("Unable to borrow parent node");
            let out_op = out_node
                .op
                .expect("Function calling this should not allow nodes with no op");
            let out_grad = out_node.grad;

            let dx = match out_op {
                Op::ADD => |_: T, out: T, _: T| out,
                Op::MUL => |_: T, out: T, other: T| other * out,
                Op::POW => {
                    |self_: T, out: T, other: T| (other * self_.powf(other - T::one())) * out
                }
                Op::TANH => |_: T, out: T, _: T| out,
                Op::SUB => |_: T, out: T, _: T| -out,
            };

            // Daggy calls them parent because they are pointing to out
            let mut input_idxs = dag.parents(out_idx);

            let input_one = input_idxs
                .walk_next(dag)
                .expect("Node doesn't have at least 1 input");
            let input_two = input_idxs.walk_next(dag);

            let l_data = dag
                .node_weight(input_one.1)
                .expect("Unable to borrow node")
                .data;

            let r_data: T;
            if let Some(input_two) = input_two {
                r_data = dag
                    .node_weight(input_two.1)
                    .expect("Unable to borrow node")
                    .data;

                let mut r = dag
                    .node_weight_mut(input_two.1)
                    .expect("Unable to mut borrow node");

                r.grad = r.grad + dx(l_data, out_grad, l_data);
                // println!("{} Grad -> {}", r.id, r.grad);
            } else {
                r_data = T::zero();
            }

            let mut l = dag
                .node_weight_mut(input_one.1)
                .expect("Unable to mut borrow node");

            l.grad = l.grad + dx(l_data, out_grad, r_data);
            // println!("{} Grad -> {}", l.id, l.grad);
        };

        Box::new(op)
    }
}

pub trait GradFn<T>
where
    T: Float + fmt::Display,
{
    fn backward(&self) -> Box<dyn Fn(&mut Dag<Value<T>, Empty>, NodeIndex)>;
}

pub struct ComputationGraph<T>
where
    T: Float + fmt::Display,
{
    graph: Dag<Value<T>, Empty>,
}

impl<T> ComputationGraph<T>
where
    T: Float + fmt::Display,
{
    pub fn new() -> Self {
        Self { graph: Dag::new() }
    }

    // Adds a scalar node to the DAG and returns its NodeIndex
    pub fn new_value(&mut self, data: T, id: &'static str) -> NodeIndex {
        let grad = T::zero();
        let new_node = Value::new(data, grad, None, id);
        self.graph.add_node(new_node)
    }

    pub fn loss_computation(&mut self, pred: &[NodeIndex], truth: &[f64]) -> NodeIndex {
        let mut result = self.new_value(T::zero(), "result");

        for (y_truth, ypred) in truth.iter().zip(pred.iter()) {
            let y_truth = self.new_value(T::from(*y_truth).unwrap(), "y");

            let error = self.new_computation(y_truth, Some(*ypred), Op::SUB, "E");

            let two = self.new_value(T::from(2.0).unwrap(), "2");
            let sq_error = self.new_computation(error, Some(two), Op::POW, "SE");

            result = self.new_computation(sq_error, Some(result), Op::ADD, "L");

            println!(
                "(y_truth - y_pred)^2 = ({} - {})^2 = {}",
                self.get_node_data(y_truth),
                self.get_node_data(*ypred),
                self.get_node_data(sq_error)
            );
        }

        result
    }

    pub fn gradient_descent_step(&mut self, params: &[NodeIndex]) {
        for param in params.iter() {
            let mut param_node = self.graph.node_weight_mut(*param).unwrap();
            param_node.data = param_node.data + (param_node.grad * T::from(-0.05).unwrap());
        }
    }

    pub fn backward_one_level(&mut self, out: NodeIndex) {
        let out_node = self.graph.node_weight(out).expect("Unable to borrow node");

        if out_node.op.is_none() {
            return;
        }

        let dx: Box<dyn Fn(&mut Dag<Value<T>, Empty>, NodeIndex)> =
            out_node.op.expect("unable to borrow op").backward();

        self.set_rootgrad_to_one();
        dx(&mut self.graph, out);
    }

    fn set_rootgrad_to_one(&mut self) {
        let mut root: NodeIndex<u32> = NodeIndex::new(0);

        for (i, node) in self.graph.raw_nodes().iter().enumerate() {
            if node.next_edge(Direction::Outgoing) == EdgeIndex::end() {
                root = NodeIndex::new(i);
                break;
            }
        }

        self.set_node_grad(root, T::one())
    }

    pub fn backward_full_pass(&mut self) {
        self.set_rootgrad_to_one();
        for i in (0..self.graph.node_count()).rev() {
            let out = NodeIndex::new(i);
            self.backward_one_level(out);
        }
    }

    // Given the index of 1 or 2 operands (Value<T> Nodes) apply an operation on their data and
    // store the result as a new node in the graph. Where the children are the 2 operands and the
    // parent is the result. Return the NodeIndex of the newly created parent.
    pub fn new_computation(
        &mut self,
        lhs: NodeIndex,
        rhs: Option<NodeIndex>,
        op: Op,
        id: &'static str,
    ) -> NodeIndex {
        let l_n = &self.graph[lhs];

        let data = match (op, rhs) {
            (Op::ADD, Some(rhs)) => l_n.data + self.graph[rhs].data,
            (Op::MUL, Some(rhs)) => l_n.data * self.graph[rhs].data,
            (Op::SUB, Some(rhs)) => l_n.data - self.graph[rhs].data,
            (Op::POW, Some(rhs)) => l_n.data.powf(self.graph[rhs].data),
            (Op::TANH, None) => l_n.data.tanh(),
            _ => panic!("Number of operands doesn't match operation requirements"),
        };

        let out = Value::new(data, T::zero(), Some(op), id);
        let out_idx = self.graph.add_node(out);

        self.graph
            .add_edge(lhs, out_idx, Empty)
            .expect("Unable to add edge between lhs and parent");

        if let Some(rhs) = rhs {
            self.graph
                .add_edge(rhs, out_idx, Empty)
                .expect("Unable to add edge between rhs and parent");
        }

        out_idx
    }

    pub fn set_node_grad(&mut self, node: NodeIndex, grad: T) {
        let n = self
            .graph
            .node_weight_mut(node)
            .expect("Unable to mut borrow node");
        n.grad = grad;
    }

    pub fn get_node_data(&self, node: NodeIndex) -> T {
        self.graph[node].data
    }

    pub fn write_dag_to_dot(&self, filename: &'static str) {
        let mut file = File::create("input.dot").unwrap();
        let mut content = String::new();

        content.push_str("digraph {\n");
        content.push_str(r#"rankdir="LR""#);
        content.push_str("\nnode [shape=record]");

        // Stored as: Node { weight, next: [outgoing, incoming]}
        // Node { weight: 0, next: [EdgeIndex(1), EdgeIndex(4294967295)] }
        // Write all the nodes and their labels (Operands and Operators)
        for (i, node) in self.graph.raw_nodes().iter().enumerate() {
            let data = format!("{:.4}", node.weight.data);
            let grad = format!("{:.4}", node.weight.grad);
            let label = node.weight.id;

            content.push('\n');
            content = format!(
                r#"{}    {} [label="{{ {} | data: {} | grad: {} }}"]"#,
                content, i, label, data, grad
            );

            if let Some(op) = node.weight.op {
                content.push('\n');
                content = format!(
                    r#"{}    {}999 [label="{}" shape=circle]"#,
                    content,
                    i,
                    op.as_str()
                );
            }
        }

        content.push('\n');

        // Draw edges from operands to operators
        let mut already_pointed_sources = Vec::<usize>::new();
        for edge in self.graph.raw_edges() {
            let source = edge.source().index();
            let target = edge.target().index();

            content.push('\n');
            // point source to targets op
            content = format!("{}    {} -> {}999", content, source, target);
            content.push('\n');

            if !already_pointed_sources.contains(&target) {
                // point targets op to target
                already_pointed_sources.push(target);
                content = format!("{}    {}999 -> {}", content, target, target);
            }
        }
        content.push_str("\n}");
        file.write_all(content.as_bytes())
            .expect("Unable to write to file");
        println!("-- Wrote to file: {} --", filename);
    }
}

impl<T> Default for ComputationGraph<T>
where
    T: Float + fmt::Display,
{
    fn default() -> Self {
        Self { graph: Dag::new() }
    }
}
