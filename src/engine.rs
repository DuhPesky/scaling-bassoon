#![allow(dead_code)]
#![allow(clippy::type_complexity)]
use daggy::{petgraph::graph::IndexType, Dag, NodeIndex, Walker};
use num_traits::Float;
use std::{fmt, fs::File, io::Write};

pub struct Value<T>
where
    T: Float + fmt::Display,
{
    data: T,
    grad: T,
    op: Option<Op>,
    grad_fn: Option<Box<dyn GradFn<T>>>,
    id: &'static str,
}

impl<T> Value<T>
where
    T: Float + fmt::Display,
{
    fn new(data: T, id: &'static str) -> Self {
        Value::with_all(data, data.abs() - data.abs(), None, None, id)
    }

    fn new_with_op(data: T, op: Op, id: &'static str) -> Self {
        Value::with_all(data, data.abs() - data.abs(), Some(op), None, id)
    }

    fn with_all(
        data: T,
        grad: T,
        op: Option<Op>,
        grad_fn: Option<Box<dyn GradFn<T>>>,
        id: &'static str,
    ) -> Self {
        Self {
            data,
            grad,
            op,
            grad_fn,
            id,
        }
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
    // Outgrad is the already calculated gradient and the parent of an expression
    // do macro for these
    fn backward(&self) -> Box<dyn Fn(&mut Dag<Value<T>, Empty>, NodeIndex)> {
        let op = |dag: &mut Dag<Value<T>, Empty>, parent_idx: NodeIndex| {
            let out = dag
                .node_weight(parent_idx)
                .expect("Unable to borrow parent node");

            let out_grad = out.grad;
            let out_op = out.op.expect("No Op on given parent index");

            let dx = match out_op {
                Op::ADD => |_: T, out: T, _: T| out,
                Op::MUL => |_: T, out: T, other: T| other * out,
                Op::POW => |self_: T, out: T, other: T| {
                    (other * self_.powf(other - T::from(1.0).unwrap())) * out
                },
                Op::TANH => |_: T, out: T, _: T| out,
                Op::SUB => |_: T, out: T, _: T| -out,
            };

            let mut child_indexes = dag.children(parent_idx);

            let child_one = child_indexes.walk_next(dag).expect("Child Index is wrong");
            let child_two = child_indexes.walk_next(dag).expect("Child index is wrong");

            let l_data = dag
                .node_weight(child_one.1)
                .expect("Unable to borrow node")
                .data;

            let r_data = dag
                .node_weight(child_two.1)
                .expect("Unable to borrow node")
                .data;

            let mut l = dag
                .node_weight_mut(child_one.1)
                .expect("Unable to mut borrow node");
            l.grad = l.grad + dx(l_data, out_grad, r_data);

            let mut r = dag
                .node_weight_mut(child_two.1)
                .expect("Unable to mut borrow node");
            r.grad = r.grad + dx(l_data, out_grad, l_data);
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
        let new_node = Value::new(data, id);
        self.graph.add_node(new_node)
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

        let node_result = match (op, rhs) {
            (Op::ADD, Some(rhs)) => l_n.data + self.graph[rhs].data,
            (Op::MUL, Some(rhs)) => l_n.data * self.graph[rhs].data,
            (Op::SUB, Some(rhs)) => l_n.data - self.graph[rhs].data,
            (Op::POW, Some(rhs)) => l_n.data.powf(self.graph[rhs].data),
            (Op::TANH, None) => l_n.data.tanh(),
            _ => panic!("Number of operands doesn't match operation requirements"),
        };

        // let backward: Box<dyn Fn(T, T, Option<T>) -> (T, Option<T>)> = op.backward();

        let out = Value::new_with_op(node_result, op, id);
        // let out = Value::with_all(node_result, )
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

    pub fn write_dag_to_dot(&self, filename: &'static str) {
        let mut file = File::create("input.dot").unwrap();
        let mut content = String::new();

        content.push_str("digraph {\n");
        content.push_str(r#"rankdir="LR""#);
        content.push_str("\nnode [shape=record]");

        // Stored as: Node { weight, next: [outgoing, incoming]}
        // Node { weight: 0, next: [EdgeIndex(1), EdgeIndex(4294967295)] }
        // Write all the nodes and their labels (Operands and Operators)
        for (index, node) in self.graph.raw_nodes().iter().enumerate() {
            let data = format!("{:.4}", node.weight.data);
            let grad = format!("{:.4}", node.weight.grad);
            let label = node.weight.id;

            content.push('\n');
            content = format!(
                r#"{}    {} [label="{{ {} | data: {} | grad: {} }}"]"#,
                content, index, label, data, grad
            );

            if let Some(op) = node.weight.op {
                content.push('\n');
                content = format!(
                    r#"{}    {}999 [label="{}" shape=circle]"#,
                    content,
                    index,
                    op.as_str()
                );
            }
        }

        content.push('\n');

        // Draw edges from operands to operators
        let mut already_pointed_sources = Vec::<usize>::new();
        for (i, edge) in self.graph.raw_edges().iter().enumerate() {
            let index = i.index();
            let source = edge.source().index();
            let target = edge.target().index();

            // println!("Edge: {:?}", edge);
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
