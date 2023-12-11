use super::op_operator::Operator;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct MatMul {
    node_name: String,
    inputs_name: Vec<String>,
    output_name: String,
}

impl MatMul {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
        let output_name = node.output[0].to_owned();
        let node_name = node.name.to_owned();
        Self {
            node_name,
            inputs_name,
            output_name,
        }
    }
}

impl Operator for MatMul {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name1 = self.inputs_name[0].clone();
        let input_name2 = self.inputs_name[1].clone();

        let input1 = inputs.get(input_name1.as_str()).unwrap().clone();
        let input2 = inputs.get(input_name2.as_str()).unwrap().clone();

        let input1_2d = input1.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "input1 is not 2-dimensional".to_string())?;
        //Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
        let input2_2d = input2.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "input2 is not 2-dimensional".to_string())?;

        let (m, k_a) = input1_2d.dim();
        let (k_b, n) = input2_2d.dim();
        if k_a != k_b {
            return Err("The inner dimensions of A' and B' do not match".to_string());
        }

        let y = input1_2d.dot(&input2_2d);
        Ok(y.into_dyn())
    }

    fn to_string(&self, verbose: &bool) -> String {
        match verbose{
            true => format!(""),
            false => format!("🚀 Running node: {}", self.node_name)
        }
        /*format!(
            "Node name: {}\nInputs name: {} {}\nOutput names: {}",
            self.node_name,
            self.inputs_name[0],
            self.inputs_name[1],
            self.output_name
        )*/
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_name.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}