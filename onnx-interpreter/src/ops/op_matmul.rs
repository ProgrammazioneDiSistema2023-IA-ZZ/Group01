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

        let input1 = inputs.get(input_name1.as_str()).unwrap();
        let input2 = inputs.get(input_name2.as_str()).unwrap();

        if input1.shape()[1] != input2.shape()[0] {
            return Err("Shapes cannot be multiplied".to_string());
        }

        let m = input1.shape()[0];
        let n = input2.shape()[1];
        let common_dim = input1.shape()[1];

        // Perform matrix multiplication
        let pooled_dims = IxDyn(&[m, n]);
        let mut pooled_data = ArrayD::zeros(pooled_dims);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..common_dim {
                    sum += input1[[i, k]] * input2[[k, j]];
                }
                pooled_data[[i, j]] = sum;
            }
        }

        Ok(pooled_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Node name: {}\nInputs name: {} {}\nOutput names: {}",
            self.node_name,
            self.inputs_name[0],
            self.inputs_name[1],
            self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_name.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}