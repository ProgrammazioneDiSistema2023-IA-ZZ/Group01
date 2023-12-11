use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct ReLU {
    node_name: String,
    input_name: String,
    output_name: String,
}

impl ReLU {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();

        Self {
            node_name,
            input_name,
            output_name,
        }
    }
}

impl Operator for ReLU {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name = self.input_name.clone();
        let input = inputs.get(input_name.as_str()).unwrap();

        let output_data = input.mapv(|x| if x > 0.0 { x } else { 0.0 });

        Ok(output_data)
    }

    fn to_string(&self, verbose: &bool) -> String {
        match verbose{
            true => format!(""),
            false => format!("🚀 Running node: {}", self.node_name)
        }
        /*format!(
            "Node name: {}\nInput name: {}\nOutput name: {}",
            self.node_name, self.input_name, self.output_name
        )*/
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}