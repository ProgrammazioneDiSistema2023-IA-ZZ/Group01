use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Reshape {
    node_name: String,
    input_name: String,
    output_name: String,
    input: Option<ArrayD<f32>>,
    shape: Vec<usize>,
}

impl Reshape {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {

        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();
        let parameter_name = node.input[1].to_owned();

        let shape = initializers.remove(parameter_name.as_str())
            .unwrap().iter().map(|&f| f as usize).collect();
        let input = initializers.remove(input_name.as_str());
        Self {
            node_name,
            input_name,
            output_name,
            input,
            shape,
        }
    }
}

impl Operator for Reshape {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {

        let input = self.input.clone().unwrap_or_else(
            || inputs.get(&self.input_name).expect("Input not found in the hashmap").clone()
        );

        let shape_size: usize = self.shape.iter().product();
        let input_size = input.len();

        if shape_size != input_size {
            return Err("Dimension is not correct for the number of data.".to_string());
        }

        let output_data = input.into_shape(self.shape.clone()).unwrap();

        Ok(output_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Node name: {}\nInput name: {}\nOutput name: {}",
            self.node_name, self.input_name, self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

