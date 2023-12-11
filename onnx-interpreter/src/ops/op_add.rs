use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Add {
    node_name: String,
    inputs: Vec<String>,
    output_name: String,
    parameter: Option<ArrayD<f32>>,
}

impl Add {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {

        let inputs:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
        let node_name= node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let parameter = initializers.remove(&inputs[1]);

        Self {
            node_name,
            inputs,
            output_name,
            parameter,
        }
    }
}

impl Operator for Add {

    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let a = inputs.get(&self.inputs[0])
            .ok_or("First input tensor not found")?;
        let b = match inputs.get(&self.inputs[1]) {
            Some(tensor) => tensor,
            None => self.parameter.as_ref().expect("Parameter tensor not found"),
        };

        let result = a + b;
        Ok(result)
    }


    fn to_string(&self, verbose: &bool) -> String {
        match verbose{
            true => format!(""),
            false => format!("🚀 Running node: {}", self.node_name)
        }
        /*format!("Node name: {}\nInputs name: {} {}\nOutput name: {}",
                self.node_name, self.inputs[0], self.inputs[1], self.output_name)*/
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}