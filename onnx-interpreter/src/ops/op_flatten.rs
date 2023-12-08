use super::op_operator::Operator;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Flatten {
    node_name: String,
    input_name: String,
    output_name: String,
    axis: i64,
}

impl Flatten {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {

        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name= node.output[0].to_owned();
        let mut axis= 1;
        if !node.attribute.is_empty(){
            axis = node.attribute[0].i.to_owned();
        }
        Self { node_name, input_name, output_name, axis }
    }
}

impl Operator for Flatten {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_tensor = inputs.get(&self.input_name)
            .ok_or("Input tensor not found")?;

        let rank = input_tensor.ndim();

        // Normalize axis value
        let axis = if self.axis < 0 {
            rank as i64 + self.axis
        } else {
            self.axis
        } as usize;

        if axis > rank {
            return Err("Axis is out of bounds for the tensor shape".to_string());
        }

        // Calculate the new shape
        let first_dim: usize = if axis == 0 { 1 } else { input_tensor.shape()[..axis].iter().product() };
        let second_dim: usize = input_tensor.shape()[axis..].iter().product();
        let new_shape = IxDyn(&[first_dim, second_dim]);

        // Create the output tensor with the same data but new shape
        let output_tensor = input_tensor.clone().into_shape(new_shape)
            .map_err(|_| "Error reshaping tensor".to_string())?;

        Ok(output_tensor)
    }

    fn to_string(&self) -> String {
        format!("Node name: {}\nInput name: {}\nOutput name: {}",
                self.node_name, self.input_name, self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}