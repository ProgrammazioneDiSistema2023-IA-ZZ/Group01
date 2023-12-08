use super::op_operator::Operator;
use ndarray::{ArrayD, Axis};
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct GlobalAveragePool {
    node_name: String,
    input_name: String,
    output_name: String,
}

impl GlobalAveragePool {
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

impl Operator for GlobalAveragePool {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        // Retrieve the input tensor
        let input_tensor = inputs.get(&self.input_name).ok_or("Input tensor not found")?;

        if input_tensor.ndim() < 3 {
            return Err("Input tensor must have at least 3 dimensions".to_string());
        }
        let axis_to_pool: Vec<_> = (2..input_tensor.ndim()).collect();

        let mut y = input_tensor.clone();
        for &axis in axis_to_pool.iter().rev() {
            y = y.mean_axis(Axis(axis)).unwrap();
        }

        for _ in axis_to_pool {
            let dim = y.ndim();
            y = y.insert_axis(Axis(dim));
        }

        Ok(y)
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