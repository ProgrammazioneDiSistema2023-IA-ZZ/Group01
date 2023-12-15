use super::op_operator::Operator;
use ndarray::{ArrayD, Axis, concatenate, IxDyn};
use std::collections::HashMap;
use std::ops::Index;
use indexmap::IndexMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Softmax {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
}

impl Softmax {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let input_name = node.input[0].to_owned();
        let node_name= node.name.to_owned();
        let output_name = node.output[0].to_owned();

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
        }
    }
}

impl Operator for Softmax {
    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let x = inputs.get(&self.input_name).ok_or("Input tensor X not found")?;
        let max = x.map_axis(Axis(1), |subarr| subarr.iter().cloned().fold(f32::MIN, f32::max));
        let mut x = x - &max.insert_axis(Axis(1));

        x.mapv_inplace(f32::exp);

        let sum = x.map_axis(Axis(1), |subarr| subarr.iter().sum::<f32>());
        x /= &sum.insert_axis(Axis(1));
        Ok(vec![x.to_owned()])
    }


    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_names(&self) -> Vec<String> {
        vec![self.output_name.clone()]
    }

    fn get_node_name(&self) -> String {
        self.node_name.clone()
    }

    fn get_op_type(&self) -> String {
        self.op_type.clone()
    }

    fn get_initializers_arr(&self) -> Vec<(String, ArrayD<f32>)> {
        vec![]
    }
}