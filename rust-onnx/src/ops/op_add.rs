use crate::errors::OnnxError;

use super::op_operator::{Initializer, Operator};
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Add {
    op_type: String,
    node_name: String,
    inputs_names: Vec<String>,
    output_name: String,
    initializers: Option<Vec<Initializer>>,
}

impl Add {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        //let inputs_names:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
        let node_name= node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let initializer_value = initializers.remove(&node.input[1]);
        let (initializers_vec, inputs_names) = match &initializer_value {
            Some(v) => {
                (Some(vec![Initializer::new(node.input[1].to_owned(), v.clone())]), vec![node.input[0].to_owned()])
            },
            None => {
                (None, vec![node.input[0].to_owned(), node.input[1].to_owned()])
            }
        };

        Self {
            op_type,
            node_name,
            inputs_names,
            output_name,
            initializers: initializers_vec,
        }
    }
}

impl Operator for Add {

    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let a = inputs.get(&self.inputs_names[0])
            .ok_or_else(||
                OnnxError::TensorNotFound("First input tensor not found".to_string())).unwrap();
        let b = match &self.initializers{
            Some(v) => v[0].get_value(),
            None => inputs.get(&self.inputs_names[1]).ok_or_else(||
                OnnxError::TensorNotFound("First input tensor not found".to_string())).unwrap()
        };

        /*inputs.get(&self.inputs_names[1]) {
            Some(tensor) => { tensor},
            None => { self.parameter_value.as_ref().expect("Parameter tensor not found") },
        };*/
        //println!("{}", self.to_string(verbose));

        let result = a + b;
        Ok(vec![result])
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_names.clone()
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

    fn get_initializers_arr(&self) -> Vec<Initializer>{
        self.initializers.clone().unwrap_or_else(|| vec![])
    }


}