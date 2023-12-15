use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::ops::Index;
use indexmap::IndexMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Add {
    op_type: String,
    node_name: String,
    inputs_names: Vec<String>,
    output_name: String,
    initializers: Option<IndexMap<String, ArrayD<f32>>>,
}

impl Add {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        //let inputs_names:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
        let node_name= node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let initializer_value = initializers.remove(&node.input[1]);
        let (initializers_map, inputs_names) = match &initializer_value {
            Some(v) => {
                (Some(IndexMap::from([(node.input[1].to_owned(), initializer_value.unwrap().clone())])), vec![node.input[0].to_owned()])
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
            initializers: initializers_map,
        }
    }
}

impl Operator for Add {

    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let a = inputs.get(&self.inputs_names[0])
            .ok_or("First input tensor not found")?;
        let b = match &self.initializers{
            Some(hm) => hm.iter().collect::<Vec<_>>()[0].1,
            None => inputs.get(&self.inputs_names[1]).unwrap()
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

    fn get_initializers_arr(&self) -> Vec<(String, ArrayD<f32>)>{
        match &self.initializers{
            Some(hm) => hm.iter().map(|v| {
                (v.0.to_owned(), v.1.to_owned())
            }
            ).collect::<Vec<_>>(),
            None => vec![]
        }
    }


}