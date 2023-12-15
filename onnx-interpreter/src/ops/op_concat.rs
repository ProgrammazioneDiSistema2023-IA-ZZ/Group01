use super::op_operator::Operator;
use ndarray::{ArrayD, Axis, concatenate, IxDyn};
use std::collections::HashMap;
use std::ops::Index;
use indexmap::IndexMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Concat {
    op_type: String,
    node_name: String,
    inputs_names: Vec<String>,
    output_name: String,
    axis: i64,
}

impl Concat {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let mut inputs_names:Vec<String> = vec![];
        for inp in &node.input{
            inputs_names.push(inp.clone());
        }
        let node_name= node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let opt_axis = node.attribute.iter().find(|attr| attr.name == "axis");
        let axis = match opt_axis{
            Some(x) => x.i,
            None => 0
        };

        Self {
            op_type,
            node_name,
            inputs_names,
            output_name,
            axis,
        }
    }
}

impl Operator for Concat {

    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let mut v = vec![];
        for inp in &self.inputs_names{
            let mut t = inputs.get(inp).ok_or("Tensor not found")?;
            if t.ndim()==0{
                return Err("Concat: one input has an empty shape".to_string());
            }
            if self.axis as usize>= t.ndim(){
                let mut new_shape = Vec::with_capacity(self.axis as usize + 1);
                new_shape.extend_from_slice(t.shape());
                new_shape.resize(self.axis as usize +1, 1);
                v.push(t.view().into_shape(IxDyn(&new_shape)).expect("Failed to reshape array").to_owned());
            }
            else{
                v.push(t.to_owned());
            }
        }
        let views = v.iter().map(|t| t.view()).collect::<Vec<_>>();

        let result = concatenate(Axis(self.axis as usize), &views).unwrap();
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
        vec![]
    }

}