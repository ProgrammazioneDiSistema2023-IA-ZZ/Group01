use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Reshape {
    node_name: String,
    input_name: String,
    output_name: String,
    input: Option<ArrayD<f32>>,
    allow_zero: i64,
    shape: Vec<isize>,
    flag_reshape_with_no_network_input: bool,
}

impl Reshape {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {

        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();
        let parameter_name = node.input[1].to_owned();

        let opt_allow_zero = node.attribute.iter().find(|attr| attr.name == "allow_zero");
        let allow_zero = match opt_allow_zero{
            Some(x) => x.i,
            None => 0
        };

        let shape = initializers.remove(parameter_name.as_str())
            .unwrap().iter().map(|&f| f as isize).collect();
        let input = initializers.remove(input_name.as_str());

        let flag_reshape_with_no_network_input = if input.is_some(){ true } else { false };
        Self {
            node_name,
            input_name,
            output_name,
            input,
            allow_zero,
            shape,
            flag_reshape_with_no_network_input
        }
    }
}

impl Operator for Reshape {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {

        let input = self.input.clone().unwrap_or_else(
            || inputs.get(&self.input_name).expect("Input not found in the hashmap").clone()
        );

        let mut target_shape = self.shape.clone();
        if !self.flag_reshape_with_no_network_input{
            target_shape[0] *= input.shape()[0] as isize;
        }
        let mut dim_to_infer = None;

        for (i, dim) in target_shape.iter_mut().enumerate(){
            if *dim == -1 {
                if dim_to_infer.is_some(){
                    return Err("Too much dimensions to infer.".to_string());
                }
                dim_to_infer = Some(i);
            } else if *dim == 0{
                if self.allow_zero == 0{
                    *dim = input.shape()[i] as isize;
                }
            }
        }

        // Handle negative dimension (inferred dimension)
        if let Some(i) = dim_to_infer{
            let product_of_dimensions: isize = target_shape.iter().filter(|&&dim| dim != -1).product();

            if input.len() as isize % product_of_dimensions != 0 {
                return Err("Cannot infer shape due to incompatible dimensions".to_string());
            }

            target_shape[i] = (input.len() as isize) / product_of_dimensions;
        }

        let shape_size: isize = target_shape.iter().product();
        let input_size = input.len();

        if shape_size != input_size as isize {
            return Err("Dimension is not correct for the number of data.".to_string());
        }

        // Convert isize dimensions to usize for reshape
        let new_shape_usize: Vec<usize> = target_shape
            .iter()
            .map(|&dim| dim as usize)
            .collect();

        let output_data = input.into_shape(new_shape_usize).unwrap();

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

