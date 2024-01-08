use crate::errors::OnnxError;

use super::op_operator::{Initializer, Operator};
use ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Reshape {
    op_type: String,
    node_name: String,
    input_name: Option<String>,
    output_name: String,
    allow_zero: i64,
    shape_initializer: Vec<Initializer>,
    data_initializer: Option<Vec<Initializer>>,
    flag_reshape_with_no_network_input: bool,
}

impl Reshape {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let parameter_name = node.input[1].to_owned();

        let opt_allow_zero = node.attribute.iter().find(|attr| attr.name == "allow_zero");
        let allow_zero = match opt_allow_zero{
            Some(x) => x.i,
            None => 0
        };

        let mut shape_initializer = Vec::new();
        let shape : Vec<isize>= initializers.remove(parameter_name.as_str())
            .unwrap().iter().map(|&f| f as isize).collect();
        shape_initializer.push(Initializer::new(parameter_name, Array::from_shape_vec(IxDyn(&[2,1]), shape.into_iter().map(|x| x as f32).collect()).unwrap()));

        let mut data_initializer = None;
        let mut input_name = None;
        let mut flag_reshape_with_no_network_input = false;
        match initializers.remove(node.input[0].as_str()){
            Some(v) => {
                data_initializer = Some(vec![
                    Initializer::new(node.input[0].clone(), v.to_owned())
                ]);
                flag_reshape_with_no_network_input=true;
            },
            None =>{
                input_name = Some(node.input[0].clone());
            }
        }

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
            allow_zero,
            shape_initializer,
            data_initializer,
            flag_reshape_with_no_network_input
        }
    }
}

impl Operator for Reshape {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input = match &self.data_initializer{
            Some(v) => v[0].get_value(),
            None => inputs.get(&self.input_name.clone().unwrap())
                .ok_or_else(||
                    OnnxError::TensorNotFound("Input tensor not found".to_string())).unwrap()
        };

        let mut target_shape = self.shape_initializer[0].get_value().clone().into_raw_vec().into_iter().map(|x| x as isize).collect::<Vec<_>>();
        if !self.flag_reshape_with_no_network_input{
            target_shape[0] *= input.shape()[0] as isize;
        }
        let mut dim_to_infer = None;

        for (i, dim) in target_shape.iter_mut().enumerate(){
            if *dim == -1 {
                if dim_to_infer.is_some(){
                    return Err(OnnxError::ShapeMismatch("Too much dimensions to infer".to_string()));
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
                return Err(OnnxError::ShapeMismatch("Cannot infer shape due to incompatible dimensions".to_string()));
            }

            target_shape[i] = (input.len() as isize) / product_of_dimensions;
        }

        let shape_size: isize = target_shape.iter().product();
        let input_size = input.len();

        if shape_size != input_size as isize {
            return Err(OnnxError::ShapeMismatch("Dimension is not correct for the number of data".to_string()));
        }

        // Convert isize dimensions to usize for reshape
        let new_shape_usize: Vec<usize> = target_shape
            .iter()
            .map(|&dim| dim as usize)
            .collect();

        let output_data = input.clone().into_shape(new_shape_usize).unwrap();

        Ok(vec![output_data])
    }

    fn get_inputs(&self) -> Vec<String> {
        if self.input_name.is_some(){
            vec![self.input_name.clone().unwrap().clone()]
        }
        else{
            vec![]
        }
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

    fn get_initializers_arr(&self) -> Vec<Initializer> {
        if self.data_initializer.is_some(){
            [
                self.data_initializer.clone().unwrap().as_slice(),
                self.shape_initializer.as_slice()
            ].concat()
        }
        else{
            self.shape_initializer.clone()
        }
    }
}

