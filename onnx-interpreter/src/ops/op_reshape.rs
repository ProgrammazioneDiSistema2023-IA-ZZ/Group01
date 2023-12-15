use super::op_operator::Operator;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use prettytable::{format, row, Table};
use crate::parser_code::onnx_ml_proto3::NodeProto;
use colored::Colorize;
use indexmap::IndexMap;

pub struct Reshape {
    op_type: String,
    node_name: String,
    input_name: Option<String>,
    output_name: String,
    allow_zero: i64,
    shape_initializer: IndexMap<String, Vec<isize>>,
    data_initializer: Option<IndexMap<String, ArrayD<f32>>>,
    flag_reshape_with_no_network_input: bool,
}

impl Reshape {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let parameter_name = node.input[1].to_owned();

        let opt_allow_zero = node.attribute.iter().find(|attr| attr.name == "allow_zero");
        let allow_zero = match opt_allow_zero{
            Some(x) => x.i,
            None => 0
        };

        let mut shape_initializer = IndexMap::new();
        let shape : Vec<isize>= initializers.remove(parameter_name.as_str())
            .unwrap().iter().map(|&f| f as isize).collect();
        shape_initializer.insert(parameter_name, shape.to_owned());

        let mut data_initializer: Option<IndexMap<String, ArrayD<f32>>> = None;
        let mut input_name = None;
        let mut flag_reshape_with_no_network_input = false;
        match initializers.remove(node.input[0].as_str()){
            Some(v) => {
                data_initializer = Some(IndexMap::from([
                    (node.input[0].clone(), v.to_owned())
                ]));
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
    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let input = match &self.data_initializer{
            Some(v) => v.iter().collect::<Vec<_>>()[0].1,
            None => inputs.get(&self.input_name.clone().unwrap()).expect("Input not found in the hashmap")
        };

        let mut target_shape = self.shape_initializer.iter().collect::<Vec<_>>()[0].1.clone();
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

        let output_data = input.clone().into_shape(new_shape_usize).unwrap();

        Ok(vec![output_data])
    }

    fn get_inputs(&self) -> Vec<String> {
        if (self.input_name.is_some()){
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

    fn get_initializers_arr(&self) -> Vec<(String, ArrayD<f32>)> {
        if self.data_initializer.is_some(){
            [
                self.data_initializer.clone().unwrap().iter().map(|x|
                (x.0.clone(), x.1.to_owned())
            ).collect::<Vec<_>>().as_slice(),
                self.shape_initializer.iter().map(|x|
                    (x.0.clone(), ArrayD::from_shape_vec(IxDyn(&vec![x.1.into_iter().map(|x| *x as f32).collect::<Vec<_>>().len()]),
                                                         x.1.into_iter().map(|x| *x as f32).collect::<Vec<_>>()
                    ).unwrap().to_owned())
                ).collect::<Vec<_>>().as_slice()
            ].concat()
        }
        else{
            self.shape_initializer.iter().map(|x|
                (x.0.clone(), ArrayD::from_shape_vec(IxDyn(&vec![x.1.into_iter().map(|x| *x as f32).collect::<Vec<_>>().len()]),
                                                     x.1.into_iter().map(|x| *x as f32).collect::<Vec<_>>()
                    ).unwrap().to_owned())
            ).collect::<Vec<_>>()
        }
    }
}

