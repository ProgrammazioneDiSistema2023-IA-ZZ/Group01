use crate::errors::OnnxError;

use super::op_operator::Operator;
use ndarray::{Array, ArrayD, Axis, concatenate, IxDyn};
use std::collections::HashMap;
use std::ops::Index;
use indexmap::IndexMap;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::parser_code::onnx_ml_proto3::NodeProto;
use ndarray_rand::rand::{SeedableRng, Rng, thread_rng};
use ndarray_rand::rand::prelude::StdRng;

pub struct Dropout {
    op_type: String,
    node_name: String,
    input_name: String,
    output_names: Vec<String>,
    ratio: f32,
    seed: Option<i64>
}

impl Dropout {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let input_name = node.input[0].to_owned();
        let node_name= node.name.to_owned();
        let mut output_names = vec![];
        for out in &node.output{
            output_names.push(out.clone());
        }
        let opt_ratio = node.attribute.iter().find(|attr| attr.name == "ratio");
        let ratio = match opt_ratio{
            Some(x) => x.f,
            None => 0.5
        };

        let opt_seed = node.attribute.iter().find(|attr| attr.name == "seed");
        let seed = match opt_seed{
            Some(x) => Some(x.i),
            None => None
        };

        Self {
            op_type,
            node_name,
            input_name,
            output_names,
            ratio,
            seed
        }
    }
}

impl Operator for Dropout {
    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let x = inputs.get(&self.input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found".to_string())).unwrap();
        let return_mask = self.output_names.len()==2;
        if self.ratio == 0.0 {
            if return_mask{
                return Ok(vec![x.to_owned(), ArrayD::ones(IxDyn(x.shape()))])
            }
            return Ok(vec![x.to_owned()])
        }
        let scale = 1.0/(1.0-self.ratio);
        let seed = self.seed.unwrap_or_else(|| {
            let mut rng = thread_rng();
            rng.gen()
        });
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mask: ArrayD<f32> = Array::random_using(x.raw_dim(), Uniform::new(0.0, 1.0), &mut rng);

        let mask_binary = mask.mapv(|a| if a >= self.ratio { 1.0 } else { 0.0 });
        let result = mask_binary.clone() * x * scale;
        if return_mask{
            return Ok(vec![result.to_owned(), mask_binary.into_dyn()])
        }
        return Ok(vec![result.to_owned()])
    }


    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_names(&self) -> Vec<String> {
        self.output_names.clone()
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