use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use indexmap::IndexMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Gemm {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
    initializers: IndexMap<String, ArrayD<f32>>,
}

impl Gemm {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let input_name = node.input[0].to_owned();

        let mut hm: IndexMap<String, ArrayD<f32>> = IndexMap::new();
        hm.insert(node.input[1].clone(), initializers.remove(&node.input[1]).unwrap().to_owned());
        let c = initializers.remove(&node.input[2]);
        if let Some(value) = c {
            hm.insert(node.input[2].clone(), value.to_owned());
        }

        let mut alpha= f32::default();
        let mut beta= f32::default();
        let mut trans_a = i64::default();
        let mut trans_b = i64::default();

        for attribute in &node.attribute{
            match attribute.name.as_str(){
                "alpha" => alpha = attribute.f.to_owned(),
                "beta" => beta = attribute.f.to_owned(),
                "transA" => trans_a = attribute.i.to_owned(),
                "transB" => trans_b = attribute.i.to_owned(),
                _ => {}
            }
        }

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
            alpha,
            beta,
            trans_a,
            trans_b,
            initializers: hm
        }
    }

    fn transpose(tensor: &ArrayD<f32>) -> ArrayD<f32> {
        tensor.t().to_owned()
    }
}

impl Operator for Gemm {
    //Y = alpha * A’ * B’ + beta * C
    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let a = inputs.get(&self.input_name).ok_or("Input tensor A not found")?;
        let b = self.initializers.iter().collect::<Vec<_>>()[0].1;
        let mut c : Option<&ArrayD<f32>> = None;
        if self.initializers.iter().collect::<Vec<_>>().len()>1{
            c = Some(self.initializers.iter().collect::<Vec<_>>()[1].1);
        }

        // Transpose A and B if needed
        let a_prime = if self.trans_a != 0 { Gemm::transpose(a) } else { a.clone() };
        let b_prime = if self.trans_b != 0 { Gemm::transpose(b) } else { b.clone() };

        //The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
        let a_prime_2d = a_prime.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "a_prime is not 2-dimensional".to_string())?;
        //Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
        let b_prime_2d = b_prime.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| "b_prime is not 2-dimensional".to_string())?;

        // Check shapes for matrix multiplication
        let (m, k_a) = a_prime_2d.dim();
        let (k_b, n) = b_prime_2d.dim();
        if k_a != k_b {
            return Err("The inner dimensions of A' and B' do not match".to_string());
        }

        // Perform matrix multiplication A' * B'
        let mut y = a_prime_2d.dot(&b_prime_2d);

        // Scale by alpha
        y *= self.alpha;

        // Add beta * C if C is provided and beta != 0
        if let Some(c_tensor) = c  {
            if self.beta != 0.0{
                y += &(c_tensor * self.beta);
            }

        }

        Ok(vec![y.into_dyn()]) // Convert to ArrayD<f32> if needed
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
        self.initializers.iter().map(|v| {
            (v.0.to_owned(), v.1.to_owned())
        }
        ).collect::<Vec<_>>()
    }
}