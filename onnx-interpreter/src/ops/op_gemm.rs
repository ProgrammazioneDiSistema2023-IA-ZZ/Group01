use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct Gemm {
    node_name: String,
    inputs_name: Vec<String>,
    output_name: String,
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
    b: ArrayD<f32>,
    c: Option<ArrayD<f32>>,
}

impl Gemm {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let b = initializers.remove(&node.input[1]).unwrap();
        let c = initializers.remove(&node.input[2]);

        let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];

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
                _ => todo!()
            }
        }

        Self {
            node_name,
            inputs_name,
            output_name,
            alpha,
            beta,
            trans_a,
            trans_b,
            b,
            c
        }
    }

    fn transpose(tensor: &ArrayD<f32>) -> ArrayD<f32> {
        tensor.t().to_owned()
    }
}

impl Operator for Gemm {
    //Y = alpha * A’ * B’ + beta * C
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let a = inputs.get(&self.inputs_name[0]).ok_or("Input tensor A not found")?;
        let b = &self.b;
        let c = &self.c;

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

        Ok(y.into_dyn()) // Convert to ArrayD<f32> if needed
    }

    fn to_string(&self) -> String {
        format!("Node name: {}\nInputs name: {} {}\nOutput names: {}",
                self.node_name, self.inputs_name[0], self.inputs_name[1], self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_name.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}