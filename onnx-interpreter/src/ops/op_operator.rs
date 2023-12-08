use ndarray::ArrayD;
use std::collections::HashMap;

pub trait Operator {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String>;
    fn to_string(&self) -> String;
    fn get_inputs(&self) -> Vec<String>;
    fn get_output_name(&self) -> String;
}