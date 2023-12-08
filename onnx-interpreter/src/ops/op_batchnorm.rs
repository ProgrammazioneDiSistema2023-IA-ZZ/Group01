use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct BatchNorm {
    node_name: String,
    input_name: String,
    output_name: String,
    epsilon: f32,
    momentum: f32,
    spatial: i64,
    gamma: ArrayD<f32>,
    beta: ArrayD<f32>,
    mean: ArrayD<f32>,
    var: ArrayD<f32>,
}

impl BatchNorm {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let input_name = node.input[0].to_owned();

        let mut epsilon = f32::default();
        let mut momentum = f32::default();
        let mut spatial = i64::default();

        for attribute in &node.attribute{
            match attribute.name.as_str(){
                "epsilon" => epsilon = attribute.f.to_owned(),
                "momentum" => momentum = attribute.f.to_owned(),
                "spatial" => spatial = attribute.i.to_owned(),
                _ => todo!()
            }
        }

        let mut gamma = None;
        let mut beta = None;
        let mut mean = None;
        let mut var = None;

        for inp_name in &node.input{
            match inp_name {
                _ if inp_name.contains("gamma") => gamma = initializers.remove(inp_name),
                _ if inp_name.contains("beta") => beta = initializers.remove(inp_name),
                _ if inp_name.contains("mean") => mean = initializers.remove(inp_name),
                _ if inp_name.contains("var") => var = initializers.remove(inp_name),
                _ => todo!()
            }
        }

        Self {
            node_name,
            input_name,
            output_name,
            epsilon,
            momentum,
            spatial,
            gamma: gamma.unwrap(),
            beta: beta.unwrap(),
            mean: mean.unwrap(),
            var: var.unwrap(),
        }
    }
}

impl Operator for BatchNorm {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let x = inputs.get(&self.input_name).ok_or("Input tensor X not found")?;
        let scale = &self.gamma;
        let b = &self.beta;

        // Assuming the second dimension is the channel
        let channel_axis = 1;
        let mut y = x.clone(); // Clone the shape and data

        for (((mut y_slice, mean_val), var_val), (scale_val, b_val)) in
        y.axis_iter_mut(ndarray::Axis(channel_axis))
            .zip(self.mean.iter())
            .zip(self.var.iter())
            .zip(scale.iter().zip(b.iter())) {

            for y_elem in y_slice.iter_mut() {
                *y_elem = ((*y_elem - mean_val) / ((var_val + self.epsilon).sqrt())) * scale_val + b_val;
            }
        }

        Ok(y)
    }

    fn to_string(&self) -> String {
        format!("Node name: {}\nInput name: {}\nOutput name: {}",
                self.node_name, self.input_name, self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}