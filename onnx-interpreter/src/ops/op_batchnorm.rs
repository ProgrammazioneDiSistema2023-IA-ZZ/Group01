use super::op_operator::Operator;
use ndarray::ArrayD;
use std::collections::HashMap;
use indexmap::IndexMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

pub struct BatchNorm {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    epsilon: f32,
    initializers: IndexMap<String, ArrayD<f32>>,
}
impl BatchNorm {
    pub fn new(node: &NodeProto, initializers: &mut IndexMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let input_name = node.input[0].to_owned();

        let opt_epsilon = node.attribute.iter().find(|attr| attr.name == "epsilon");
        let epsilon = match opt_epsilon{
            Some(x) => x.f,
            None => 1e-5
        };

        let mut hm: IndexMap<String, ArrayD<f32>> = IndexMap::new();

        for inp_name in &node.input{
            match inp_name {
                _ if inp_name.contains("gamma") =>{
                    hm.insert(inp_name.to_owned(), initializers.remove(inp_name).unwrap());
                } ,
                _ if inp_name.contains("beta") => {
                    hm.insert(inp_name.to_owned(), initializers.remove(inp_name).unwrap());
                },
                _ if inp_name.contains("mean") =>{
                    hm.insert(inp_name.to_owned(), initializers.remove(inp_name).unwrap());
                },
                _ if inp_name.contains("var") =>{
                    hm.insert(inp_name.to_owned(), initializers.remove(inp_name).unwrap());
                },
                _ => {  }
            }
        }

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
            epsilon,
            initializers: hm
        }
    }
}

impl Operator for BatchNorm {
    fn execute(&mut self, inputs: &IndexMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, String> {
        let x = inputs.get(&self.input_name).ok_or("Input tensor X not found")?;
        let scale = self.initializers.iter()
            .filter(|x|x.0.contains("gamma"))
            .collect::<Vec<_>>()[0].1;
        let b = self.initializers.iter()
            .filter(|x|x.0.contains("beta"))
            .collect::<Vec<_>>()[0].1;
        let mean = self.initializers.iter()
            .filter(|x|x.0.contains("mean"))
            .collect::<Vec<_>>()[0].1;
        let var = self.initializers.iter()
            .filter(|x|x.0.contains("var"))
            .collect::<Vec<_>>()[0].1;

        // Assuming the second dimension is the channel
        let channel_axis = 1;
        let mut y = x.clone(); // Clone the shape and data

        for (((mut y_slice, mean_val), var_val), (scale_val, b_val)) in
            y.axis_iter_mut(ndarray::Axis(channel_axis))
                .zip(mean.iter())
                .zip(var.iter())
                .zip(scale.iter().zip(b.iter())) {
                for y_elem in y_slice.iter_mut() {
                    *y_elem = ((*y_elem - mean_val) / ((var_val + self.epsilon).sqrt())) * scale_val + b_val;
                }
        }

        Ok(vec![y])
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