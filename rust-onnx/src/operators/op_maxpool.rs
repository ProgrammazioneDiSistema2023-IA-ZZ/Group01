use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[allow(dead_code)]
pub struct MaxPool {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    kernel_shape: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    ceil_mode: Option<i64>,
    storage_order: Option<i64>,
    dilations: Option<Vec<i64>>,
}

impl MaxPool {
    pub fn new(node: &NodeProto, _initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();

        let mut ceil_mode : Option<i64> = None;
        let mut dilations = None;
        let mut pads= None;
        let mut strides = None;
        let mut kernel_shape = None;
        let mut storage_order = None;

        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "ceil_mode" => ceil_mode = Some(attribute.i),
                "strides" => strides = Some(attribute.ints.clone()),
                "kernel_shape" => kernel_shape = Some (attribute.ints.clone()),
                "dilations" => dilations = {
                    Some(attribute.ints.clone()) },
                "pads" => pads = Some ( attribute.ints.clone()),
                "storage_order" => storage_order = Some(attribute.i),
                // Handle other attributes
                _ => {}
            }
        }

        MaxPool {
            op_type,
            node_name,
            input_name,
            output_name,
            kernel_shape,
            strides,
            pads,
            ceil_mode,
            storage_order,
            dilations,
        }
    }
}

impl Operator for MaxPool {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input_name = &self.input_name;
        let input = inputs.get(input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found".to_string())).unwrap();

        // Validate input tensor dimensions (assuming 4D tensor: [N, C, H, W])
        if input.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch("Input tensor must be 4D".to_string()));
        }

        let kernel_shape = self.kernel_shape.clone().unwrap();
        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; kernel_shape.len()]);
        let pads = self.pads.clone().unwrap_or_else(|| vec![0; kernel_shape.len()].repeat(2));//vec![0; x.ndim() - 2]
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; kernel_shape.len()]);//vec![0; x.ndim() - 2]

        //let mut res = ArrayD::<f32>::zeros(vec![]);

        let n_dims = kernel_shape.len();
        let new_pads: Vec<(i64, i64)> = (0..n_dims)
            .map(|i| (pads[i ], pads[i + n_dims ]))
            .collect();

        let input_spatial_shape = input.shape()[2..].to_vec();
        let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

        if let Some(_) = self.ceil_mode{
            for i in 0..input_spatial_shape.len(){
                output_spatial_shape[i] =  (((input_spatial_shape[i] as f32
                    + (new_pads[i].0 + new_pads[i].1) as f32
                    - (dilations[i] as f32 * (kernel_shape[i] as f32 - 1.0) + 1.0))
                    / strides[i] as f32)
                    + 1.0)
                    .ceil() as i32;
            }

        } else{
            for i in 0..input_spatial_shape.len(){
                output_spatial_shape[i] =  (((input_spatial_shape[i] as f32
                    + (new_pads[i].0 + new_pads[i].1) as f32
                    - (dilations[i] as f32 * (kernel_shape[i] as f32 - 1.0) + 1.0))
                    / strides[i] as f32)
                    + 1.0)
                    .floor() as i32;
            }
        }


        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let in_height = input.shape()[2] as i64;
        let in_width = input.shape()[3] as i64;

        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];
        let stride_y = strides[0];
        let stride_x = strides[1];

        // Calculate output dimensions
        let output_height = ((in_height - kernel_height + pads[0] + pads[2] ) / stride_y + 1).max(0);
        let output_width = ((in_width - kernel_width + pads[1] + pads[3]) / stride_x + 1).max(0);

        let output_dims = IxDyn(&[batch_size, channels, output_height as usize, output_width as usize]);
        let mut output_data = ArrayD::from_elem(output_dims, f32::MIN);

        for n in 0..batch_size {
            for c in 0..channels {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let mut max_val = f32::MIN;

                        for ky in 0..kernel_height {
                            for kx in 0..kernel_width {
                                let in_y = y * stride_y + ky - pads[0];
                                let in_x = x * stride_x + kx - pads[1];

                                if in_y >= 0 && in_x >= 0 && in_y < in_height && in_x < in_width {
                                    let val = input[[n, c, in_y as usize, in_x as usize]];
                                    max_val = max_val.max(val);
                                }
                            }
                        }

                        output_data[[n , c , y as usize, x as usize]] = max_val;
                    }
                }
            }
        }

        Ok(vec![output_data])
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

    fn get_initializers_arr(&self) -> Vec<Initializer> {
        vec![]
    }
}