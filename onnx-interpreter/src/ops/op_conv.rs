use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD, s, IxDyn, ShapeBuilder, Dimension};
use std::collections::HashMap;
use crate::errors::OnnxError;
use crate::parser_code::onnx_ml_proto3::NodeProto;

#[derive(PartialEq, Clone)]
pub enum AutoPad {
    SAME_LOWER,
    SAME_UPPER,
    NOTSET,
    VALID,
}

#[derive(Clone)]
pub struct Conv {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    kernel_shape: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
    auto_pad: Option<AutoPad>,
    pads: Option<Vec<usize>>,
    group: usize,
    dilations: Option<Vec<usize>>,
    initializers: Vec<Initializer>,
}

impl Conv {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let mut input_name = node.input[0].to_owned();
        let mut kernel_name = node.input[1].to_owned();

        let mut initializers_vec = Vec::new();
        initializers_vec.push(Initializer::new(kernel_name.clone(), initializers.remove(kernel_name.as_str()).unwrap().to_owned()));

        let mut bias_name = None;
        if node.input.len() == 3{
            bias_name = Some(node.input[2].to_owned());
            initializers_vec.push(Initializer::new(bias_name.clone().unwrap(), initializers.remove(bias_name.unwrap().as_str()).unwrap().to_owned()));
        }

        let mut kernel_shape = None;
        let mut strides = None;
        let mut group = 1usize; // default value
        let mut dilations = None;
        let mut pads = None;
        let mut auto_pad = None;

        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "kernel_shape" => kernel_shape = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
                "strides" => strides = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
                "group" => group = attribute.i.to_owned() as usize,
                "dilations" => dilations = Some(attribute.ints.iter().map(|x| *x  as usize).collect()),
                "pads" => pads = Some(attribute.ints.iter().map(|x| *x  as usize).collect()),
                "auto_pad" => {
                    auto_pad = Some(match String::from_utf8(attribute.s.clone()) {
                        Ok(value) => match value.as_str() {
                            "SAME_UPPER" => AutoPad::SAME_UPPER,
                            "SAME_LOWER" => AutoPad::SAME_LOWER,
                            "VALID" => AutoPad::VALID,
                            _ => AutoPad::NOTSET,
                        },
                        Err(_) => AutoPad::NOTSET,
                    });
                }
                // Handle other attributes like "auto_pad" as needed
                _ => {}
            }
        }



        Conv {
            op_type,
            node_name,
            input_name,
            output_name,
            kernel_shape,
            strides,
            auto_pad,
            pads,
            group,
            dilations,
            initializers: initializers_vec
        }
    }

    fn execute_conv(x: ArrayD<f32>, w: ArrayD<f32>, b: Option<&ArrayD<f32>>, auto_pad: &Option<AutoPad>, dilations: &Vec<usize>, kernel_shape: &Vec<usize>, pads: &Vec<usize>, strides: &Vec<usize>) -> Result<Vec<ArrayD<f32>>, String> {

        let mut w_copy = w.clone();
        let mut kernel_shape_copy = kernel_shape.clone();
        let mut pads_copy = pads.clone();

        let mut res = ArrayD::<f32>::zeros(vec![]);

        if dilations[0] != 1 || dilations.iter().min() != dilations.iter().max() {
            let nd = dilations.len();
            let mut new_kernel_shape = Vec::new();
            let mut new_shape = w_copy.shape().to_vec();
            new_shape.truncate(new_shape.len() - nd);

            for (i, &d) in dilations.iter().enumerate() {
                let di = w_copy.ndim() - nd + i;
                new_shape.push(w_copy.shape()[di] + (w_copy.shape()[di] - 1) * (d - 1));
                new_kernel_shape.push(kernel_shape_copy[i] + (kernel_shape_copy[i] - 1) * (d - 1));
            }

            let mut new_w = ArrayD::zeros(IxDyn(&new_shape));

            for idx in w_copy.indexed_iter() {
                let mut new_idx = Vec::new();

                // Manually push each dimension into the Vec
                for &dim_size in idx.0.slice() {
                    new_idx.push(dim_size);
                }
                // Extend the new_idx with zeros, the number of zeros is determined by 'nd'
                new_idx.resize(new_idx.len() + nd, 0);

                for (i, &d) in dilations.iter().enumerate() {
                    if d > 1 {
                        new_idx[w_copy.ndim() - nd + i] *= d;
                    }
                }

                new_w[new_idx.as_slice()] = *idx.1;
            }

            w_copy = ArrayD::from_shape_vec(IxDyn(&new_w.shape()), new_w.iter().cloned().collect()).unwrap();
            kernel_shape_copy = new_kernel_shape;

        }

        if auto_pad.is_some(){
            pads_copy = match auto_pad.as_ref().unwrap() {
                AutoPad::SAME_LOWER | AutoPad::SAME_UPPER | AutoPad::VALID => {
                    let mut head = Vec::new();
                    let mut tail = Vec::new();

                    for i in 0..(x.ndim() - 2) {
                        let d = x.shape()[i];
                        let target_size = (d + strides[i] - 1) / strides[i];
                        let pad_needed = (target_size - 1) * strides[i] + kernel_shape_copy[i] - d;

                        let pad_head = match auto_pad.as_ref().unwrap() {
                            AutoPad::SAME_LOWER => (pad_needed + 1) / 2,
                            _ => pad_needed / 2,
                        };

                        let pad_tail = pad_needed - pad_head;
                        head.push(pad_head);
                        tail.push(pad_tail);
                    }

                    [head, tail].concat()
                },
                _ => pads.clone(),
            };
        }

        if x.ndim() == 4 {
            let (sN, sC, sH, sW) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
            let (kh, kw) = (kernel_shape_copy[0], kernel_shape_copy[1]);
            let (sth, stw) = (strides[0], strides[1]);

            let h_out = (((sH as i32 - kh as i32 + pads_copy[0] as i32 + pads_copy[2] as i32) / sth as i32) + 1) as usize;
            let w_out = (((sW as i32 - kw as i32 + pads_copy[1] as i32 + pads_copy[3] as i32) / stw as i32) + 1) as usize;

            let h0 = pads_copy[0] as i32;
            let w0 = pads_copy[1] as i32;
            let oh = -1 * (kh as i32 % 2);
            let ow = -1 * (kw as i32 % 2);
            let (bh, bw) = (-h0, -w0);
            let (eh, ew) = (h_out as i32 * sth as i32, w_out as i32 * stw as i32);

            res = ArrayD::<f32>::zeros(vec![sN, w_copy.shape()[0], h_out, w_out]);

            if b.is_some(){
                let bias_shape = [1, b.unwrap().len(), 1, 1];
                let broadcasted_bias = b.unwrap().view().into_shape(IxDyn(&bias_shape))
                    .or(Err("Bias cannot be broadcast to the result tensor shape"))?;

                // Add the bias to the result tensor
                res += &broadcasted_bias;
            }

            for n in 0..sN {
                for nw in 0..w_copy.shape()[0] {
                    for c in 0..sC {

                        let w_slice = w_copy.slice(s![nw..nw+1, c..c+1, .., ..]);
                        for io in (bh..eh).step_by(sth) {
                            let hr = (io - bh) / sth as i32;
                            if hr as usize >= h_out {
                                continue;
                            }
                            let i = io + kh as i32 %2;
                            let ih1 = (i+oh).max(0) as usize;
                            let ih2 = (i + oh +kh as i32).min(sH as i32) as usize;

                            for jo in (bw..ew).step_by(stw) {
                                let wr = (jo - bw) / stw as i32;
                                if wr as usize >= w_out {
                                    continue;
                                }
                                let j = jo + kw as i32 %2;
                                let iw1 = (j+ow).max(0) as usize;
                                let iw2 = (j + kw as i32 +ow).min(sW as i32) as usize;

                                let img_slice = x.slice(s![n..n+1, c..c+1, ih1..ih2, iw1..iw2]);
                                let mut value;
                                // Perform the convolution operation and sum to res
                                if img_slice.shape() != w_slice.shape() {
                                    let (jh1, jh2) = (
                                        std::cmp::max(-oh - i, 0) as usize,
                                        std::cmp::min(kh as i32, kh as i32 + sH as i32 - (i + oh + kh as i32)) as usize
                                    );
                                    let (jw1, jw2) = (
                                        std::cmp::max(-ow - j, 0) as usize,
                                        std::cmp::min(kw as i32, kw as i32 + sW as i32 - (j + ow + kw as i32)) as usize
                                    );

                                    let w_adjusted = w_slice.slice(s![..1, ..1, jh1..jh2, jw1..jw2]);

                                    if img_slice.shape() != w_adjusted.shape() {
                                        return Err(format!(
                                            "Unexpected shape {:?} != {:?}, oh={}, ow={}, i={}, j={}, kh={}, kw={}, sH={}, sW={}, sth={}, stw={}.",
                                            img_slice.shape(), w_adjusted.shape(), oh, ow, i, j, kh, kw, sH, sW, sth, stw
                                        ));
                                    }

                                    value = Conv::dot_product(&img_slice.iter().cloned().collect::<Vec<f32>>(),
                                                        &w_adjusted.iter().cloned().collect::<Vec<f32>>()).unwrap();
                                } else {
                                    value = Conv::dot_product(&img_slice.iter().cloned().collect::<Vec<f32>>(),
                                                        &w_slice.iter().cloned().collect::<Vec<f32>>()).unwrap();
                                }

                                // Update res tensor
                                // res is the result tensor defined in the function's scope

                                *res.get_mut([n, nw, hr as usize, wr as usize]).unwrap() += value;
                            }
                        }
                    }
                }
            }
        }
        Ok(vec![res])
    }

    fn dot_product(a: &Vec<f32>, b: &Vec<f32>) -> Result<f32, OnnxError> {
        if a.len() != b.len() {
            return Err(OnnxError::ShapeMismatch("Vectors must be of the same length".to_string()));
        }
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }

}

impl Operator for Conv{
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {

        // 1. Retrieve input tensors X and W, and optionally B.
        // 2. Apply padding according to `auto_pad` or `pads`.
        // 3. Handle dilations and groups.
        // 4. Perform the convolution operation.
        // 5. Return the output tensor Y.

        let input_name = &self.input_name;
        let x = inputs.get(input_name).ok_or(OnnxError::TensorNotFound("First input tensor not found".to_string())).unwrap();
        let mut w = self.initializers[0].get_value();
        let mut b_init=None ;
        if self.initializers.len()>1{
            b_init = Some(self.initializers[1].get_value());
        }

        if x.ndim()<3{
            return Err(OnnxError::ShapeMismatch(format!("The input must have at least 3 dimensions but its shape is {:?}", x.shape())));
        }

        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; x.shape()[2]*x.shape()[3]]);//vec![0; x.ndim() - 2]
        let mut kernel_shape = self.kernel_shape.clone().unwrap_or_else(
            || w.shape()[2..].to_vec());
        let mut pads = self.pads.clone().unwrap_or_else(|| vec![0; x.shape()[2]*x.shape()[3]].repeat(2));//vec![0; x.ndim() - 2]
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; x.shape()[2]*x.shape()[3]]);//vec![0; x.ndim() - 2]

        // Initial shape checks
        if x.shape()[1] != w.shape()[1] * self.group || w.shape()[0] % self.group != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Shape inconsistencies, X.shape={:?}, W.shape={:?}, group={}, \
                W should be {:?}.",
                x.shape(),
                w.shape(),
                self.group,
                (w.shape()[0], x.shape()[1] / self.group, w.shape()[1..].iter().product::<usize>() / x.shape()[1] * self.group)
            )));
        }

        if self.group>1{
            let mut res = vec![];
            let mut td = 0;
            let mg = w.shape()[0]/self.group;
            let dw = w.shape()[1];

            //Iterate over the batch
            for b in 0..x.shape()[0]{
                for g in 0..self.group{
                    let gx_view = x.slice(s![b..b+1, g*dw..(g+1)*dw, .., ..]);
                    let gw_view = w.slice(s![g*mg..(g+1)*mg, .., .., ..]);
                    // Check if the sliced shapes are correct
                    if gx_view.shape()[1] != dw || gw_view.shape()[0] != mg {
                        return Err(OnnxError::ShapeMismatch(
                            format!("Incorrect shape after slicing for group {}. gx_view.shape={:?}, gw_view.shape={:?}", g, gx_view.shape(), gw_view.shape())));
                    }
                    let gx: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&gx_view.shape()), gx_view.iter().cloned().collect()).unwrap();
                    let gw: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&gw_view.shape()), gw_view.iter().cloned().collect()).unwrap();
                    //x: , w: , bias: , auto_pad: , dilations: , kernel_shape: , pads: , strides:
                    let cv = Conv::execute_conv(gx, gw, None, &self.auto_pad, &dilations, &kernel_shape, &pads, &strides).unwrap();
                    if b==0 {
                        td += cv[0].shape()[1];
                    }
                    res.push((b, cv))
                }
            }
            let mut new_shape = vec![x.shape()[0]];
            new_shape.extend_from_slice(&res[0].1[0].shape()[1..]);
            new_shape[1] = td;
            let mut result : ArrayD<f32> = ArrayD::zeros(IxDyn(&new_shape));
            let mut p= 0;
            for (b, cv) in res.iter(){
                let mut slice = result.slice_mut(s![*b..*b+1, p..p+cv[0].shape()[1], .., ..]);
                slice.assign(&cv[0].view());
                p+=cv[0].shape()[1];
                if p >= result.shape()[1]{
                    p=0;
                }
            }
            if b_init.is_some(){
                let mut new_shape = vec![1; result.ndim()];
                new_shape[1] = b_init.unwrap().shape()[0];
                if let b_value = b_init.unwrap(){
                    let tmp = b_value.clone().into_shape(IxDyn(&new_shape)).unwrap();
                    result=result+tmp;
                }
            }
            let expected_output_channels = w.shape()[0];
            let final_shape = result.shape();
            if final_shape[1] != expected_output_channels {
                return Err(OnnxError::ShapeMismatch(format!(
                    "The number of output channels in the final result does not match the expected number. Expected {}, got {}. Final shape: {:?}",
                    expected_output_channels, final_shape[1], final_shape
                )));
            }
            return Ok(vec![result]);
        }

        Ok(Conv::execute_conv(x.clone(), w.clone(), b_init, &self.auto_pad, &dilations, &kernel_shape, &pads, &strides).unwrap())
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
        self.initializers.clone()
    }
}

