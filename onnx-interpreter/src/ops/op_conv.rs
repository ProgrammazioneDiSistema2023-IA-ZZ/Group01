use super::op_operator::Operator;
use ndarray::{ArrayD, s, IxDyn};
use std::collections::HashMap;
use crate::parser_code::onnx_ml_proto3::NodeProto;

#[derive(PartialEq)]
pub enum AutoPad {
    SAME_LOWER,
    SAME_UPPER,
    NOTSET,
    VALID,
}

pub struct Conv {
    node_name: String,
    input_name: String,
    output_name: String,
    kernel_shape: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
    auto_pad: Option<AutoPad>,
    pads: Option<Vec<usize>>,
    group: usize,
    dilations: Option<Vec<usize>>,
    kernel_weights: ArrayD<f32>,
    bias: Option<ArrayD<f32>>,
}

impl Conv {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let mut input_name = node.input[0].to_owned();
        let mut kernel_name = node.input[1].to_owned();

        let kernel_weights = initializers.remove(kernel_name.as_str()).unwrap();

        let mut bias_name = None;
        if node.input.len() == 3{
            bias_name = Some(node.input[2].to_owned());
        }

        let mut bias = None;
        if let Some (b_name) = bias_name{
            bias = initializers.remove(b_name.as_str());
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
            node_name,
            input_name,
            output_name,
            kernel_shape,
            strides,
            auto_pad,
            pads,
            group,
            dilations,
            kernel_weights,
            bias,
        }
    }
}
fn dot_product(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}
impl Operator for Conv{
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {

        // 1. Retrieve input tensors X and W, and optionally B.
        // 2. Apply padding according to `auto_pad` or `pads`.
        // 3. Handle dilations and groups.
        // 4. Perform the convolution operation.
        // 5. Return the output tensor Y.

        let input_name = &self.input_name;
        let x = inputs.get(input_name).ok_or("Input not found")?;
        let mut w = &self.kernel_weights;
        let b = self.bias.as_ref();


        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; x.shape()[2]*x.shape()[3]]);//vec![0; x.ndim() - 2]
        let mut kernel_shape = self.kernel_shape.clone().unwrap_or_else(
            || w.shape()[2..].to_vec());
        let mut pads = self.pads.clone().unwrap_or_else(|| vec![0; x.shape()[2]*x.shape()[3]].repeat(2));//vec![0; x.ndim() - 2]
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; x.shape()[2]*x.shape()[3]]);//vec![0; x.ndim() - 2]

        let mut res = ArrayD::<f32>::zeros(vec![]);

        // Initial shape checks
        if x.shape()[1] != w.shape()[1] * self.group || w.shape()[0] % self.group != 0 {
            return Err(format!(
                "Shape inconsistencies, X.shape={:?}, W.shape={:?}, group={}, \
                W should be {:?}.",
                x.shape(),
                w.shape(),
                self.group,
                (w.shape()[0], x.shape()[1] / self.group, w.shape()[1..].iter().product::<usize>() / x.shape()[1] * self.group)
            ));
        }

        //todo righe 18-65
        /*
            if self.group>1{
                let td = 0;
                let mg = w.shape()[0]/self.group;
                let dw = w.shape()[1];

                //Iterate over the batch
                for b in 0..x.shape()[0]{
                    for g in 0..self.group{
                        let gx = x.slice(s![b..b+1, g*dw..(g+1)*dw, .., ..]);
                        let gw = w.slice(s![g*mg..(g+1)*mg, .., ..]);
                        //x: , w: , bias: , auto_pad: , dilations: , kernel_shape: , pads: , strides:
                        let cv = Conv::execute_conv(gx, gw, None, self.auto_pad, dilations, kernel_shape, pads, strides);
                    }
                }

            }

         */

        //righe 67-83. Check if we to dilatate the image, todo change the weights in case we apply dilation
        if dilations[0] != 1 || dilations.iter().min() != dilations.iter().max() {
            // Compute the dilated kernel
            let nd = dilations.len();
            let mut new_kernel_shape = Vec::new();

            let dilation_h = dilations[0];
            let dilation_w = dilations[1];
            let original_shape = w.shape();
            let new_shape = [
                original_shape[0] + (original_shape[0] - 1) * (dilation_h - 1),
                original_shape[1] + (original_shape[1] - 1) * (dilation_w - 1),
            ];

            let mut new_w = ArrayD::zeros(IxDyn(&new_shape));

            for i in 0..original_shape[0] {
                for j in 0..original_shape[1] {
                    if let Some(val) = w.get([i, j]) {
                        let new_i = i * dilation_h;
                        let new_j = j * dilation_w;
                        if let Some(target) = new_w.get_mut([new_i, new_j]) {
                            *target = *val;
                        }
                    }
                }
            }

            // Update w and kernel_shape
            w = &new_w;
            kernel_shape = new_kernel_shape;
        }



        //righe 85-99
        if self.auto_pad.is_some(){
            pads = match self.auto_pad.as_ref().unwrap() {
                AutoPad::SAME_LOWER | AutoPad::SAME_UPPER | AutoPad::VALID => {
                    let mut head = Vec::new();
                    let mut tail = Vec::new();

                    for i in 0..(x.ndim() - 2) {
                        let d = x.shape()[i];
                        let target_size = (d + strides[i] - 1) / strides[i];
                        let pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d;

                        let pad_head = match self.auto_pad.as_ref().unwrap() {
                            AutoPad::SAME_LOWER => (pad_needed + 1) / 2,
                            _ => pad_needed / 2,
                        };

                        let pad_tail = pad_needed - pad_head;
                        head.push(pad_head);
                        tail.push(pad_tail);
                    }

                    [head, tail].concat()
                },
                _ => self.pads.clone().unwrap_or_else(Vec::new),
            };
        }


        //todo righe 101-145, input shape [x,y,z]

        //righe 147-203
        if x.ndim() == 4 {
            let (sN, sC, sH, sW) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
            let (kh, kw) = (kernel_shape[0], kernel_shape[1]);
            let (sth, stw) = (strides[0], strides[1]);
            let pads = pads;

            let h_out = ((sH as i32 - kh as i32 + pads[0] as i32 + pads[2] as i32) / sth as i32 + 1) as usize;
            let w_out = ((sW as i32 - kw as i32 + pads[1] as i32 + pads[3] as i32) / stw as i32 + 1) as usize;

            let h0 = pads[0] as i32;
            let w0 = pads[1] as i32;
            let oh = -1 * (kh as i32 % 2) ;
            let ow = -1 * (kw as i32 % 2) ;
            let (bh, bw) = (-h0, -w0);
            let (eh, ew) = (h_out as i32 * sth as i32, w_out as i32 * stw as i32);

            res = ArrayD::<f32>::zeros(vec![sN, self.kernel_weights.shape()[0], h_out, w_out]);
            println!("Res shape: {:?}", res.shape());

            if let Some(b) = &self.bias {
                let bias_shape = [1, b.shape()[0], 1, 1];
                let broadcasted_bias = b.view().into_shape(IxDyn(&bias_shape))
                    .or(Err("Bias cannot be broadcast to the result tensor shape"))?;

                // Add the bias to the result tensor
                res += &broadcasted_bias;
            }

            for n in 0..sN {
                for nw in 0..self.kernel_weights.shape()[0] {
                    for c in 0..sC {

                        let w_slice = self.kernel_weights.slice(s![nw..nw+1, c..c+1, .., ..]);
                        for io in (bh..eh).step_by(sth as usize) {
                            let hr = (io - bh) / sth as i32;
                            if hr as usize >= h_out {
                                continue;
                            }
                            let i = io + kh as i32 %2;
                            let ih1 = (i+oh).max(0) as usize;
                            let ih2 = (i + oh +kh as i32).min(sH as i32) as usize;

                            for jo in (bw..ew).step_by(stw as usize) {
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

                                    value = dot_product(&img_slice.iter().cloned().collect::<Vec<f32>>(),
                                                        &w_adjusted.iter().cloned().collect::<Vec<f32>>());
                                } else {
                                    value = dot_product(&img_slice.iter().cloned().collect::<Vec<f32>>(),
                                                        &w_slice.iter().cloned().collect::<Vec<f32>>());
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

        Ok(res)
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