extern crate ndarray;
use ndarray::{ArrayD, IxDyn, Axis, Zip, Dimension, IntoDimension, Ix2,
              Array, ArrayViewD, s};
use std::collections::HashMap;

#[derive(PartialEq)]
pub enum AutoPad {
    SAME_LOWER,
    SAME_UPPER,
    NOTSET,
    VALID,
}

pub trait Operator {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String>;
    fn to_string(&self) -> String;
    fn get_inputs(&self) -> Vec<String>;
    fn get_output_name(&self) -> String;
}

pub struct Add {
    name: String,
    inputs: Vec<String>,
    output_name: String,
    parameter: Option<ArrayD<f32>>,
}

impl Add {
    pub fn new(name: String, inputs: Vec<String>, output_name: String, parameter: Option<ArrayD<f32>>) -> Self {
        Self {
            name,
            inputs,
            output_name,
            parameter,
        }
    }
}

impl Operator for Add {

    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let a = inputs.get(&self.inputs[0])
            .ok_or("First input tensor not found")?;
        let b = match inputs.get(&self.inputs[1]) {
            Some(tensor) => tensor,
            None => self.parameter.as_ref().expect("Parameter tensor not found"),
        };

        let result = a + b;
        Ok(result)
    }


    fn to_string(&self) -> String {
        format!("Name: {}\nInput: {} {}\nOutput names: {} \n", self.name, self.inputs[0], self.inputs[1], self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct BatchNorm {
    name: String,
    inputs: String,
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
    pub fn new(
        name: String,
        inputs: String,
        output_name: String,
        epsilon: f32,
        momentum: f32,
        spatial: i64,
        gamma: ArrayD<f32>,
        beta: ArrayD<f32>,
        mean: ArrayD<f32>,
        var: ArrayD<f32>
    ) -> Self {
        Self {
            name,
            inputs,
            output_name,
            epsilon,
            momentum,
            spatial,
            gamma,
            beta,
            mean,
            var,
        }
    }
}

impl Operator for BatchNorm {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let x = inputs.get(&self.inputs).ok_or("Input tensor X not found")?;
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
        format!("Name: {}\nInput: {}\nOutput names: {}\n", self.name, self.inputs, self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.inputs.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct Conv {
    name: String,
    input: String,
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
    pub fn new(
        name: String,
        input: String,
        output_name: String,
        kernel_shape: Option<Vec<usize>>,
        strides: Option<Vec<usize>>,
        auto_pad: Option<AutoPad>,
        pads: Option<Vec<usize>>,
        group: usize,
        dilations: Option<Vec<usize>>,
        kernel_weights: ArrayD<f32>,
        bias: Option<ArrayD<f32>>,
    ) -> Self {
        Conv {
            name,
            input,
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

        let input_name = &self.input;
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
        format!("Name: {}\nInput: {}\nOutput names: {}\n", self.name, self.input, self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct Flatten {
    name: String,
    input_name: String,
    output_name: String,
    axis: i32,
}

impl Flatten {
    pub fn new(name: String, input_name: String, output_name: String, axis: i32) -> Self {
        Self { name, input_name, output_name, axis }
    }
}

impl Operator for Flatten {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_tensor = inputs.get(&self.input_name)
            .ok_or("Input tensor not found")?;

        let rank = input_tensor.ndim();

        // Normalize axis value
        let axis = if self.axis < 0 {
            rank as i32 + self.axis
        } else {
            self.axis
        } as usize;

        if axis > rank {
            return Err("Axis is out of bounds for the tensor shape".to_string());
        }

        // Calculate the new shape
        let first_dim: usize = if axis == 0 { 1 } else { input_tensor.shape()[..axis].iter().product() };
        let second_dim: usize = input_tensor.shape()[axis..].iter().product();
        let new_shape = IxDyn(&[first_dim, second_dim]);

        // Create the output tensor with the same data but new shape
        let output_tensor = input_tensor.clone().into_shape(new_shape)
            .map_err(|_| "Error reshaping tensor".to_string())?;

        Ok(output_tensor)
    }

    fn to_string(&self) -> String {
        format!("Name: {}\nInput: {}\nOutput name: {}", self.name, self.input_name, self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct Gemm {
    name: String,
    inputs: Vec<String>,
    output_name: String,
    alpha: f32,
    beta: f32,
    transA: i64,
    transB: i64,
    B: ArrayD<f32>,
    C: Option<ArrayD<f32>>,
}

impl Gemm {
    pub fn new(name: String, inputs: Vec<String>, output_name: String,
               alpha: f32, beta: f32, transA: i64, transB: i64,
               B: ArrayD<f32>, C: Option<ArrayD<f32>>) -> Self {
        Self {
            name,
            inputs,
            output_name,
            alpha,
            beta,
            transA,
            transB,
            B,
            C
        }
    }

    fn transpose(tensor: &ArrayD<f32>) -> ArrayD<f32> {
        tensor.t().to_owned()
    }
}

impl Operator for Gemm {
    //Y = alpha * A’ * B’ + beta * C
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let a = inputs.get(&self.inputs[0]).ok_or("Input tensor A not found")?;
        let b = &self.B;
        let c = &self.C;

        // Transpose A and B if needed
        let a_prime = if self.transA != 0 { Gemm::transpose(a) } else { a.clone() };
        let b_prime = if self.transB != 0 { Gemm::transpose(b) } else { b.clone() };

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
        format!("Name: {}\nInput: {} {}\nOutput names: {} \n", self.name, self.inputs[0], self.inputs[1], self.output_name)
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct GlobalAveragePool {
    name: String,
    input_name: String,
    output_name: String,
}

impl GlobalAveragePool {
    pub fn new(name: String, input_name: String, output_name: String) -> Self {
        Self {
            name,
            input_name,
            output_name,
        }
    }
}

impl Operator for GlobalAveragePool {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        // Retrieve the input tensor
        let input_tensor = inputs.get(&self.input_name).ok_or("Input tensor not found")?;

        if input_tensor.ndim() < 3 {
            return Err("Input tensor must have at least 3 dimensions".to_string());
        }
        let axis_to_pool: Vec<_> = (2..input_tensor.ndim()).collect();

        let mut y = input_tensor.clone();
        for &axis in axis_to_pool.iter().rev() {
            y = y.mean_axis(Axis(axis)).unwrap();
        }

        for _ in axis_to_pool {
            let dim = y.ndim();
            y = y.insert_axis(Axis(dim));
        }

        Ok(y)
    }

    fn to_string(&self) -> String {
        format!(
            "Name: {}\nInput: {}\nOutput name: {}",
            self.name, self.input_name, self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct MatMul {
    name: String,
    inputs_name: Vec<String>,
    output_name: String,
}

impl MatMul {
    pub fn new(name: String, inputs_name: Vec<String>, output_name: String) -> Self {
        Self {
            name,
            inputs_name,
            output_name,
        }
    }
}

impl Operator for MatMul {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name1 = self.inputs_name[0].clone();
        let input_name2 = self.inputs_name[1].clone();

        let input1 = inputs.get(input_name1.as_str()).unwrap();
        let input2 = inputs.get(input_name2.as_str()).unwrap();

        if input1.shape()[1] != input2.shape()[0] {
            return Err("Shapes cannot be multiplied".to_string());
        }

        let m = input1.shape()[0];
        let n = input2.shape()[1];
        let common_dim = input1.shape()[1];

        // Perform matrix multiplication
        let pooled_dims = IxDyn(&[m, n]);
        let mut pooled_data = ArrayD::zeros(pooled_dims);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..common_dim {
                    sum += input1[[i, k]] * input2[[k, j]];
                }
                pooled_data[[i, j]] = sum;
            }
        }

        Ok(pooled_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Name: {}\nInput: {} {}\nOutput names: {} \n",
            self.name,
            self.inputs_name[0],
            self.inputs_name[1],
            self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_name.clone()
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct MaxPool {
    name: String,
    input: String,
    output_name: String,
    kernel_shape: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    autopad: Option<AutoPad>,
    ceil_mode: Option<i64>,
    dilations: Option<Vec<i64>>,
}

impl MaxPool {
    pub fn new(
        name: String,
        input: String,
        output_name: String,
        kernel_shape: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        autopad: Option<AutoPad>,
        ceil_mode: Option<i64>,
        dilations: Option<Vec<i64>>,
    ) -> Self {
        MaxPool {
            name,
            input,
            output_name,
            kernel_shape,
            strides,
            pads,
            autopad,
            ceil_mode,
            dilations,
        }
    }
}

impl Operator for MaxPool {
    //todo considering also dilations
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name = self.input.clone();
        let input = inputs.get(input_name.as_str()).unwrap();

        // Validate input tensor dimensions (assuming 4D tensor: [N, C, H, W])
        if input.ndim() != 4 {
            return Err("Input tensor must be 4D".to_string());
        }

        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; input.shape()[2]*input.shape()[3]]);
        let mut kernel_shape = self.kernel_shape.clone().unwrap();
        let mut pads = self.pads.clone().unwrap_or_else(|| vec![0; input.shape()[2]*input.shape()[3]].repeat(2));//vec![0; x.ndim() - 2]
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; input.shape()[2]*input.shape()[3]]);//vec![0; x.ndim() - 2]

        let mut res = ArrayD::<f32>::zeros(vec![]);

        let n_dims = kernel_shape.len();
        let new_pads: Vec<(i64, i64)> = (0..n_dims)
            .map(|i| (pads[i as usize], pads[(i + n_dims) as usize]))
            .collect();

        let mut input_spatial_shape = input.shape()[2..].to_vec();
        let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

        if let Some(ceil_mode) = self.ceil_mode{
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

        //auto_pad should be a deprecated attribute, so i didn't put code for it


        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let in_height = input.shape()[2] as i64;
        let in_width = input.shape()[3] as i64;

        let kernel_height = kernel_shape[0] as i64;
        let kernel_width = kernel_shape[1] as i64;
        let stride_y = strides[0] as i64;
        let stride_x = strides[1] as i64;

        // Calculate output dimensions
        let output_height = ((in_height - kernel_height + 2 * pads[0] ) / stride_y + 1).max(0);
        let output_width = ((in_width - kernel_width + 2 * pads[1]) / stride_x + 1).max(0);

        let output_dims = IxDyn(&[batch_size, channels, output_height as usize, output_width as usize]);
        let mut output_data = ArrayD::from_elem(output_dims, f32::MIN);

        for n in 0..batch_size {
            for c in 0..channels {
                for y in 0..output_height as i64 {
                    for x in 0..output_width as i64 {
                        let mut max_val = f32::MIN;

                        for ky in 0..kernel_height {
                            for kx in 0..kernel_width {
                                let in_y = y * stride_y + ky - pads[0];
                                let in_x = x * stride_x + kx - pads[1];

                                if in_y >= 0 && in_x >= 0 && in_y < in_height && in_x < in_width {
                                    let val = input[[n as usize, c as usize, in_y as usize, in_x as usize]];
                                    max_val = max_val.max(val);
                                }
                            }
                        }

                        output_data[[n as usize, c as usize, y as usize, x as usize]] = max_val;
                    }
                }
            }
        }

        Ok(output_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Name: {}\nInput: {}\nOutput names: {}",
            self.name,
            self.input,
            self.output_name,

        )
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct ReLU {
    name: String,
    input_name: String,
    output_name: String,
}

impl ReLU {
    pub fn new(name: String, input_name: String, output_name: String) -> Self {
        Self {
            name,
            input_name,
            output_name,
        }
    }
}

impl Operator for ReLU {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name = self.input_name.clone();
        let input = inputs.get(input_name.as_str()).unwrap();

        let output_data = input.mapv(|x| if x > 0.0 { x } else { 0.0 });

        Ok(output_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Name: {}\nInput: {}\nOutput name: {}",
            self.name, self.input_name, self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}

pub struct Reshape {
    name: String,
    input: String,
    output_name: String,
    shape: Vec<usize>,
}

impl Reshape {
    pub fn new(name: String, input: String, output_name: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            input,
            output_name,
            shape,
        }
    }
}

impl Operator for Reshape {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<ArrayD<f32>, String> {
        let input_name = self.input.clone();
        let input = inputs.get(input_name.as_str()).unwrap();

        let shape_size: usize = self.shape.iter().product();
        let input_size = input.len();

        if shape_size != input_size {
            return Err("Dimension is not correct for the number of data.".to_string());
        }

        let output_data = input.clone().into_shape(self.shape.clone()).unwrap();

        Ok(output_data)
    }

    fn to_string(&self) -> String {
        format!(
            "Name: {}\nInput: {}\nOutput name: {}",
            self.name, self.input, self.output_name
        )
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input.clone()]
    }

    fn get_output_name(&self) -> String {
        self.output_name.clone()
    }
}



