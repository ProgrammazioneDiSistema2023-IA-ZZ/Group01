use ndarray::{Array, Array2, Array3, Array4, Ix, IxDyn, ArrayView2, ArrayD, Axis};
use std::fs;
use std::iter::FromIterator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryInto;
use std::fs::File;
use byteorder::{ByteOrder, LittleEndian};
use indexmap::IndexMap;
use prettytable::{format, row, Row, Table, Cell, Attr};
extern crate protobuf;
use protobuf::{Message};
use crate::parser_code::onnx_ml_proto3::{ModelProto, TensorProto};
use crate::ops::*;
use crate::errors::OnnxError;
use image::io::Reader as ImageReader;
use image::{imageops};
use std::io::Write;


const MIN_SIZE: u32 = 256;
const CROP_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE_FACTOR: f32 = 255.0;

pub fn serialize_image_to_pb(image_path: &str, pb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    //let img_f32 = img.clone().into_bytes().iter().map(|&b| b as f32).collect();
    let (width, height) = (img.width(), img.height());

    let (scaled_width, scaled_height) = if width > height {
        (MIN_SIZE * width / height, MIN_SIZE)
    } else {
        (MIN_SIZE, MIN_SIZE * height / width)
    };

    img = img.resize(scaled_width, scaled_height, imageops::FilterType::Gaussian);

    let crop_x = (scaled_width - CROP_SIZE) / 2;
    let crop_y = (scaled_height - CROP_SIZE) / 2;

    img = img.crop_imm(crop_x, crop_y, CROP_SIZE, CROP_SIZE);

    let img_rgb = img.to_rgb8();
    let raw_data = img_rgb.into_raw();

    let mut r_color = Vec::new();
    let mut g_color = Vec::new();
    let mut b_color = Vec::new();

    for i in 0..raw_data.len() / 3 {
        r_color.push(raw_data[3 * i]);
        g_color.push(raw_data[3 * i + 1]);
        b_color.push(raw_data[3 * i + 2]);
    }

    let r_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), r_color).unwrap();
    let g_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), g_color).unwrap();
    let b_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), b_color).unwrap();

    let mut rgb_arr: Array3<u8> =
        ndarray::stack(Axis(2), &[r_array.view(), g_array.view(), b_array.view()]).unwrap();
    // Transpose from HWC to CHW
    rgb_arr.swap_axes(0, 2);

    let mean = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            MEAN[0] * SCALE_FACTOR,
            MEAN[1] * SCALE_FACTOR,
            MEAN[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let std = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            STD[0] * SCALE_FACTOR,
            STD[1] * SCALE_FACTOR,
            STD[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let mut arr_float: Array3<f32> = rgb_arr.mapv(|x| x as f32);

    arr_float -= &mean;
    arr_float /= &std;

    let arr_float_batch: Array4<f32> = arr_float.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    let flat: Vec<f32> = arr_d.iter().cloned().collect(); // Step 1: Flatten the array
    let mut img_bytes: Vec<u8> = Vec::with_capacity(flat.len() * 4); // Step 2: Allocate Vec<u8>

    for &value in &flat {
        let byte_repr: [u8; 4] = value.to_le_bytes(); // Convert each f32 to 4 bytes
        img_bytes.extend_from_slice(&byte_repr); // Append bytes to Vec<u8>
    }

    let image_proto = TensorProto {
        dims: vec![1i64, 3i64, CROP_SIZE as i64, CROP_SIZE as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "data".to_string(),
        raw_data: img_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    image_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

pub fn serialize_g_image_to_pb(image_path: &str, pb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    //let img_f32 = img.clone().into_bytes().iter().map(|&b| b as f32).collect();
    let (width, height) = (img.width(), img.height());

    img = img.resize(28, 28, imageops::FilterType::Gaussian);

    let img_gray = img.to_luma8();
    let raw_data = img_gray.into_raw();

    let gray_array: Array2<u8> =
        Array::from_shape_vec((28 as usize, 28 as usize), raw_data).unwrap();


    let mut arr_float: Array2<f32> = gray_array.mapv(|x| x as f32);

    let arr_f_im: Array3<f32> = arr_float.insert_axis(Axis(0));
    let arr_float_batch: Array4<f32> = arr_f_im.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    let flat: Vec<f32> = arr_d.iter().cloned().collect(); // Step 1: Flatten the array
    let mut img_bytes: Vec<u8> = Vec::with_capacity(flat.len() * 4); // Step 2: Allocate Vec<u8>

    for &value in &flat {
        let byte_repr: [u8; 4] = value.to_le_bytes(); // Convert each f32 to 4 bytes
        img_bytes.extend_from_slice(&byte_repr); // Append bytes to Vec<u8>
    }

    let image_proto = TensorProto {
        dims: vec![1i64, 1i64, 28 as i64, 28 as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "Input3".to_string(),
        raw_data: img_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    image_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

pub fn load_model(file_path: &String) -> ModelProto {
    // Load and deserialize your .onnx file here
    let model_bytes = std::fs::read(file_path).expect("Failed to read .onnx file");
    let mut model = ModelProto::new(); // Create an instance of ModelProto

    // Use the parse_from_bytes method to deserialize the model
    model
        .merge_from_bytes(&model_bytes)
        .expect("Failed to parse .onnx file");

    model
}

pub fn raw_data_to_array(data: &TensorProto, dims: Vec<usize>)->Result<ArrayD<f32>, OnnxError>{
    let data_array = data.raw_data
        .chunks_exact(4)
        .map(|chunk| {
            chunk.try_into()
                .map_err(|_| OnnxError::ConversionError("Slice with incorrect length".to_string()))
                .map(|bytes| f32::from_le_bytes(bytes))
        })
        .collect();

    match data_array {
        Ok(data) => {
            ArrayD::from_shape_vec(dims, data)
                .map_err(|_| OnnxError::ShapeMismatch("Failed to create array with given shape.".to_string()))
        },
        Err(e) => Err(e),
    }
}

pub fn load_data(file_path: &String) -> Result<(ArrayD<f32>, String), OnnxError> {
    // Read the file contents into a buffer
    let buffer = fs::read(file_path).map_err(|e| format!("Failed to open the file: {}", e)).unwrap();
    let mut data = TensorProto::new();

    // Decode the buffer using the generated Rust structs
    data
        .merge_from_bytes(&buffer)
        .expect( "Failed to parse .pb file");

    let dims = data.dims.iter().map(|&d| d as Ix).collect::<Vec<_>>();
    let num_elements = dims.iter().product::<usize>();
    let bytes_per_element = 4; // For f32
    let expected_length = num_elements * bytes_per_element;
    println!("{:?}", &data.float_data);

    if data.raw_data.len() != expected_length {
        return Err(OnnxError::ShapeMismatch("Data length mismatch.".to_string()));
    }

    let data_array = raw_data_to_array(&data, dims).unwrap();

    Ok((data_array, data.name))
}

pub fn load_predictions(file_path: &String) -> Result<ArrayD<f32>, OnnxError> {
    // Read the file contents into a buffer
    let buffer = std::fs::read(file_path).expect("failed to open the file");
    let mut data = TensorProto::new();

    // Decode the buffer using the generated Rust structs
    data
        .merge_from_bytes(&buffer)
        .expect("Failed to parse .pb file");

    let dims = data.dims.iter().map(|&d| d as usize).collect::<Vec<_>>();

    let num_elements = dims.iter().product::<usize>();
    let bytes_per_element = 4; // For f32
    let expected_length = num_elements * bytes_per_element;

    if data.raw_data.len() != expected_length {
        return Err(OnnxError::ShapeMismatch("Data length mismatch.".to_string()));
    }

    raw_data_to_array(&data, dims)
}

/***********TO CHECK***********/
pub fn read_initialiazers(model_initializers: &[TensorProto] ) -> IndexMap<String, Array<f32, IxDyn>> {
    let mut initializer_set: IndexMap<String, Array<f32, IxDyn>> = IndexMap::new();

    for initializer in model_initializers {
        // Prepare to hold the data
        let data:Vec<f32>;
        // Check the data type and extract the data accordingly
        match initializer.data_type {
            1 => { // Floating-point data
                if initializer.float_data.is_empty() {
                    data = initializer.raw_data.chunks(4)
                        .map(|chunk| {
                            let arr: [u8; 4] = chunk
                                .try_into()
                                .expect("Slice with incorrect length");
                            LittleEndian::read_f32(&arr) as f32
                        })
                        .collect();
                } else {
                    data = initializer.float_data.iter().map(|&val| val as f32).collect();
                }
            },
            7 => { // 64-bit integer data
                data = initializer.int64_data.iter().map(|&val| val as f32).collect();
            },
            _ => panic!("Unsupported tensor data type")
        }

        let dims = initializer.dims.iter().map(|&d| d as Ix).collect::<Vec<_>>();

        let dynamic_dims = IxDyn(&dims);


        let ndarray_data = Array::from_shape_vec(dynamic_dims, data)
            .map_err(|_| "Failed to create ndarray from data".to_string()).unwrap();
        initializer_set.insert(initializer.name.clone(), ndarray_data);
    }

    initializer_set
}

pub fn print_nodes(model: &ModelProto) {
    let mut optypes = HashSet::new();


    // Access the graph using the .as_ref() method
    if let Some(graph) = model.graph.as_ref() {
        //println!("Initializer: {:?}", &graph.initializer);
        // Iterate through the nodes in the graph
        for node in &graph.node {
            //println!("Node Name: {}", node.name);
            //println!("Operation Type: {}", node.op_type);
            optypes.insert(&node.op_type);

            // You can access other properties of the node as needed
            // For example, to print the input and output names:
            println!("OpType: {:?}", &node.op_type);
            println!("Input Names: {:?}", &node.input);
            println!("Output Name: {:?}", &node.output);
            println!("Node Name: {:?}", &node.name);
            println!("Attributes: {:?}", &node.attribute);
            println!();

            //println!("Full node: {:?}", &node);
        }
        // Now, 'node_initializers' contains the initializers for this node
    } else {
        println!("No graph found in the model.");
    }
    println!("{:?}", optypes);
}

pub fn argmax(arr: &[f32]) -> Option<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
}


pub fn argmax_per_row(matrix: &ArrayD<f32>) -> Vec<usize> {
    matrix
        .axis_iter(Axis(0)) // Iterate over rows
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0) // Default to 0 if the row is empty
        })
        .collect()
}

pub fn model_proto_to_struct(model: &ModelProto, initializer_set: &mut IndexMap<String, ArrayD<f32>>)
    ->Vec<Box<dyn op_operator::Operator>>{
    let mut model_vec: Vec<Box<dyn op_operator::Operator>> = Vec::new();

    if let Some(graph) = model.graph.as_ref() {

        // Iterate through the nodes in the graph
        for node in &graph.node {
            match node.op_type.as_str() {
                "Add" => {
                    model_vec.push(Box::new(op_add::Add::new(node,initializer_set)));
                },
                "BatchNormalization" => {
                    model_vec.push(Box::new(op_batchnorm::BatchNorm::new(
                        node,initializer_set)))
                },
                "Conv" => {
                    model_vec.push(Box::new(op_conv::Conv::new(
                       node, initializer_set
                    )))
                },
                "Flatten" =>{
                    if node.attribute.is_empty(){
                        model_vec.push(Box::new(op_flatten::Flatten::new(node, initializer_set)))
                    }
                },
                "Gemm" => {
                    model_vec.push(Box::new(op_gemm::Gemm::new(node, initializer_set)));
                },
                "GlobalAveragePool" => {
                    model_vec.push(Box::new(op_globalaveragepooling::
                    GlobalAveragePool::new(node, initializer_set)));
                },
                "MatMul" => {

                    model_vec.push(Box::new(op_matmul::MatMul::new(node, initializer_set)));
                },
                "MaxPool" => {

                    model_vec.push(Box::new(op_maxpool::MaxPool::new(
                        node, initializer_set
                    )))
                },
                "Relu" => {
                    model_vec.push(Box::new(op_relu::ReLU::new(
                        node, initializer_set
                    )));
                },
                "Reshape" => {
                        model_vec.push(Box::new(op_reshape::Reshape::new(
                            node,initializer_set
                        )));

                },
                "Concat"=>{
                    model_vec.push(Box::new(op_concat::Concat::new(
                        node, initializer_set
                    )))
                }
                "Dropout"=>{
                    model_vec.push(Box::new(op_dropout::Dropout::new(
                        node, initializer_set
                    )))
                }
                "Softmax"=>{
                    model_vec.push(Box::new(op_softmax::Softmax::new(
                        node, initializer_set
                    )))
                }
                _ => {
                    //Insert the error OperationNotImplemented
                    // TODO Handle the case where the initializer is not found, eventually blocking the whole program
                }
            }

        }
    }
    model_vec
}

pub fn compute_error_rate(vec1: &[usize], vec2: &[usize]) -> Result<f32, &'static str> {

    if vec1.is_empty() || vec2.is_empty() {
        return Err("Vectors cannot be empty.");
    }

    if vec1.len() != vec2.len() {
        return Err("Vectors must be of the same length.");
    }


    let count = vec1.iter().zip(vec2.iter()).filter(|(&x, &y)| x != y).count();
    Ok(count  as f32/vec1.len() as f32)
}

pub fn compute_accuracy(vec1: &[usize], vec2: &[usize]) -> Result<f32, &'static str> {

    if vec1.is_empty() || vec2.is_empty() {
        return Err("Vectors cannot be empty.");
    }

    if vec1.len() != vec2.len() {
        return Err("Vectors must be of the same length.");
    }


    let count = vec1.iter().zip(vec2.iter()).filter(|(&x, &y)| x == y).count();
    Ok(count  as f32/vec1.len() as f32)
}

pub fn display_model_info(model_name: String, model_version: i64, number_of_nodes: usize) {
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    table.set_titles(Row::new(vec![
        Cell::new("ONNX model information")
            .with_style(Attr::Bold)
            .with_hspan(2)
    ]));
    table.add_row(row![
        "Model name",
        model_name,
    ]);
    table.add_row(row![
        "Model version",
        model_version,
    ]);
    table.add_row(row![
        "Total number of nodes",
        number_of_nodes,
    ]);
    table.printstd();
    println!("Execution Starting...\n");
}


