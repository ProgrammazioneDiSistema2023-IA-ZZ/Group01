use ndarray::{Array, Ix, IxDyn, ArrayD, Axis};
use std::fs;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::File;
use std::io::Write;
use byteorder::{ByteOrder, LittleEndian};
use prettytable::{format, row, Row, Table, Cell, Attr};
extern crate protobuf;
use protobuf::{Message};
use crate::parser_code::onnx_ml_proto3::{ModelProto, TensorProto};
use crate::ops::*;
use crate::errors::OnnxError;
use std::path::PathBuf;
use colored::Colorize;
use crate::labels_mapping::IMAGENET_CLASSES;


pub fn load_model(file_path: &PathBuf) -> ModelProto {
    // Load and deserialize your .onnx file here
    let model_bytes = fs::read(file_path).expect("Failed to read .onnx file");
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

pub fn load_data(file_path: &PathBuf) -> Result<(ArrayD<f32>, String), OnnxError> {
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
    //println!("{:?}", dims);
    //println!("{:?}", num_elements);
    //println!("{:?}", &data.float_data);

    if data.raw_data.len() != expected_length {
        //println!("{:?}", data.raw_data.len());
        //println!("{:?}", expected_length);
        return Err(OnnxError::ShapeMismatch("Data length mismatch.".to_string()));
    }

    let data_array = raw_data_to_array(&data, dims).unwrap();
    //let temp = TensorProto{raw_data: vec![], ..data.clone()};
    //println!("{:?}", temp);

    Ok((data_array, data.name))
}

pub fn load_ground_truth(file_path: &PathBuf) -> Result<ArrayD<f32>, OnnxError> {
    // Read the file contents into a buffer
    let buffer = fs::read(file_path).expect("Failed to open the file");
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
pub fn read_initialiazers(model_initializers: &[TensorProto] ) -> HashMap<String, Array<f32, IxDyn>> {
    let mut initializer_set: HashMap<String, Array<f32, IxDyn>> = HashMap::new();

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

pub fn print_results (model: &String, files: &Vec<String>, predictions: &Vec<usize>, ground_truth: &Vec<usize>, error_rate: &f32, accuracy: &f32){
    let file_name = "results.txt";
    let max_elements = 50;
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    if model=="mobilenetv2-7" || model=="resnet18-v1-7" || model=="resnet18-v2-7"{
        table.set_titles(row![
            "Input file".bold(),
            "Prediction".bold(),
            "Predicted label".bold(),
            "Ground truth".bold(),
            "Ground truth label".bold()
        ]);
    }
    else{
        table.set_titles(row![
            "Input file".bold(),
            "Prediction".bold(),
            "Ground truth".bold()
        ]);
    }
    for (i, f) in files.iter().enumerate(){
        let correct_prediction = predictions[i]== ground_truth[i];
        let prediction_string = predictions[i].to_string();
        let gt_string = ground_truth[i].to_string();
        let (f_style, prediction_style, gt_style) = if correct_prediction {
            (f.bright_green(), prediction_string.bright_green(), gt_string.bright_green())
        } else {
            (f.bright_red(), prediction_string.bright_red(), gt_string.bright_red())
        };
        if model=="mobilenetv2-7" || model=="resnet18-v1-7" || model=="resnet18-v2-7"{
            let predicted_label = IMAGENET_CLASSES[predictions[i]];
            let ground_truth_label = IMAGENET_CLASSES[ground_truth[i]];
            let (pl_style, gtl_style) = if correct_prediction {
                (predicted_label.bright_green(), ground_truth_label.bright_green())
            } else {
                (predicted_label.bright_red(), ground_truth_label.bright_red())
            };
            table.add_row(row![
                f_style, prediction_style, pl_style, gt_style, gtl_style
            ]);
        }
        else{
            table.add_row(row![
            f_style, prediction_style, gt_style
        ]);
        }
    }
    if files.len()<=max_elements {
        println!("Results:\n\n{}", table.to_string());
    }
    else{
        let mut reduced_table = Table::new();
        reduced_table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
        if model=="mobilenetv2-7" || model=="resnet18-v1-7" || model=="resnet18-v2-7"{
            reduced_table.set_titles(row![
            "Input file".bold(),
            "Prediction".bold(),
            "Predicted label".bold(),
            "Ground truth".bold(),
            "Ground truth label".bold()
        ]);
        }
        else{
            reduced_table.set_titles(row![
            "Input file".bold(),
            "Prediction".bold(),
            "Ground truth".bold()
        ]);
        }
        for (i, f) in files.iter().take(max_elements).enumerate(){
            let correct_prediction = predictions[i]== ground_truth[i];
            let prediction_string = predictions[i].to_string();
            let gt_string = ground_truth[i].to_string();
            let (f_style, prediction_style, gt_style) = if correct_prediction {
                (f.bright_green(), prediction_string.bright_green(), gt_string.bright_green())
            } else {
                (f.bright_red(), prediction_string.bright_red(), gt_string.bright_red())
            };
            if model=="mobilenetv2-7" || model=="resnet18-v1-7" || model=="resnet18-v2-7"{
                let predicted_label = IMAGENET_CLASSES[predictions[i]];
                let ground_truth_label = IMAGENET_CLASSES[ground_truth[i]];
                let (pl_style, gtl_style) = if correct_prediction {
                    (predicted_label.bright_green(), ground_truth_label.bright_green())
                } else {
                    (predicted_label.bright_red(), ground_truth_label.bright_red())
                };
                reduced_table.add_row(row![
                f_style, prediction_style, pl_style, gt_style, gtl_style
            ]);
            }
            else{
                reduced_table.add_row(row![
            f_style, prediction_style, gt_style
        ]);
            }
        }
        println!("The dataset size is too big for all the results to be printed in the console.\nTo improve \
        readability, the results have been stored in the file \"{}\".\nHowever, here is a sneak peek to the first {} \
        results:\n\n{}", &file_name, max_elements, reduced_table.to_string());
        let mut out = File::create(file_name).unwrap();
        out.write_all(strip_ansi_codes(table.to_string()).as_bytes()).unwrap();
    }
    println!("Error rate: {}\nAccuracy: {}", error_rate, accuracy);
}

fn strip_ansi_codes(input: String) -> String {
    let re = regex::Regex::new("\x1B\\[[0-9;]*[a-zA-Z]").unwrap();
    re.replace_all(input.as_str(), "").to_string()
}

pub fn model_proto_to_struct(model: &ModelProto, initializer_set: &mut HashMap<String, ArrayD<f32>>)
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

pub fn display_model_info(model_name: &String, model_version: i64, number_of_nodes: usize) {
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


