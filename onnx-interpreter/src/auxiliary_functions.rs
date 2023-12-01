use ndarray::{Array, Dim, Ix, IxDyn, ArrayView2, ArrayD, Axis};
use std::fs;
use std::iter::FromIterator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryInto;

extern crate protobuf;
use protobuf::{Message};
use crate::parser_code::onnx_ml_proto3::{ModelProto, TensorProto};

use byteorder::{ByteOrder, LittleEndian};


use crate::lib::{Operator, Add, ReLU, MatMul, MaxPool, Conv, Reshape, AutoPad, Flatten,
                 GlobalAveragePool, Gemm, BatchNorm };


pub fn topological_sort(dependencies: HashMap<String, HashSet<String>>) -> Vec<String> {
    let mut in_degree = HashMap::new();
    let mut graph = HashMap::new();

    // Initialize in_degree and graph
    for (node, deps) in dependencies.iter() {
        in_degree.entry(node).or_insert(0);
        println!("Node: {:?}", node);
        for dep in deps {
            *in_degree.entry(dep).or_insert(0) += 1;
            graph.entry(node.clone()).or_insert_with(HashSet::new).insert(dep.clone());
        }
    }

    // Find all nodes with in_degree 0
    let mut queue = VecDeque::new();
    for (node, &degree) in in_degree.iter() {
        println!("Node: {}", node);
        println!("Degree: {}", degree);
        if degree == 0 {
            queue.push_back(node.clone());
        }
    }

    let mut order = Vec::new();
    while let Some(node) = queue.pop_front() {
        order.push(node.clone());

        if let Some(neighbors) = graph.get(node.as_str()) {
            for neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(&neighbor.to_string()) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    if order.len() == in_degree.len() {
        order
    } else {
        vec![] // Cycle detected or graph is incomplete
    }
}

pub fn load_model(file_path: &str) -> ModelProto {
    // Load and deserialize your .onnx file here
    let model_bytes = std::fs::read(file_path).expect("Failed to read .onnx file");
    let mut model = ModelProto::new(); // Create an instance of ModelProto

    // Use the parse_from_bytes method to deserialize the model
    model
        .merge_from_bytes(&model_bytes)
        .expect("Failed to parse .onnx file");

    model
}

pub fn load_data(file_path: &str) -> Result<(ArrayD<f32>, String), String> {
    // Read the file contents into a buffer
    let buffer = fs::read(file_path).map_err(|e| format!("Failed to open the file: {}", e))?;
    let mut data = TensorProto::new();

    // Decode the buffer using the generated Rust structs
    data
        .merge_from_bytes(&buffer)
        .expect( "Failed to parse .pb file");

    let dims = data.dims.iter().map(|&d| d as Ix).collect::<Vec<_>>();
    let num_elements = dims.iter().product::<usize>();
    let bytes_per_element = 4; // For f32
    let expected_length = num_elements * bytes_per_element;

    if data.raw_data.len() != expected_length {
        return Err("Data length mismatch.".to_string());
    }

    let data_array: Vec<f32> = data.raw_data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Slice with incorrect length")) as f32)
        .collect();

    let dynamic_dims = IxDyn(&dims);
    let ndarray_data = Array::from_shape_vec(dynamic_dims, data_array)
        .map_err(|_| "Failed to create ndarray from data".to_string())?;


    Ok((ndarray_data, data.name))
}

pub fn read_initialiazers(model_initializers: &[TensorProto], initializer_set: &mut HashMap<String, ArrayD<f32>>)
    -> () {
    //let mut initializer_set: HashMap<String, Array<f32, IxDyn>> = HashMap::new();

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

pub fn model_proto_to_struct(model: &ModelProto, initializer_set: &HashMap<String, Array<f32, IxDyn>>)
    ->Vec<Box<dyn Operator>>{
    let mut model_vec: Vec<Box<dyn Operator>> = Vec::new();

    if let Some(graph) = model.graph.as_ref() {

        // Iterate through the nodes in the graph
        for node in &graph.node {
            match node.op_type.as_str() {
                "Add" => {
                    let initializer = initializer_set.get(&node.input[1]);

                    let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];

                    let add_op = if let Some(init) = initializer {
                        // If initializer exists, use it
                        Add::new(node.name.to_owned(), inputs_name, node.output[0].to_owned(), Some(init.to_owned()))
                    } else {
                        // If initializer does not exist, handle accordingly
                        Add::new(node.name.to_owned(), inputs_name, node.output[0].to_owned(), None)
                    };

                    model_vec.push(Box::new(add_op));

                },
                "BatchNormalization" => {
                    println!("Batch Normalization: {:?}", &node);
                    let gamma = initializer_set.get(&node.input[1]).unwrap();
                    let beta = initializer_set.get(&node.input[2]).unwrap();
                    let mean = initializer_set.get(&node.input[3]).unwrap();
                    let var = initializer_set.get(&node.input[4]).unwrap();

                    model_vec.push(Box::new(BatchNorm::new(
                        node.name.to_owned(),
                        node.input[0].to_owned(),
                        node.output[0].to_owned(),
                        node.attribute[0].f as f32,
                        node.attribute[1].f as f32,
                        node.attribute[2].i,
                        gamma.to_owned(),
                        beta.to_owned(),
                        mean.to_owned(),
                        var.to_owned()

                    )))
                },
                "Conv" => {
                    let kernel_weights = initializer_set.get(&node.input[1]).unwrap();
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
                            "group" => group = attribute.i as usize,
                            "dilations" => dilations = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
                            "pads" => pads = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
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

                    let mut bias = None;
                    if node.name.len() == 3 {
                        bias = Some(initializer_set.get(&node.input[2]).unwrap().to_owned());
                    }

                    model_vec.push(Box::new(Conv::new(
                        node.name.to_owned(),
                        node.input[0].to_owned(),
                        node.output[0].to_owned(),
                        kernel_shape,
                        strides,
                        auto_pad,
                        pads,
                        group,
                        dilations,
                        kernel_weights.to_owned(),
                        bias,
                    )))
                },
                "Flatten" =>{
                    if node.attribute.is_empty(){
                        model_vec.push(Box::new(Flatten::new(node.name.to_owned(), node.input[0].to_owned(), node.output[0].to_owned(), 1)))
                    }
                },
                "Gemm" => {
                    let B = initializer_set.get(&node.input[1]).unwrap();
                    let C = initializer_set.get(&node.input[2]);

                    let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];

                    let gemm_op = if let Some(c) = C{
                        Gemm::new(node.name.to_owned(), inputs_name,
                                  node.output[0].to_owned(),
                                  node.attribute[0].f as f32,
                                  node.attribute[1].f as f32,
                                  node.attribute[2].i,
                                  node.attribute[3].i,
                                  B.to_owned(),
                                  Some(c.to_owned()),
                        )
                    } else{
                        Gemm::new(node.name.to_owned(), inputs_name,
                                  node.output[0].to_owned(),
                                  node.attribute[0].f as f32,
                                  node.attribute[1].f as f32,
                                  node.attribute[2].i,
                                  node.attribute[3].i,
                                  B.to_owned(),
                                  None,
                        )
                    };

                    model_vec.push(Box::new(gemm_op))
                },
                "GlobalAveragePool" => {
                    model_vec.push(Box::new(GlobalAveragePool::new(node.name.to_owned(), node.input[0].to_owned(), node.output[0].to_owned())));
                },
                "MatMul" => {
                    let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
                    model_vec.push(Box::new(MatMul::new(node.name.to_owned(), inputs_name,
                                                        node.output[0].to_owned())));
                },
                "MaxPool" => {
                    let mut auto_pad:Option<AutoPad> = None;             //default is 'NOTSET'
                    let mut ceil_mode : Option<i64> = None;
                    let mut dilations = None;
                    let mut pads= None;
                    let mut strides = None;
                    let mut kernel_shape = None;

                    for attribute in &node.attribute {
                        match attribute.name.as_str() {
                            "ceil_mode" => ceil_mode = Some(attribute.i),
                            "strides" => strides = Some(attribute.ints.iter().map(|&i| i as i64).collect()),
                            "kernel_shape" => kernel_shape = Some (attribute.ints.iter().map(|&i| i as i64).collect()),
                            "dilations" => dilations = {
                                Some(attribute.ints.iter().map(|i| *i as i64).collect::<Vec<i64>>()) },
                            "pads" => pads = Some ( attribute.ints.iter().map(|i| *i as i64).collect::<Vec<i64>>()),
                            "auto_pad"=>{
                                auto_pad = Some(match String::from_utf8(attribute.s.clone()) {
                                    Ok(value) => match value.as_str() {
                                        "SAME_UPPER" => AutoPad::SAME_UPPER,
                                        "SAME_LOWER" => AutoPad::SAME_LOWER,
                                        "VALID" => AutoPad::VALID,
                                        _ => AutoPad::NOTSET,
                                    },
                                    Err(_) => AutoPad::NOTSET,
                                });
                            },
                            // Handle other attributes
                            _ => {}
                        }
                    }
                    model_vec.push(Box::new(MaxPool::new(
                        node.name.to_owned(),
                        node.input[0].to_owned(),
                        node.output[0].to_owned(),
                        kernel_shape.to_owned(),
                        strides.to_owned(),
                        pads.to_owned(),
                        auto_pad,
                        ceil_mode.to_owned(),
                        dilations.to_owned(),
                    )))
                },
                "Relu" => {
                    model_vec.push(Box::new(ReLU::new(node.name.to_owned(), node.input[0].to_owned(), node.output[0].to_owned())));
                },
                "Reshape" => {
                    if let Some(initializer) = initializer_set.get(node.input[1].as_str()) {
                        // Convert the array_base to Vec<i64> here
                        let vec_usize = initializer.iter().map(|&f| f as usize).collect();

                        model_vec.push(Box::new(Reshape::new(
                            node.name.to_owned(),
                            node.input[0].to_owned(),
                            node.output[0].to_owned(),
                            vec_usize
                        )));
                    } else {
                        // TODO Handle the case where the initializer is not found
                        // This could be returning an error or using a default value
                    }
                },
                _ => {
                    // TODO Handle the case where the initializer is not found, eventually blocking the whole program
                }
            }

        }
    }
    model_vec
}
pub fn load_predictions(file_path: &str) -> Result<ArrayD<f32>, String> {
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
        return Err("Data length mismatch.".to_string());
    }

    let data_array: Vec<f32> = data
        .raw_data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Slice with incorrect length")) as f32)
        .collect();

    let data: Vec<f32> = data_array.iter().map(|&x| x as f32).collect();

    let ndarray_data = ArrayD::from_shape_vec(dims.clone(), data).unwrap();

    Ok(ndarray_data)
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

