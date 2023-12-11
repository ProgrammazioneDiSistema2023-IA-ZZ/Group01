mod auxiliary_functions;
mod parser_code;
mod ops;

extern crate protobuf;

use std::collections::HashMap;
use ndarray::ArrayD;
use std::env;
use indicatif::{ProgressBar, ProgressStyle};

use crate::auxiliary_functions::{
                            load_data, read_initialiazers, load_model, model_proto_to_struct,
                            print_nodes, argmax, load_predictions, argmax_per_row,
                            compute_error_rate, compute_accuracy, display_model_info };

mod display;
use display::menu;


fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (chosen_model, verbose) = menu();

    let data_path = format!("models/{}/test_data_set_0/input_0.pb", chosen_model);
    let output_path = format!("models/{}/test_data_set_0/output_0.pb", chosen_model);
    let model_path = format!("models/{}/model.onnx", chosen_model);

    // Load ONNX model
    let model = load_model(&model_path);

    //print_nodes(&model);
    //println!("{:?}", model.graph.initializer);

    let mut version= 0;
    for op_set in &model.opset_import{
        version = op_set.version.clone();
    }
/*
    if version != 8{
        panic!("This ONNX Parser works only for version 8");
    }

 */

    let mut initialiazers: HashMap<String, ArrayD<f32>> = HashMap::new();
    initialiazers = read_initialiazers(&model.graph.initializer);

    let mut model_read = model_proto_to_struct(&model, &mut initialiazers);
    //to delete/change the print when code is ok, for now it's useful for debugging
    /*for node in &model_read{
        println!("{}", node.to_string());
        println!();
    }*/
    println!("The total number of nodes is {}", model_read.len());

    display_model_info(chosen_model, version);

    let mut inputs: HashMap<String, ArrayD<f32>> = HashMap::new();

    let (input_image, input_name) = load_data(&data_path).unwrap();
    inputs.insert(input_name, input_image);


    let final_layer_name = &model.graph.output[0].name;

    /*let mut dependencies = HashMap::new();


    for op in &model_read{
        let output = op.get_output_name();
        for input in op.get_inputs() {
            dependencies.entry(input).or_insert_with(HashSet::new).insert(output.clone());
        }
    }

    // Sort nodes topologically
    let sorted_node_names = topological_sort(dependencies);*/

    // Execute nodes in sorted order
    for node in model_read.iter_mut() {
            println!("{}", node.to_string(&verbose));
            let output = node.execute(&inputs).unwrap();
            inputs.insert(node.get_output_name(), output.clone());
    }

    let final_output = inputs.get(final_layer_name).unwrap();
    println!("Final output: {:?}", final_output);

    let final_result = argmax_per_row(final_output);
    println!("Final result: {:?}", &final_result);

    let predictions = load_predictions(&output_path).unwrap();
    let final_predictions = argmax_per_row(&predictions);
    println!("Predictions: {:?}", &final_predictions);
    /*
            let error_rate = compute_error_rate(&final_result, &final_predictions).unwrap();
            println!("Error rate: {}", error_rate);

            let accuracy = compute_accuracy(&final_result, &final_predictions).unwrap();
            println!("Accuracy: {}", accuracy);

             */
}



