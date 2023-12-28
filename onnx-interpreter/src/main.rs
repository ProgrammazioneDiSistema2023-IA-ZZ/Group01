mod auxiliary_functions;
mod parser_code;
mod ops;

extern crate protobuf;

use std::collections::HashMap;
use ndarray::{ArrayD, Axis};
use std::env;
use std::time::Instant;
use colored::Colorize;
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};

use crate::auxiliary_functions::{
                            load_data, read_initialiazers, load_model, model_proto_to_struct,
                            print_nodes, argmax, load_predictions, argmax_per_row,
                            compute_error_rate, compute_accuracy, display_model_info,
                            serialize_g_image_to_pb};

mod display;
pub mod errors;

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

    let mut initialiazers: IndexMap<String, ArrayD<f32>> = IndexMap::new();
    initialiazers = read_initialiazers(&model.graph.initializer);

    let mut model_read = model_proto_to_struct(&model, &mut initialiazers);
    //to delete/change the print when code is ok, for now it's useful for debugging
    /*for node in &model_read{
        println!("{}", node.to_string());
        println!();
    }*/

    display_model_info(chosen_model, version, model_read.len());

    let mut inputs: IndexMap<String, ArrayD<f32>> = IndexMap::new();

    // PER MNIST serialize_g_image_to_pb("D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\images\\img_2.jpg",
           //                 "D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\models\\mnist-8\\test_data_set_0\\test.pb");

    let (input_image, input_name) = load_data(&data_path).unwrap();

    inputs.insert(input_name, input_image);//ndarray::stack(Axis(0), &[input_image.clone().index_axis_move(Axis(0), 0).view(), input_image.clone().index_axis_move(Axis(0), 0).view()]).unwrap());


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

    let bar = ProgressBar::new(model_read.len() as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("\n{bar:60.green/white} {percent}% [{pos}/{len} nodes]")
            .unwrap()
            .progress_chars("█▁"),
    );
    let start = Instant::now();
    for node in model_read.iter_mut() {
            bar.println(format!("🚀 Running node: {} {}", node.get_op_type().bold(), node.get_node_name().bold()));
            let output = node.execute(&inputs).unwrap();
            if verbose{
                bar.println(node.to_string(&inputs, &output));
            }
            for (i, out) in output.iter().enumerate(){
                inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
            }
            bar.inc(1);
    }
    bar.finish();
    let run_time = start.elapsed();
    println!("\n✅  The network has been successfully executed in {:?}\n", run_time);

    let final_output = inputs.get(final_layer_name).unwrap();
    println!("Final output: {:?}", final_output);

    let final_result = argmax_per_row(final_output);
    println!("Final result: {:?}", &final_result);

    let predictions = load_predictions(&output_path).unwrap();
    let final_predictions = argmax_per_row(&predictions);
    println!("Predictions: {:?}", &final_predictions);

    /*let sub = final_output - &predictions;
    for elem in sub.iter() {
        print!("{:?}, ", elem);
    }
    println!(); // New line at the end*/
    /*
            let error_rate = compute_error_rate(&final_result, &final_predictions).unwrap();
            println!("Error rate: {}", error_rate);

            let accuracy = compute_accuracy(&final_result, &final_predictions).unwrap();
            println!("Accuracy: {}", accuracy);

             */
}



