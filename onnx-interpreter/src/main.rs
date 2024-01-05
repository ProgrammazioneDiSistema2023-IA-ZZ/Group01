mod auxiliary_functions;
mod parser_code;
mod ops;
mod utils_images;

extern crate protobuf;

use std::collections::HashMap;
use ndarray::{ArrayD, Axis, IxDyn};
use std::env;
use std::fs;
use std::time::Instant;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::auxiliary_functions::{
                            load_data, read_initialiazers, load_model, model_proto_to_struct,
                            print_nodes, argmax, load_predictions, argmax_per_row,
                            compute_error_rate, compute_accuracy, display_model_info};

use crate::utils_images::{ serialize_image };

mod display;
pub mod errors;

use display::menu;


fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let flag_serialize = true;

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

    display_model_info(&chosen_model, version, model_read.len());

    let mut inputs: HashMap<String, ArrayD<f32>> = HashMap::new();

    // PER MNIST serialize_g_image_to_pb("D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\images\\img_2.jpg",
           //                 "D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\models\\mnist-8\\test_data_set_0\\test.pb");

    //let (input_image, input_name) = load_data(&data_path).unwrap();

    //inputs.insert(input_name, input_image);//ndarray::stack(Axis(0), &[input_image.clone().index_axis_move(Axis(0), 0).view(), input_image.clone().index_axis_move(Axis(0), 0).view()]).unwrap());

    let test_data_path = format!("models/{}/dataset", &chosen_model);
    let result_test_path = format!("models/{}/test_dataset/", &chosen_model);

    if flag_serialize {
        serialize_image(&test_data_path, &result_test_path, &chosen_model);
    }
    let mut arrays = Vec::new();

    match fs::read_dir(format!("{}/images", &result_test_path)){
        Ok(dir)=>{
            for file_res in dir{

                let file_path = file_res.unwrap().path().to_str().unwrap().to_string();
                // Check if the file is DS_Store, if so, skip it
                if file_path.contains(".DS_Store")  {
                    continue;
                }


                let label_file_name = file_path.split('/').last().unwrap();
                let label_path = format!("{}/labels/{}", &result_test_path, label_file_name);

                let (input_image, input_name) = load_data(&file_path).unwrap();
                let label = load_predictions(&label_path).unwrap();
                arrays.push((input_image, label));
            }
        },
        Err(_)=>{panic!("Not a directory")}
    }

    let (images, labels): (Vec<_>, Vec<_>) = arrays.into_iter().map(|(a, b)| (a, b)).unzip();

    let batch = images.len();
    let shape = images[0].shape();
    let c = shape[1].clone();
    let h = shape[2].clone();
    let w = shape[3].clone();
    let new_s = vec![batch, c, h, w];

    let flat_vec: Vec<f32> = images.into_iter()
        .flat_map(|array| array.into_raw_vec())
        .collect();


    let input_stack= ArrayD::
    from_shape_vec(IxDyn(&new_s), flat_vec).unwrap();

    let shape_label = labels[0].shape();
    let new_s_label = vec![batch, shape_label[1]];

    let flat_labels:Vec<f32> = labels.into_iter()
        .flat_map(|array| array.into_raw_vec())
        .collect();


    let label_stack= ArrayD::
    from_shape_vec(IxDyn(&new_s_label), flat_labels).unwrap();


    let input_name = model.graph.input[0].name.clone();
    //println!("{:?}", &model.graph.input);
    //println!("Input name: {}", input_name);

    println!("Input: {:?}", label_stack);

    inputs.insert(input_name, input_stack);

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
    //println!("Final output: {:?}", final_output);

    let final_result = argmax_per_row(final_output);
    println!("Final result: {:?}", &final_result);

    let predictions = load_predictions(&output_path).unwrap();
    //let final_predictions = argmax_per_row(&predictions);
    //println!("Predictions: {:?}", &final_predictions);


    let final_predictions = argmax_per_row(&label_stack);
    println!("Predictions: {:?}", &final_predictions);

    let error_rate = compute_error_rate(&final_result, &final_predictions).unwrap();
    println!("Error rate: {}", error_rate);

    let accuracy = compute_accuracy(&final_result, &final_predictions).unwrap();
    println!("Accuracy: {}", accuracy);
}



