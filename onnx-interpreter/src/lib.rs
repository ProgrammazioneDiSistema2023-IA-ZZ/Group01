mod auxiliary_functions;
mod parser_code;
mod ops;
mod utils_images;

extern crate protobuf;

use std::collections::HashMap;
use ndarray::{ArrayD, Axis, IxDyn};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::auxiliary_functions::{
    load_data, read_initialiazers, load_model, model_proto_to_struct,
    print_nodes, argmax, load_ground_truth, argmax_per_row,
    compute_error_rate, compute_accuracy, display_model_info};

use crate::utils_images::{ serialize_images };

mod display;
pub mod errors;

use display::menu;


fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (chosen_model, verbose, test_dataset, folder_name, batch_size) = menu();

    let model_path = PathBuf::from(format!("models/{}/model.onnx", chosen_model));

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

    let test_input_path = PathBuf::from(format!("models/{}/test_data_set_0/input_0.pb", chosen_model));
    let test_label_path = PathBuf::from(format!("models/{}/test_data_set_0/output_0.pb", chosen_model));

    let custom_dataset_path = PathBuf::from(format!("models/{}/{}", &chosen_model, &folder_name));
    let custom_dataset_serialized_path = PathBuf::from(format!("models/{}/{}_serialized", &chosen_model, &folder_name));

    let mut label_stack;

    match test_dataset{
        true=>{
            let (input_image, input_name) = load_data(&test_input_path).unwrap();
            label_stack = load_ground_truth(&test_label_path).unwrap();
            inputs.insert(input_name, input_image);
        },
        false=>{
            serialize_images(&custom_dataset_path, &custom_dataset_serialized_path, &chosen_model).unwrap();
            let mut arrays = Vec::new();
            let serialized_folder_path = custom_dataset_serialized_path.join("images");
            match fs::read_dir(serialized_folder_path){
                Ok(dir)=>{
                    for file_res in dir{
                        let file_path = file_res.unwrap().path();
                        let filename = file_path.file_stem().and_then(|stem| stem.to_str()).map(|s| s.to_string()).unwrap();
                        let extension = file_path.extension().and_then(|ext| ext.to_str()).unwrap();
                        if !(extension=="pb"){
                            continue;
                        }
                        let label_file_name = filename.clone() + "." + extension;
                        let label_path = custom_dataset_serialized_path.join("labels").join(label_file_name);
                        let (input_image, _) = load_data(&file_path).unwrap();
                        //println!("{:?}", label_path_string);
                        let label = load_ground_truth(&label_path).unwrap();
                        arrays.push((input_image, label));
                    }
                },
                Err(_)=>{panic!("Not a directory")}
            }

            let (images, labels): (Vec<ArrayD<f32>>, Vec<ArrayD<f32>>) = arrays.into_iter().unzip();
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


            label_stack= ArrayD::from_shape_vec(IxDyn(&new_s_label), flat_labels).unwrap();

            let input_name = model.graph.input[0].name.clone();
            //println!("{:?}", &model.graph.input);
            //println!("Input name: {}", input_name);
            //println!("{:?}", input_stack);
            inputs.insert(input_name, input_stack);
        }
    }
    //println!("{:?}", &inputs);
    //println!("{:?}", &label_stack);

    //ndarray::stack(Axis(0), &[input_image.clone().index_axis_move(Axis(0), 0).view(), input_image.clone().index_axis_move(Axis(0), 0).view()]).unwrap());

    let final_layer_name = &model.graph.output[0].name;
    //println!("{:?}", final_layer_name);

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
    let predictions = argmax_per_row(final_output);
    println!("Network predictions: {:?}", &predictions);
    /*

    match test_dataset{
        true=>{
            //let ground_truth = load_ground_truth(&test_label_path).unwrap();
            let ground_truth_labels = argmax_per_row(&ground_truth);
            println!("Ground truth labels: {:?}", &ground_truth_labels);
        },
        false=>{
            let ground_truth_labels = argmax_per_row(&label_stack);
            println!("Ground truth labels: {:?}", &ground_truth_labels);
        }
    }

     */
    let ground_truth_labels = argmax_per_row(&label_stack);
    println!("Ground truth labels: {:?}", &ground_truth_labels);

    let error_rate = compute_error_rate(&predictions, &ground_truth_labels).unwrap();
    println!("Error rate: {}", error_rate);

    let accuracy = compute_accuracy(&predictions, &ground_truth_labels).unwrap();
    println!("Accuracy: {}", accuracy);
    //println!("Final output: {:?}", final_output);
}



