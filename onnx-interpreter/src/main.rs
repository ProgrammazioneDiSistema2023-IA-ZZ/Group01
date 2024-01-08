mod auxiliary_functions;
mod parser_code;
mod ops;
mod utils_images;

extern crate protobuf;

use crate::ops::op_operator::Operator;

use std::collections::{HashMap, HashSet};
use ndarray::{ArrayD, IxDyn, s};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::auxiliary_functions::{load_data, read_initialiazers, load_model, model_proto_to_struct, load_ground_truth, argmax_per_row, compute_error_rate, compute_accuracy, display_model_info, print_results};

use crate::utils_images::{ serialize_images };

mod display;
pub mod errors;
mod labels_mapping;

use display::menu;

use rayon::prelude::*;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (chosen_model, verbose, test_dataset, folder_name) = menu();

    let model_path = PathBuf::from("models").join(&chosen_model).join("model.onnx");

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

    //let mut inputs: HashMap<String, ArrayD<f32>> = HashMap::new();

    // PER MNIST serialize_g_image_to_pb("D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\images\\img_2.jpg",
           //                 "D:\\PoliTo\\Progetto\\Group01\\onnx-interpreter\\models\\mnist-8\\test_data_set_0\\test.pb");

    let test_input_path = PathBuf::from("models").join(&chosen_model).join("test_data_set_0").join("input_0.pb");
    let test_label_path = PathBuf::from("models").join(&chosen_model).join("test_data_set_0").join("output_0.pb");

    let custom_dataset_path = PathBuf::from("models").join(&chosen_model).join(&folder_name);
    let custom_dataset_serialized_path = PathBuf::from("models").join(&chosen_model)
        .join(folder_name + "_serialized");

    let mut label_stack;
    let mut file_paths = vec![];

    let images_vec = match test_dataset{
        true=>{
            let (input_image, _) = load_data(&test_input_path).unwrap();
            file_paths.push(test_input_path.to_str().unwrap().to_string());
            label_stack = load_ground_truth(&test_label_path).unwrap();
            let mut images = vec![];
            for i in 0..input_image.shape()[0]{
                images.push(input_image.slice(s![i..i+1, .., .., ..]).to_owned().into_dyn());
            }
            images
            //inputs.insert(input_name, input_image);
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
                        file_paths.push(file_path.to_str().unwrap().to_string());
                        let label_file_name = filename.clone() + "." + extension;
                        let label_path = custom_dataset_serialized_path.join("labels").join(label_file_name);
                        let (input_image, _) = load_data(&file_path).unwrap();
                        //println!("{:?}", label_path);
                        let label = load_ground_truth(&label_path).unwrap();
                        arrays.push((input_image, label));
                    }
                },
                Err(_)=>{panic!("Not a directory")}
            }

            let (images, labels): (Vec<ArrayD<f32>>, Vec<ArrayD<f32>>) = arrays.into_iter().unzip();
            let batch = images.len();
            /*
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

             */

            let shape_label = labels[0].shape();
            let new_s_label = vec![batch, shape_label[1]];

            let flat_labels:Vec<f32> = labels.into_iter()
                .flat_map(|array| array.into_raw_vec())
                .collect();


            label_stack= ArrayD::from_shape_vec(IxDyn(&new_s_label), flat_labels).unwrap();

            images
        }
    };
    //println!("{:?}", &inputs);
    //println!("{:?}", &label_stack);

    //ndarray::stack(Axis(0), &[input_image.clone().index_axis_move(Axis(0), 0).view(), input_image.clone().index_axis_move(Axis(0), 0).view()]).unwrap());

    let input_name = model.graph.input[0].name.clone();
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
    let (paths, fork_nodes, join_nodes) = find_dependencies(&mut model_read);

    let start = Instant::now();

    let model_final_output = images_vec.par_iter()
        .map(|img| {

            let bar = ProgressBar::new(model_read.len() as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("\n{bar:60.green/white} {percent}% [{pos}/{len} nodes]")
                    .unwrap()
                    .progress_chars("█▁"),
            );
            let mut inputs = HashMap::new();
            inputs.insert(input_name.clone(), img.clone());

            //(paths, fork_nodes, join_nodes)
            let mut node_index = 0;
            while node_index < model_read.len() { // Use iter instead of iter_mut if possible
                // Update progress bar safely (if necessary)
                let node = &model_read[node_index];
                bar.println(format!("🚀 Running node: {} {}", node.get_op_type().bold(), node.get_node_name().bold()));
                let start_node = Instant::now();
                let output = node.execute(&inputs)
                    .expect("Node execution failed"); // Handle the error properly
                let run_time_node = start_node.elapsed();
                if verbose{
                    bar.println(node.to_string(&inputs, &output, &run_time_node));
                }

                for (i, out) in output.iter().enumerate() {
                    inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
                }

                if let Some(branches) = paths.get(&node.get_node_name()) {
                    let branches_len = branches.iter().map(|v| v.len()).collect::<Vec<usize>>();
                    let inside_inputs: Vec<HashMap<String, ArrayD<f32>>> = branches.par_iter().enumerate().map(|(i, branch)| {
                        let mut inner_inputs = HashMap::new();
                        for (i, out) in output.iter().enumerate() {
                            inner_inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
                        }

                        let node_executed = branches_len.iter().take(i).sum::<usize>();
                        let model_to_run: Vec<&Box<dyn Operator>> = model_read
                            [node_index+1+node_executed .. node_index+1+node_executed + branch.len()]
                            .iter()
                            .collect();

                        for node in model_to_run {
                            let start_node = Instant::now();
                            let output = node.execute(&inner_inputs)
                                .expect("Node execution failed"); // Consider handling errors more gracefully
                            let run_time_node = start_node.elapsed();
                            if verbose{
                                bar.println(node.to_string(&inner_inputs, &output, &run_time_node));
                            }
                            for (i, out) in output.iter().enumerate() {
                                inner_inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
                            }
                        }

                        inner_inputs
                    }).collect::<Vec<HashMap<String, ArrayD<f32>>>>();

                    for hash in inside_inputs {
                        for (key, value) in hash {
                            inputs.insert(key, value);
                        }
                    }

                    let total_node_run = branches_len.iter().sum::<usize>();
                    node_index += total_node_run;
                    bar.inc(total_node_run as u64);
                }

                node_index+=1;
                bar.inc(1);
            }

            // Return the output for this particular input
            inputs.get(final_layer_name)
                .expect("Final layer output not found").clone()
        })
        .collect::<Vec<ArrayD<f32>>>();

    // Execute nodes in sorted order
    /*
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
            let start_node = Instant::now();
            let output = node.execute(&inputs).unwrap();
            let run_time_node = start_node.elapsed();
            if verbose{
                bar.println(node.to_string(&inputs, &output, &run_time_node));
            }
            for (i, out) in output.iter().enumerate(){
                inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
            }
            bar.inc(1);
    }
    bar.finish();

     */
    let run_time = start.elapsed();
    println!("\n✅  The network has been successfully executed in {:?}\n", run_time);

    let shape = model_final_output[0].shape();
    let batch= model_final_output.len();
    let c = shape[1].clone();
    let new_s = vec![batch, c ];

    let flat_vec: Vec<f32> = model_final_output.into_iter()
        .flat_map(|array| array.into_raw_vec())
        .collect();


    let final_output = ArrayD::
    from_shape_vec(IxDyn(&new_s), flat_vec).unwrap();

    let predictions = argmax_per_row(&final_output);

    let ground_truth_labels = argmax_per_row(&label_stack);

    let error_rate = compute_error_rate(&predictions, &ground_truth_labels).unwrap();

    let accuracy = compute_accuracy(&predictions, &ground_truth_labels).unwrap();

    print_results(&chosen_model, &file_paths, &predictions, &ground_truth_labels, &error_rate, &accuracy);

}

fn find_dependencies(model_read: &mut Vec<Box<dyn Operator>>)
                     -> (HashMap<String, Vec<Vec<String>>>, HashSet<String>, HashSet<String>){
    let mut dependencies = HashMap::new();
    let mut join_nodes = HashSet::new();

    for node in model_read.iter() {
        let node_name = node.get_node_name();
        dependencies.entry(node_name.clone()).or_insert(vec![]);

        let input_names = node.get_inputs();
        if input_names.len() > 1 {
            join_nodes.insert(node_name.clone());
        }
        for input in input_names {
            dependencies.entry(input.clone()).or_default().push(node_name.clone());
        }
    }

    for successors in dependencies.values_mut() {
        if successors.len() > 1 {
            successors.retain(|s| !join_nodes.contains(s));
        }
    }

    let fork_nodes: HashSet<String> = dependencies
        .iter()
        .filter_map(|(node, successors)| {
            if successors.len() > 1 {
                Some(node.clone())
            } else {
                None
            }
        })
        .collect();

    let mut current_sub_path = Vec::new();
    let mut paths = HashMap::new();
    let mut start_path: Option<String> = None;
    let mut previous_fork_node: Option<&String> = None;

    for node in model_read {
        let node_name = node.get_node_name();

        // Check if the node is a fork node
        if fork_nodes.contains(&node_name) {
            start_path = Some(node_name.clone());
        } else if let Some(sp) = &start_path {
            // Check if the node is a join node
            if join_nodes.contains(&node_name) {
                previous_fork_node = Some(sp);
                start_path = None;
            } else {
                // Add node to the current branch
                current_sub_path.push(node_name.clone());
                if dependencies.get(&node_name).unwrap().iter().any(|v| join_nodes.contains(v)) {
                    //salva il path corrente e resettalo
                    if !current_sub_path.is_empty() {
                        paths.entry(sp.clone()).or_insert_with(Vec::new).push(current_sub_path.clone());
                        current_sub_path.clear();
                    }
                }
            }
        }
    }

    println!("Number of fork nodes: {}", paths.keys().len());
    for (start, path_nodes) in &paths {
        println!();
        println!("Node fork:{}", start);
        println!("Path: {:?}", path_nodes);
    }

    (paths, fork_nodes, join_nodes)
}



