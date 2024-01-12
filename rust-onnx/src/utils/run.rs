use std::collections::HashMap;
use std::time::Instant;
use colored::Colorize;
use indicatif::{MultiProgress, ParallelProgressIterator};
use ndarray::{ArrayBase, ArrayD, IxDyn, OwnedRepr};
use rayon::prelude::*;
use crate::operators::op_operator::Operator;
use crate::utils::auxiliary_functions::setup_progress_bar;

pub fn run(images_vec: &Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>,
           model_read: &Vec<Box<dyn Operator>>,
           input_name: &String,
           dependencies: &HashMap<String, Vec<Vec<String>>>,
           verbose: &bool,
           final_layer_name: &String) -> ArrayBase<OwnedRepr<f32>, IxDyn> {

    let multi_progress = MultiProgress::new();

    let progress_bar_images = setup_progress_bar(&multi_progress, images_vec.len() as u64, true, "images");
    let progress_bar_nodes = setup_progress_bar(&multi_progress,
                                model_read.len() as u64 *images_vec.len() as u64, false, "executed nodes");

    progress_bar_images.set_position(0);
    progress_bar_nodes.set_position(0);

    let network_timer = Instant::now();

    let model_final_output = images_vec.par_iter().enumerate()
        .map(|(index, img)| {

            let mut inputs = HashMap::new();
            inputs.insert(input_name.clone(), img.clone());

            let mut node_index = 0;
            if let Some(branches) = dependencies.get("START") {
                let branches_len = branches.iter().map(|v| v.len()).collect::<Vec<usize>>();
                let inside_inputs: Vec<HashMap<String, ArrayD<f32>>> = branches.par_iter().enumerate().map(|(i, branch)| {
                    let mut inner_inputs = inputs.clone();

                    let node_executed = branches_len.iter().take(i).sum::<usize>();
                    let model_to_run: Vec<&Box<dyn Operator>> = model_read
                        [node_index + node_executed..node_index + node_executed + branch.len()]
                        .iter()
                        .collect();

                    for node in model_to_run {
                        let output = node.execute(&inner_inputs)
                            .expect("Node execution failed"); // Consider handling errors more gracefully
                        progress_bar_nodes.inc(1);
                        if *verbose {
                            progress_bar_nodes.suspend(||{
                                println!("{}", node.to_string(&inner_inputs, &output, index.to_string()));
                            });
                        }
                        else{
                            progress_bar_nodes.suspend(||{
                                println!("{}", format!("🚀 Executed node: {} {} for image: {}", node.get_op_type().bold(), node.get_node_name().bold(), index.to_string().bold()));
                            });
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
            }

            //(paths, fork_nodes, join_nodes)
            while node_index < model_read.len() { // Use iter instead of iter_mut if possible
                // Update progress bar safely (if necessary)
                let node = &model_read[node_index];
                let output = node.execute(&inputs)
                    .expect("Node execution failed"); // Handle the error properly
                progress_bar_nodes.inc(1);
                if *verbose {
                    progress_bar_nodes.suspend(||{
                        println!("{}", node.to_string(&inputs, &output, index.to_string()));
                    });
                }
                else{
                    progress_bar_nodes.suspend(||{
                        println!("{}", format!("🚀 Executed node: {} {} for image: {}", node.get_op_type().bold(), node.get_node_name().bold(), index.to_string().bold()));
                    });
                }

                for (i, out) in output.iter().enumerate() {
                    inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
                }

                if let Some(branches) = dependencies.get(&node.get_node_name()) {
                    let branches_len = branches.iter().map(|v| v.len()).collect::<Vec<usize>>();
                    let inside_outputs: Vec<HashMap<String, ArrayD<f32>>> = branches.par_iter().enumerate().map(|(i, branch)| {
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
                            let output = node.execute(&inner_inputs)
                                .expect("Node execution failed"); // Consider handling errors more gracefully
                            progress_bar_nodes.inc(1);
                            if *verbose {
                                progress_bar_nodes.suspend(||{
                                    println!("{}", node.to_string(&inner_inputs, &output, index.to_string()));
                                });
                            }
                            else{
                                progress_bar_nodes.suspend(||{
                                    println!("{}", format!("🚀 Executed node: {} {} for image: {}", node.get_op_type().bold(), node.get_node_name().bold(), index.to_string().bold()));
                                });
                            }
                            for (i, out) in output.iter().enumerate() {
                                inner_inputs.insert(node.get_output_names()[i].clone(), out.to_owned());
                            }
                        }

                        inner_inputs
                    }).collect::<Vec<HashMap<String, ArrayD<f32>>>>();

                    for hash in inside_outputs {
                        for (key, value) in hash {
                            inputs.insert(key, value);
                        }
                    }

                    let total_node_run = branches_len.iter().sum::<usize>();
                    node_index += total_node_run;
                }

                node_index+=1;
            }

            // Return the output for this particular input
            inputs.get(final_layer_name)
                .expect("Final layer output not found").clone()
        })
        .progress_with(progress_bar_images)
        .collect::<Vec<ArrayD<f32>>>();

    let run_time = network_timer.elapsed();
    println!("\n\n✅  The network has been successfully executed in {:?}\n", run_time);

    let shape = model_final_output[0].shape();
    let batch_size= model_final_output.len();
    let c = shape[1];
    let new_s = vec![batch_size, c ];

    let flat_vec: Vec<f32> = model_final_output.into_iter()
        .flat_map(|array| array.into_raw_vec())
        .collect();


    ArrayD::from_shape_vec(IxDyn(&new_s), flat_vec).unwrap()
}