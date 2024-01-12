mod onnx_parser;
mod models;
mod operators;
mod utils;
mod datasets;

extern crate protobuf;
use std::env;
use crate::utils::auxiliary_functions::{argmax_per_row, compute_error_rate, compute_accuracy, display_model_info, print_results, find_dependencies};
use crate::utils::run::run;
use crate::onnx_parser::parser::{load_images_and_labels, load_model};
use crate::utils::display::menu;
use crate::utils::serialization_utils::serialize_custom_dataset;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (chosen_model, verbose, test_dataset, folder_name, model_path) = menu();

    let (mut model_read, input_name, final_layer_name) = load_model(&model_path);

    if !test_dataset{
        serialize_custom_dataset(&chosen_model, &folder_name);
    }

    let (images_vec, label_stack, file_paths) = load_images_and_labels(&chosen_model, &folder_name, &test_dataset);

    let dependencies = find_dependencies(&mut model_read);

    display_model_info(chosen_model.as_str(), model_read.len());

    let final_output = run(&images_vec, &model_read, &input_name, &dependencies, &verbose, &final_layer_name);

    let predictions = argmax_per_row(&final_output);

    let ground_truth_labels = argmax_per_row(&label_stack);

    let error_rate = compute_error_rate(&predictions, &ground_truth_labels).unwrap();

    let accuracy = compute_accuracy(&predictions, &ground_truth_labels).unwrap();

    print_results(chosen_model, &file_paths, &predictions, &ground_truth_labels, &error_rate, &accuracy);

}