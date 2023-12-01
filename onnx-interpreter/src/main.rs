
mod parser_code;

extern crate protobuf;

use std::collections::{HashSet, HashMap};

mod lib;
use lib::{Operator, Add, MatMul, AutoPad };

mod auxiliary_functions;

use auxiliary_functions::{topological_sort, load_data, read_initialiazers, load_model,
                          model_proto_to_struct, print_nodes, argmax, load_predictions,
                          argmax_per_row, compute_error_rate, compute_accuracy };
use ndarray::ArrayD;


fn main() {

    //let data_path= "input_0_resnet18.pb";
    //let output_path = "output_0_resnet18.pb";
    let data_path= "input_0_mnist.pb";
    let output_path = "output_0_mnist.pb";

    //let model_path = "resnet18-v1-7.onnx";
    let model_path = "mnist-8.onnx";

    // Load ONNX model
    let model = load_model(model_path);

    print_nodes(&model);
    //println!("{:?}", model.graph.initializer);

    let mut inputs: HashMap<String, ArrayD<f32>> = HashMap::new();
    read_initialiazers(&model.graph.initializer, &mut inputs);

    let mut model_read = model_proto_to_struct(&model, &inputs);
    for node in &model_read{
        println!("{}", node.to_string());
        println!();
    }
    println!("In totale ci sono {} nodi", model_read.len());

    let (input_image, input_name) = load_data(data_path).unwrap();
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
            println!("{:?}", node.to_string());
            let output = node.execute(&inputs).unwrap();
            inputs.insert(node.get_output_name(), output.clone());
            println!("Output shape: {:?}", output.shape());
    }

    let final_output = inputs.get(final_layer_name).unwrap();
    println!("Final output: {:?}", final_output);

    let final_result = argmax_per_row(final_output);
    println!("Final result: {:?}", &final_result);

    let predictions = load_predictions(output_path).unwrap();
    let final_predictions = argmax_per_row(&predictions);
    println!("Predictions: {:?}", &final_predictions);
    /*
            let error_rate = compute_error_rate(&final_result, &final_predictions).unwrap();
            println!("Error rate: {}", error_rate);

            let accuracy = compute_accuracy(&final_result, &final_predictions).unwrap();
            println!("Accuracy: {}", accuracy);

             */
}



