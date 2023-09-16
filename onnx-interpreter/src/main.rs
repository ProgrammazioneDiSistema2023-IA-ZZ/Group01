mod parser_code;

extern crate protobuf;

use std::collections::HashSet;
use protobuf::Message;

fn main() {
    // Load your ONNX model here
    let model = load_model();

    // Print the model or perform other actions with it
    //println!("{:?}", model);
    print_nodes(&model);
}

fn load_model() -> parser_code::onnx_ml_proto3::ModelProto {
    // Load and deserialize your .onnx file here
    let model_bytes = std::fs::read("resnet18-v1-7.onnx").expect("Failed to read .onnx file");
    let mut model = parser_code::onnx_ml_proto3::ModelProto::new(); // Create an instance of ModelProto

    // Use the parse_from_bytes method to deserialize the model
    model
        .merge_from_bytes(&model_bytes)
        .expect("Failed to parse .onnx file");

    model
}

fn print_nodes(model: &parser_code::onnx_ml_proto3::ModelProto) {
    let mut optypes = HashSet::new();
    // Access the graph using the .as_ref() method
    if let Some(graph) = model.graph.as_ref() {
        // Iterate through the nodes in the graph
        for node in &graph.node {
            println!("Node Name: {}", node.name);
            println!("Operation Type: {}", node.op_type);
            optypes.insert(&node.op_type);

            // You can access other properties of the node as needed
            // For example, to print the input and output names:
            println!("Input Names: {:?}", &node.input);
            println!("Output Names: {:?}", &node.output);
        }
    } else {
        println!("No graph found in the model.");
    }
    println!("{:?}",optypes);
}