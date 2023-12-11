use std::fs;
use std::env;

fn main() {

    env::set_var("RUST_BACKTRACE", "1");

    let folder_path = "src/parser_code";

    // Create a new folder
    match fs::create_dir(folder_path) {
        Ok(_) => {
            println!("Created folder: {}", folder_path);
        }
        Err(e) => {
            eprintln!("Error creating folder: {}", e);
        }
    }

    //Create the Parser;
    //crea i file .rs nella cartella parser_code
    protobuf_codegen::Codegen::new()
        // Use `protoc` parser, optional.
        .protoc()
        // Use `protoc-bin-vendored` bundled protoc command, optional.
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap())
        // All inputs and imports from the inputs must reside in `includes` directories.
        .includes(&["protos"])
        // Inputs must reside in some of include paths.
        .input("protos/onnx-ml.proto3")
        .input("protos/onnx-data.proto3")
        .input("protos/onnx-operators-ml.proto3")
        // Specify output directory relative to Cargo output directory.
        .out_dir("./src/parser_code")
        .run_from_script();
}