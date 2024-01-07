use std::fs;
use std::env;
use std::path::PathBuf;

fn main() {

    env::set_var("RUST_BACKTRACE", "1");

    let folder_path = PathBuf::from("src").join("parser_code");

    // Create a new folder
    match fs::create_dir(&folder_path) {
        Ok(_) => {
            println!("Created folder: {}", &folder_path.to_str().unwrap());
        }
        Err(e) => {
            eprintln!("Error creating folder: {}", e);
        }
    }

    let onnx_ml_proto3_path = PathBuf::from("protos").join("onnx-ml.proto3");
    let onnx_data_proto3_path = PathBuf::from("protos").join("onnx-data.proto3");
    let onnx_operators_ml_proto3_path = PathBuf::from("protos").join("onnx-operators-ml.proto3");

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
        .input(onnx_ml_proto3_path)
        .input(onnx_data_proto3_path)
        .input(onnx_operators_ml_proto3_path)
        // Specify output directory relative to Cargo output directory.
        .out_dir(folder_path)
        .run_from_script();
}