use std::fs;
use std::fs::File;
use ndarray::{Array, Array2, Array3, Array4, ArrayD, Axis};

extern crate protobuf;
use protobuf::{Message};
use crate::parser_code::onnx_ml_proto3::{TensorProto};

use image::io::Reader as ImageReader;
use image::{imageops};
use std::io::Write;
use std::path::{PathBuf};

pub fn serialize_images (custom_dataset_path: &PathBuf, custom_dataset_serialized_path: &PathBuf, chosen_model: &String)->Result<(),Box<dyn std::error::Error>>{
    const MNIST: &str = "mnist-12";
    const RESNETV1: &str = "resnet18-v1-7";
    const RESNETV2: &str = "resnet18-v2-7";
    const MOBILENET: &str = "mobilenetv2-7";
    const MNIST_NUM_CLASSES : usize = 10;
    const IMAGENET_NUM_CLASSES : usize = 1000;

    if custom_dataset_serialized_path.exists() && custom_dataset_serialized_path.is_dir(){
        fs::remove_dir_all(&custom_dataset_serialized_path).unwrap();
    }
    fs::create_dir(&custom_dataset_serialized_path).unwrap();
    fs::create_dir(&custom_dataset_serialized_path.join("images")).unwrap();
    fs::create_dir(custom_dataset_serialized_path.join("labels")).unwrap();
    match fs::read_dir(custom_dataset_path) {
        Ok(dataset_dir) => {
            for label_dir_res in dataset_dir {
                match label_dir_res {
                    Ok(label_dir) => {
                        // Check if the file is DS_Store, if so, skip it
                        if label_dir.file_name().to_str().unwrap().parse::<i32>().is_err(){
                            continue;
                        }
                        let label = label_dir.file_name().into_string().unwrap().parse::<f32>().unwrap();
                        let files_dir = fs::read_dir(label_dir.path()).unwrap();
                        for file_res in files_dir {
                            let file = file_res.unwrap();
                            match file.path().extension().and_then(|ext| ext.to_str()) {
                                Some("jpg") | Some("png") | Some("jpeg")=> {},
                                _ => continue,
                            }
                            let image_path = file.path();
                            let filename = image_path.file_stem().and_then(|stem| stem.to_str()).map(|s| s.to_string()).unwrap();
                            let extension = image_path.extension().and_then(|ext| ext.to_str()).unwrap();
                            let img_name = filename.clone() + "." + extension;
                            let image_pb_path = custom_dataset_serialized_path.join("images").join(filename.clone() + ".pb");
                            //println!("{:?}", img_res_path_filename);
                            //["mnist-8", "resnet18-v1-7", "resnet18-v2-7", "squeezenet1.0-8", "mobilenetv2-7"]

                            let mut num_classes = 0;
                            match chosen_model.as_str() {
                                MNIST => {
                                    serialize_mnist_image_to_pb(&image_path, &image_pb_path, img_name.clone()).unwrap();
                                    num_classes = MNIST_NUM_CLASSES;
                                },
                                RESNETV1 | RESNETV2 | MOBILENET => {
                                    serialize_imagenet_to_pb(&image_path, &image_pb_path).unwrap();
                                    num_classes = IMAGENET_NUM_CLASSES;
                                },
                                _ => {
                                    println!("Serialization for this model is not yet supported.");
                                }
                            };
                            let label_pb_path = custom_dataset_serialized_path.join("labels").join(filename + ".pb");
                            serialize_label(label, label_pb_path, img_name, num_classes).unwrap();
                        }
                    },
                    Err(_) => {}
                }
            }
        },
        Err(_) => {}
    }
    Ok(())
}

//The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
//and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. 224x224
pub fn serialize_imagenet_to_pb(image_path: &PathBuf, pb_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    const MIN_SIZE: u32 = 256;
    const CROP_SIZE: u32 = 224;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const SCALE_FACTOR: f32 = 255.0;

    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    let (width, height) = (img.width(), img.height());

    let (scaled_width, scaled_height) = if width > height {
        (MIN_SIZE * width / height, MIN_SIZE)
    } else {
        (MIN_SIZE, MIN_SIZE * height / width)
    };

    img = img.resize(scaled_width, scaled_height, imageops::FilterType::Triangle);

    let crop_x = (scaled_width - CROP_SIZE) / 2;
    let crop_y = (scaled_height - CROP_SIZE) / 2;

    img = img.crop_imm(crop_x, crop_y, CROP_SIZE, CROP_SIZE);

    let img_rgb = img.to_rgb8();
    let raw_data = img_rgb.into_raw();

    let mut r_color = Vec::new();
    let mut g_color = Vec::new();
    let mut b_color = Vec::new();

    for i in 0..raw_data.len() / 3 {
        r_color.push(raw_data[3 * i]);
        g_color.push(raw_data[3 * i + 1]);
        b_color.push(raw_data[3 * i + 2]);
    }

    let r_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), r_color).unwrap();
    let g_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), g_color).unwrap();
    let b_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), b_color).unwrap();

    let mut rgb_arr: Array3<u8> =
        ndarray::stack(Axis(2), &[r_array.view(), g_array.view(), b_array.view()]).unwrap();
    // Transpose from HWC to CHW
    rgb_arr.swap_axes(0, 2);

    let mean = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            MEAN[0] * SCALE_FACTOR,
            MEAN[1] * SCALE_FACTOR,
            MEAN[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let std = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            STD[0] * SCALE_FACTOR,
            STD[1] * SCALE_FACTOR,
            STD[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let mut arr_float: Array3<f32> = rgb_arr.mapv(|x| x as f32);

    arr_float -= &mean;
    arr_float /= &std;

    let arr_float_batch: Array4<f32> = arr_float.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    let flat: Vec<f32> = arr_d.iter().cloned().collect(); // Step 1: Flatten the array
    let mut img_bytes: Vec<u8> = Vec::with_capacity(flat.len() * 4); // Step 2: Allocate Vec<u8>

    for &value in &flat {
        let byte_repr: [u8; 4] = value.to_le_bytes(); // Convert each f32 to 4 bytes
        img_bytes.extend_from_slice(&byte_repr); // Append bytes to Vec<u8>
    }

    let image_proto = TensorProto {
        dims: vec![1i64, 3i64, CROP_SIZE as i64, CROP_SIZE as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "data".to_string(),
        raw_data: img_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    image_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

pub fn serialize_mnist_image_to_pb(image_path: &PathBuf, pb_path: &PathBuf, img_name: String) -> Result<(), Box<dyn std::error::Error>> {
    const MNIST_SIZE : u32 = 28;
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();

    img = img.resize(MNIST_SIZE, MNIST_SIZE, imageops::FilterType::Triangle);

    let img_gray = img.to_luma8();
    let raw_data = img_gray.into_raw();

    let gray_array: Array2<u8> =
        Array::from_shape_vec((MNIST_SIZE as usize, MNIST_SIZE as usize), raw_data).unwrap();


    let mut arr_float: Array2<f32> = gray_array.mapv(|x| x as f32);

    let arr_f_im: Array3<f32> = arr_float.insert_axis(Axis(0));
    let arr_float_batch: Array4<f32> = arr_f_im.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    let flat: Vec<f32> = arr_d.iter().cloned().collect(); // Step 1: Flatten the array
    let mut img_bytes: Vec<u8> = Vec::with_capacity(flat.len() * 4); // Step 2: Allocate Vec<u8>

    for &value in &flat {
        let byte_repr: [u8; 4] = value.to_le_bytes(); // Convert each f32 to 4 bytes
        img_bytes.extend_from_slice(&byte_repr); // Append bytes to Vec<u8>
    }

    let image_proto = TensorProto {
        dims: vec![1i64, 1i64, MNIST_SIZE as i64, MNIST_SIZE as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: img_name.to_string(),
        raw_data: img_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    image_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    //println!("{:?}", pb_path);
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}


pub fn serialize_label(label: f32, pb_path: PathBuf, img_name: String, num_classes: usize) -> Result<(), Box<dyn std::error::Error>> {

    let mut label_bytes: Vec<u8> = Vec::with_capacity(num_classes * 4); // Step 2: Allocate Vec<u8>

    // Initialize a Vec<f32> with a size of num_classes, all elements set to 0.0
    let mut array: Vec<f32> = vec![0.0; num_classes];

    array[label as usize] = 1.0; // Set the value at `label` index

    for &num in &array {
        label_bytes.extend_from_slice(&num.to_le_bytes()); // Convert each f32 to 4 bytes and extend the Vec<u8>
    }


    let label_proto = TensorProto {
        dims: vec![1i64, num_classes as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: img_name.to_string(),
        raw_data: label_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    label_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}
