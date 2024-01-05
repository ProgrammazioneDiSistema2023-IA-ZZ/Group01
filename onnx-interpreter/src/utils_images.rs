use std::fs;
use std::fs::File;
use ndarray::{Array, Array2, Array3, Array4, Ix, IxDyn, ArrayView2, ArrayD, Axis};

extern crate protobuf;
use protobuf::{Message};
use crate::parser_code::onnx_ml_proto3::{ModelProto, TensorProto};

use image::io::Reader as ImageReader;
use image::{GrayImage, imageops};
use std::io::Write;
use dialoguer::Error;




const MIN_SIZE: u32 = 256;
const CROP_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE_FACTOR: f32 = 255.0;
//The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
//and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. 224x224
pub fn serialize_image_to_pb(image_path: &str, pb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    let (width, height) = (img.width(), img.height());

    let (scaled_width, scaled_height) = if width > height {
        (MIN_SIZE * width / height, MIN_SIZE)
    } else {
        (MIN_SIZE, MIN_SIZE * height / width)
    };

    img = img.resize(scaled_width, scaled_height, imageops::FilterType::Gaussian);

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

pub fn serialize_imagenet_to_pb(image_path: &str, pb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    //let img_f32 = img.clone().into_bytes().iter().map(|&b| b as f32).collect();
    let (width, height) = (img.width(), img.height());

    let (scaled_width, scaled_height) = if width > height {
        (224 * width / height, 224)
    } else {
        (224, 224 * height / width)
    };

    img = img.resize(scaled_width, scaled_height, imageops::FilterType::Gaussian);

    let img_rgb = img.to_rgb8();
    let raw_data = img_rgb.into_raw();

    let mut r_color = Vec::new();
    let mut g_color = Vec::new();
    let mut b_color = Vec::new();
    println!("Raw data len: {}", raw_data.len());

    for i in 0..raw_data.len() / 3 {
        r_color.push(raw_data[3 * i]);
        g_color.push(raw_data[3 * i + 1]);
        b_color.push(raw_data[3 * i + 2]);
    }

    println!("Raw data len: {}", raw_data.len());

    let r_array: Array2<u8> =
        Array::from_shape_vec((scaled_width as usize, scaled_height as usize), r_color).unwrap();
    let g_array: Array2<u8> =
        Array::from_shape_vec((scaled_width as usize, scaled_height as usize), g_color).unwrap();
    let b_array: Array2<u8> =
        Array::from_shape_vec((scaled_width as usize, scaled_height as usize), b_color).unwrap();

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
        dims: vec![1i64, 3i64, scaled_width as i64, scaled_height as i64],
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

pub fn serialize_image (test_data_path: &String, result_test_path: &String, chosen_model: &String)->Result<(),Box<dyn std::error::Error>>{
    match fs::read_dir(&test_data_path) {
        Ok(dataset_dir) => {
            for label_dir_res in dataset_dir {
                match label_dir_res {
                    Ok(label_dir) => {

                        // Check if the file is DS_Store, if so, skip it
                        if label_dir.file_name().to_string_lossy() == ".DS_Store" {
                            continue;
                        }
                        let label: f32 = label_dir.file_name().into_string().unwrap().parse().unwrap();
                        let files_dir = fs::read_dir(label_dir.path()).unwrap();
                        for file_res in files_dir {
                            let file = file_res.unwrap();
                            if file.file_name().to_string_lossy() == ".DS_Store" {
                                continue;
                            }

                            let img_path = file.path().to_string_lossy().into_owned(); //models/mnist-8/dataset/9/99.jpg
                            let img_name = img_path.split("/").collect::<Vec<&str>>().last().unwrap().clone(); //img.jpg
                            let filename = img_name.split(".").collect::<Vec<&str>>()[0].clone(); //img

                            let img_res_path_filename = format!("{}/images/{}.pb", &result_test_path, filename);
                            //["mnist-8", "resnet18-v1-7", "resnet18-v2-7", "squeezenet1.0-8", "mobilenetv2-7"]
                            const MNIST: &str = "mnist-12";
                            const RESNET: &str = "resnet18-v1-7";

                            let num_classes = match chosen_model.as_str() {
                                MNIST => {
                                    serialize_g_image_to_pb(&img_path, &img_res_path_filename[..], img_name);
                                    10
                                },
                                RESNET => {
                                    serialize_image_to_pb(&img_path, &img_res_path_filename[..]);
                                    1000
                                },
                                _ => {
                                    serialize_imagenet_to_pb(&img_path, &img_res_path_filename[..]);
                                    1000
                                }
                            };
                            let label_res_path = format!("{}/labels/{}.pb", &result_test_path, filename);
                            serialize_label(label, label_res_path, img_name, num_classes);

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

pub fn serialize_g_image_to_pb(image_path: &str, pb_path: &str, img_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the image file
    println!("Calling the function serialize_g_image_to_pb");
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();

    img = img.resize(28, 28, imageops::FilterType::Gaussian);

    let img_gray = img.to_luma8();
    let raw_data = img_gray.into_raw();

    let gray_array: Array2<u8> =
        Array::from_shape_vec((28 as usize, 28 as usize), raw_data).unwrap();


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
        dims: vec![1i64, 1i64, 28 as i64, 28 as i64],
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
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}


pub fn serialize_label(label: f32, pb_path: String, img_name: &str, num_classes: usize) -> Result<(), Box<dyn std::error::Error>> {

    let mut label_bytes: Vec<u8> = Vec::with_capacity(num_classes * 4); // Step 2: Allocate Vec<u8>

    // Initialize a Vec<f32> with a size of num_classes, all elements set to 0.0
    let mut array: Vec<f32> = vec![0.0; num_classes];

    array[label as usize] = 1.0; // Set the value at `label` index

    for &num in &array {
        label_bytes.extend_from_slice(&num.to_le_bytes()); // Convert each f32 to 4 bytes and extend the Vec<u8>
    }


    let label_proto = TensorProto {
        dims: vec![1, num_classes as i64],
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
