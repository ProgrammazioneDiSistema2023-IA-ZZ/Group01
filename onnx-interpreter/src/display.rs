use std::{fs, process};
use dialoguer::{theme::ColorfulTheme, Select, Input};
use std::io::{self, Write};
use std::path::{PathBuf};
use crate::utils_images::serialize_imagenet_to_pb;

fn print_intro() {
    let program_name_and_description = "
  _____           _    \x1b[31m______\x1b[0m  _   _ _   ___   __     _____       _   _          _____        __            _                           _ _ _
 |  __ \\         | |  \x1b[31m/\x1b[0m  \x1b[30m__\x1b[0m  \x1b[31m\\\x1b[0m| \\ | | \\ | \\ \\ / /_   / ____|     | | | |        |_   _|      / _|          ( )                    /\\   | | | |
 | |__) |   _ ___| |_\x1b[31m|\x1b[0m\x1b[30m__/\x1b[0m__\x1b[30m\\__\x1b[0m\x1b[31m|\x1b[0m  \\| |  \\| |\\ V /(_) | |  __  ___ | |_| |_ __ _    | |  _ __ | |_ ___ _ __  |/  ___ _ __ ___      /  \\  | | | |
 |  _  / | | / __| __\x1b[30m|__\x1b[0m|__|\x1b[30m__|\x1b[0m . ` | . ` | > <     | | |_ |/ _ \\| __| __/ _` |   | | | '_ \\|  _/ _ \\ '__|    / _ \\ '_ ` _ \\    / /\\ \\ | | | |
 | | \\ \\ |_| \\__ \\ |_|  \x1b[30m\\__/\x1b[0m  | |\\  | |\\  |/ . \\ _  | |__| | (_) | |_| || (_| |  _| |_| | | | ||  __/ |      |  __/ | | | | |  / ____ \\| | |_|
 |_|  \\_\\__,_|___/\\__|\\______/|_| \\_|_| \\_/_/ \\_(_)  \\_____|\\___/ \\__|\\__\\__,_| |_____|_| |_|_| \\___|_|       \\___|_| |_| |_| /_/    \\_\\_|_(_)
 ----------------------------------------------------------------------------------------------------------------------------------------------
    - ⚙️ Rust-based ONNX inference engine.
 ----------------------------------------------------------------------------------------------------------------------------------------------\n";

    println!(
        "{}",
        program_name_and_description
    );
}

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    // Flush stdout to ensure the escape codes are written to the screen
    io::stdout().flush().unwrap();
}


pub fn menu() -> (String, bool, bool, String, Option<usize>){
    print_intro();

    let models_names = vec!["MNIST (opset-version=12)", "ResNet-18 (v1, opset-version=7)",
                            "ResNet-18 (v2, opset-version=7)", "MobileNet (v2, opset-version=7)"];

    let models = vec!["mnist-12", "resnet18-v1-7", "resnet18-v2-7", "mobilenetv2-7"];

    let mut options = Vec::new();
    for model_name in &models_names{
        options.push("Run ".to_string() + model_name + " in inference mode");
    }
    options.push("Exit".to_string());

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("What do you want to do?")
        .items(&options)
        .default(0)
        .interact()
        .unwrap();

    if selection == options.len()-1{
        println!("Exiting...");
        process::exit(0);
    }

    println!();

    let test_dataset = match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("You now have two options. You can run ".to_string() + models_names[selection] + " on:\n1) \
        the test dataset provided by ONNX, with an image and a label already serialized into .pb files;\n\
        2) your custom dataset with images and labels to be serialized into .pb files.\n\
        Please select an option.")
        .items(&["Run on the test dataset", "Run on my custom dataset", "Go back to the main menu"])
        .default(0)
        .interact()
        .unwrap()
    {
        0 => true,
        1 => false,
        2 => {
            clear_screen();
            return menu();
        }
        _ => false,
    };

    println!();

    let mut folder_name = String::new();
    let mut batch_size : Option<usize> = None;
    let mut verbose = false;

    match test_dataset{
        true => {
            verbose = match Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Run ".to_string() + models_names[selection] + " in verbose mode?")
                .items(&["Yes", "No", "Go back to the main menu"])
                .default(0)
                .interact()
                .unwrap()
            {
                0 => true,
                1 => false,
                2 => {
                    clear_screen();
                    return menu();
                }
                _ => false,
            };
        },
        false =>{
            folder_name = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Provide the folder name for your custom dataset.\n\
                Be sure to respect the following constraints:\n\
                1) the dataset folder must be placed under the folder of the model you want to run (e.g. \"mnist-12/my-dataset/\");\n\
                2) the dataset folder must include subfolders whose names match the label, in numeric format, of the images they contain;\n\
                3) at least one subfolder that follows the naming convention mentioned above must reside within the dataset folder;\n\
                4) accepted image formats are .jpg or .png;\n\
                5) all subfolders following the expected naming convention must include at least a .jpg or .png file.\n\
                For example, you may have a \"my-dataset/\" folder under \"resnet18-v2-7/\" with a \"207/\" subfolder that includes \
                a .jpg or .png image of a golden retriever, since 207 is the label for a golden retriever in the ImageNet dataset.\
            \n(type 'BACK' to go back to the main menu)")
                .interact()
                .unwrap();
            if folder_name.trim().to_uppercase() == "BACK" {
                clear_screen();
                return menu();
            }
            println!();

            let model_path = PathBuf::from("models").join(models[selection]);
            let dataset_path = model_path.join(format!("{}", folder_name));

            if dataset_path.exists() && dataset_path.is_dir(){
                let mut subfolders_with_numerical_name_counter = 0;
                let mut subfolders_with_no_jpg_or_png_images_counter = 0;
                for entry in fs::read_dir(dataset_path.clone()).unwrap(){
                    let path = entry.unwrap().path();
                    if path.is_dir(){
                        let dir_name = path.file_name().unwrap().to_str().unwrap();
                        if !dir_name.parse::<i32>().is_err(){
                            subfolders_with_numerical_name_counter+=1;
                            let mut jpg_png_flag = false;
                            for sub_entry in fs::read_dir(path).unwrap(){
                                let sub_path = sub_entry.unwrap().path();
                                match sub_path.extension().and_then(|ext| ext.to_str()) {
                                    Some("jpg") | Some("png") => jpg_png_flag=true,
                                    _ => {},
                                }
                            }
                            if jpg_png_flag==false{
                                subfolders_with_no_jpg_or_png_images_counter+=1;
                            }
                        }
                    }
                }
                if subfolders_with_numerical_name_counter==0{
                    match Select::with_theme(&ColorfulTheme::default())
                        .with_prompt("The folder \"".to_string() +  dataset_path.to_str().unwrap() + "\\\" exists \
                        but does not include any subfolder respecting the numerical naming convention.")
                        .items(&["Go back to the main menu"])
                        .default(0)
                        .interact()
                        .unwrap()
                    {
                        _ => {clear_screen(); return menu();}
                    };
                }
                if subfolders_with_no_jpg_or_png_images_counter!=0{
                    match Select::with_theme(&ColorfulTheme::default())
                        .with_prompt("Between the subfolders respecting the numerical naming convention, there is \
                        at least a subfolder that does not include any .jpg or .png file.")
                        .items(&["Go back to the main menu"])
                        .default(0)
                        .interact()
                        .unwrap()
                    {
                        _ => {clear_screen(); return menu();}
                    };
                }

            }
            else{
                match Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("There is no folder with this name under \"".to_string() +  model_path.to_str().unwrap() + "\\\".")
                    .items(&["Go back to the main menu"])
                    .default(0)
                    .interact()
                    .unwrap()
                {
                    _ => {clear_screen(); return menu();}
                };
            }

            batch_size = match Select::with_theme(&ColorfulTheme::default())
                .with_prompt("The model ". to_string() + models_names[selection] + " can run on images distributed over multiple batches and \
                benefit from parallelization across batches.\nPlease, select a value for the batch size.")
                .items(&["1", "2", "4", "Go back to the main menu"])
                .default(0)
                .interact()
                .unwrap()
            {
                0 => Some(1usize),
                1 => Some(2usize),
                2 => Some(4usize),
                3 => {
                clear_screen();
                return menu();
            }
                _ => None,
            };
            println!();
            verbose = match Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Run ".to_string() + models_names[selection] + " in verbose mode?")
                .items(&["Yes", "No", "Go back to the main menu"])
                .default(0)
                .interact()
                .unwrap()
            {
                0 => true,
                1 => false,
                2 => {
                    clear_screen();
                    return menu();
                }
                _ => false,
            };
        }
    }

    (models[selection].to_string(), verbose, test_dataset, folder_name, batch_size)

}