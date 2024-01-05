use std::process;
use dialoguer::{theme::ColorfulTheme, Select, Input};
use std::io::{self, Write};
use std::path::Path;
use crate::utils_images::serialize_image_to_pb;

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


pub fn menu() -> (String, bool){
    print_intro();

    let options = &["Run MNIST (opset-version=12) in inference mode", "Run ResNet-18 (v1, \
        opset-version=7) in inference mode", "Run ResNet-18 (v2, opset-version=7) in inference mode",
        "Run SqueezeNet (v1.0, opset-version=8) in inference mode", "Run MobileNet (v2, opset-version=7) in inference mode",
        "Serialize an image into a .pb file", "Exit"]
        .to_vec();

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
    else if selection==options.len()-2{
        let folder_name : String= Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Provide the name of the folder (under images/) with the image to process.\n\
            The folder should not be empty and should contain an image file with the same name of the folder, in .jpg or .png format.\
            \n(type 'BACK' to go back)")
            .interact()
            .unwrap();

        if folder_name.trim().to_uppercase() == "BACK" {
            clear_screen();
            return menu();
        }

        let path_string = &("images/".to_string()+&folder_name);
        let path = Path::new(path_string);

        if path.exists() && path.is_dir(){
            let file_path_jpg = path.join(format!("{}.jpg", folder_name));
            let file_path_png = path.join(format!("{}.png", folder_name));
            let out_file_path = path.join("input.pb");
            if file_path_png.exists(){
                serialize_image_to_pb(file_path_png.to_str().unwrap(), out_file_path.to_str().unwrap());
                match Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("Done.")
                    .items(&["Go back"])
                    .default(0)
                    .interact()
                    .unwrap()
                {
                    _ => {clear_screen(); return menu();}
                };
            }
            else if file_path_jpg.exists(){
                serialize_image_to_pb(file_path_jpg.to_str().unwrap(), out_file_path.to_str().unwrap());
                match Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("Done.")
                    .items(&["Go back"])
                    .default(0)
                    .interact()
                    .unwrap()
                {
                    _ => {clear_screen(); return menu();}
                };
            }
            else{
                match Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("The folder exists but it does not include a .jpg or .png file with the same name.")
                    .items(&["Go back"])
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
                .with_prompt("There is no folder with this name under images/ (folder does not exist/the provided name is \
                actually the name of a file and not the name of a folder).")
                .items(&["Go back"])
                .default(0)
                .interact()
                .unwrap()
            {
                _ => {clear_screen(); return menu();}
            };
        }
    }

    let models = vec!["mnist-12", "resnet18-v1-7", "resnet18-v2-7", "squeezenet1.0-8", "mobilenetv2-7"];

    let models_names = vec!["MNIST (opset-version=12)", "ResNet-18 (v1, opset-version=7)",
        "ResNet-18 (v2, opset-version=7)", "SqueezeNet (v1.0, opset-version=8)", "MobileNet (v2, opset-version=7)"];

    let verbose = match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Run ".to_string() + models_names[selection] + " in verbose mode?")
        .items(&["Yes", "No", "Go back"])
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

    (models[selection].to_string(), verbose)

}