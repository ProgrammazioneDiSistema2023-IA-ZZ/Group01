use std::process;
use dialoguer::{theme::ColorfulTheme, Select};
use std::io::{self, Write};

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

    let options = &["Run MNIST (opset-version=8) in inference mode", "Run ResNet-18 (v1, \
        opset-version=7) in inference mode", "Run ResNet-18 (v2, opset-version=7) in inference mode",
        "Run ResNet-152 (v2, opset-version=7) in inference mode", "Exit"]
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

    let models = vec!["mnist-8", "resnet18-v1-7", "resnet18-v2-7", "resnet152-v2-7"];

    let models_names = vec!["MNIST (opset-version=8)", "ResNet-18 (v1, opset-version=7)",
        "ResNet-18 (v2, opset-version=7)", "ResNet-152 (v2, opset-version=7)"];

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