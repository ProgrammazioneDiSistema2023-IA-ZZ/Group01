import rust_onnx
import numpy as np
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def dataset_menu(model_name):
    while True:
        print(f"You now have two options. You can run {model_name} on:")
        print("1) the test dataset provided by ONNX, with an image and a label already serialized into .pb files;")
        print("2) your custom dataset with images and labels to be serialized into .pb files.")
        print("3) Go back to the main menu")

        choice = input("\nPlease select an option: ")

        if choice == '1':
            return True
        elif choice == '2':
            return False
        elif choice == '3':
            return None
        else:
            print("Invalid choice. Please try again.")
            continue

def menu():
    models_names = ["MNIST (opset-version=12)", "ResNet-18 (v1, opset-version=7)",
                    "ResNet-34 (v2, opset-version=7)", "MobileNet (v2, opset-version=7)"]

    options = [f"Run {name} in inference mode" for name in models_names] + ["Exit"]
    for i, option in enumerate(options, 1):
        print(f"{i}) {option}")

    selection = input("What do you want to do?\n")
    try:
        selection = int(selection)
        if selection == len(options):
            print("Exiting...")
            sys.exit(0)
    except ValueError:
        print("Please enter a valid number.")
        return menu()

    chosen_model = models_names[selection - 1]

    test_dataset = dataset_menu(chosen_model)
    if test_dataset is None:  # User chose to go back to the main menu
        clear_screen()
        return menu()


    folder_name = ""

    if not test_dataset:
        folder_name = input("Provide the folder name for your custom dataset.\n(type 'back' to go back to the main menu)\n")
        if folder_name.lower() == 'back':
            clear_screen()
            return menu()
    verbose = input(f"Run {chosen_model} in verbose mode? (yes/no)\n").lower() == 'yes'

    clear_screen()
    return chosen_model, verbose, test_dataset, folder_name, selection-1

if __name__ == '__main__':

    models_paths =[
        "./models/mnist-12/model.onnx",
        "./models/resnet18-v1-7/model.onnx",
        "./models/resnet34-v2-7/model.onnx",
        "./models/mobilenetv2-7/model.onnx"
    ]
    model, verbose_mode, is_test_dataset, folder_name, id_model = menu()
    print(f"Model: {model}")
    print(f"Verbose Mode: {verbose_mode}")
    print(f"Is Test Dataset: {is_test_dataset}")
    print(f"Selection: {id_model}")
    if is_test_dataset:
        print(f"The network {model} will be run on image provided by ONNX")
    else:
        print(f"Folder Name: {folder_name}")

    model, input_name, final_layer_name= rust_onnx.py_load_model(models_paths[id_model])

    if not is_test_dataset:
        rust_onnx.py_serialize_custom_dataset(id_model, folder_name)

    images, probs, paths = rust_onnx.py_load_data_and_labels(id_model, folder_name, is_test_dataset)

    output = rust_onnx.py_run_model (images, model, input_name, verbose_mode, final_layer_name, id_model)
    predictions = np.argmax(output, axis=1).tolist()
    groundtruth = np.argmax(probs, axis=1).tolist()
    print (f"Prediction: {predictions}")
    print(f"Groundtruth: {groundtruth}")

    rust_onnx.py_print_results(id_model, paths, output, probs)


