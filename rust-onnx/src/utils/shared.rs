pub const MODELS_NAMES: &[&str;4] = &["MNIST (opset-version=12)",
                                     "ResNet-18 (v1, opset-version=7)",
                                     "ResNet-34 (v2, opset-version=7)",
                                     "MobileNet (v2, opset-version=7)"];

pub enum Model {
    MNIST,
    ResNet18v17,
    ResNet34v27,
    MobileNetv27,
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
            Model::MNIST => "mnist-12",
            Model::ResNet18v17 => "resnet18-v1-7",
            Model::ResNet34v27 => "resnet34-v2-7",
            Model::MobileNetv27 => "mobilenetv2-7",
        }
    }

    pub fn from_index(index: usize) -> Option<Model> {
        match index {
            0 => Some(Model::MNIST),
            1 => Some(Model::ResNet18v17),
            2 => Some(Model::ResNet34v27),
            3 => Some(Model::MobileNetv27),
            _ => None,
        }
    }

    pub fn get_num_classes(&self) -> usize{
        match self{
            Model::MNIST => 10,
            Model::ResNet18v17 => 1000,
            Model::ResNet34v27 => 1000,
            Model::MobileNetv27 => 1000,
        }
    }
}