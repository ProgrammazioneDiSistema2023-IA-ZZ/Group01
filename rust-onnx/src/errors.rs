

#[derive(Debug)]
pub enum OnnxError{

    TensorNotFound(String),

    ShapeMismatch(String),

    BroadcastNotPossible(String),

    AxisOutOfBounds(String),

    ConversionError(String),

    OperationNotImplemented (String),

    UnsupportedDataType(String),

    DecodingNotPossible(String),

}

