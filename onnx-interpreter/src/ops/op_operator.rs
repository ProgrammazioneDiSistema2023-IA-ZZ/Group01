//use super::super::errors;
use crate::errors::OnnxError;

use ndarray::ArrayD;
use std::collections::HashMap;
use std::time::Duration;
use prettytable::{format, row, Table};
use colored::Colorize;

#[derive(Clone)]
pub struct Initializer{
    name: String,
    value: ArrayD<f32>
}

impl Initializer{
    pub fn new(name: String, value: ArrayD<f32>)-> Self{
        Self{name, value}
    }
    pub fn get_name(&self)->String{
        self.name.clone()
    }
    pub fn get_value(&self)->&ArrayD<f32>{
        &self.value
    }
}

pub trait Operator {
    fn execute(&mut self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError>;
    fn to_string(&self, inputs: &HashMap<String, ArrayD<f32>>, outputs: &Vec<ArrayD<f32>>, execution_time: &Duration) -> String{
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
        table.set_titles(row![
            "Operand type".bright_red(),
            "Operand name".bright_red(),
            "Operand shape".bright_red()
        ]);
        for input in self.get_inputs(){
            let mut shape = inputs.get(&input).unwrap().shape().iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(" x ");
            shape.insert(0, '<');
            shape.push('>');
            table.add_row(row![
                "Input".bright_blue(),
                input.bright_blue(),
                shape.bright_blue()
        ]);
        }
        for initializer in self.get_initializers_arr(){
            let mut shape = initializer.get_value().shape().iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(" x ");
            shape.insert(0, '<');
            shape.push('>');
            table.add_row(row![
                "Initializer".bright_yellow(),
                initializer.get_name().bright_yellow(),
                shape.bright_yellow()
            ]);
        }
        for (i, output) in outputs.iter().enumerate(){
            let mut shape = output.shape().iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(" x ");
            shape.insert(0, '<');
            shape.push('>');
            table.add_row(row![
            "Output".bright_green(),
            self.get_output_names()[i].bright_green(),
            shape.bright_green()
        ]);
        }
        return format!("👌 Done. Node info:\n{}Execution time for the node: {:?}\n\n", table.to_string(), execution_time);
    }
    fn get_inputs(&self) -> Vec<String>;
    fn get_output_names(&self) -> Vec<String>;
    fn get_node_name(&self) -> String;
    fn get_op_type(&self) -> String;
    fn get_initializers_arr(&self) -> Vec<Initializer>;
}