#![feature(int_log)]

mod matrix;
mod qubit;
mod quantum_gate;
mod quantum_circuit;
mod quantum_algorithm;
mod visualization;

use nannou::prelude::*;
use crate::visualization::{Model, model, update, view};

fn main() {
    nannou::app(model).update(update).run();
}