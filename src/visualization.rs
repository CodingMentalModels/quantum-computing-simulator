use nalgebra::{Complex, ComplexField};
use nannou::{prelude::*, color};

use crate::{quantum_circuit::QuantumCircuit, quantum_gate::{QuantumGate, QuantumRegister}};

const BACKGROUND_COLOR: u32 = 0x121212;
const TEXT_COLOR: u32 = 0x03DAC5;

const BASIS_DELTA_Y: f32 = 15.0;
const COMPLEX_RECT_HEIGHT: f32 = 10.;
const BASIS_OFFSET_X: f32 = 20.0;
const PADDING: f32 = 10.0;
const GATE_TO_REGISTER_DELTA_X: f32 = 50.0;
const REGISTER_TO_GATE_DELTA_X: f32 = 50.0;

pub struct Model {
    window: window::Id,
    circuit: QuantumCircuit,
    input: QuantumRegister,
}

pub fn model(app: &App) -> Model {
    let window = app.new_window().view(view).build().unwrap();

    // Instantiation
    let mut circuit = QuantumCircuit::new(2);
    circuit.add_gate(QuantumGate::hadamard(), vec![0]);
    circuit.add_gate(QuantumGate::global_rotation(1, TAU/4.), vec![0]);
    circuit.add_gate(QuantumGate::hadamard(), vec![1]);
    circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);

    let input = QuantumRegister::basis(2, 0);
    
    Model { window, circuit, input }
}

pub fn update(_app: &App, _model: &mut Model, _update: Update) {}

pub fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(color::rgb_u32(BACKGROUND_COLOR));
    
    let mut xy = Point2::new(app.window_rect().left() + COMPLEX_RECT_HEIGHT / 2.0 + PADDING, 0.);
    draw_register(&draw, &model.input, xy);

    let mut result = model.input.clone();
    for gate in model.circuit.get_gates() {
        xy = xy + vec2(REGISTER_TO_GATE_DELTA_X, 0.);
        // draw_gate(&draw, gate, xy);
        xy = xy + vec2(GATE_TO_REGISTER_DELTA_X, 0.);
        result = gate.apply(result);
        draw_register(&draw, &result, xy);
        
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_register(draw: &Draw, register: &QuantumRegister, xy: Vec2) {
    let x = xy.x;
    let mut y = xy.y;
    for i in 0..register.len() {
        let coefficient = register.get_coefficient(i);
        draw_basis(draw, register.n_qubits(), i, coefficient, vec2(x, y));
        y -= BASIS_DELTA_Y;
    }
}

fn draw_basis(draw: &Draw, n_qubits: usize, index: usize, coefficient: Complex<f32>, xy: Vec2) {
    let x = xy.x;
    let y = xy.y;
    draw.text(&format!("|{}>", to_binary_string(n_qubits, index)))
        .x_y(x + BASIS_OFFSET_X, y)
        .color(color::rgb_u32(TEXT_COLOR));
    draw_coefficient(draw, coefficient, vec2(x, y));
}

fn draw_coefficient(draw: &Draw, coefficient: Complex<f32>, xy: Vec2) {
    let x = xy.x;
    let y = xy.y;
    let height = coefficient.abs() * COMPLEX_RECT_HEIGHT;
    let argument = coefficient.argument();
    let hue = argument / std::f32::consts::TAU;
    let color = color::hsv(hue, 1.0, 1.0);
    // draw.rect()
    //     .x_y(x, y)
    //     .w_h(COMPLEX_RECT_HEIGHT, COMPLEX_RECT_HEIGHT)
    //     .color(WHITE);
    draw.rect()
        .x_y(x, y)
        .w_h(COMPLEX_RECT_HEIGHT, height)
        .color(color);
    // draw.text(&format!("{:.2}", coefficient))
    //     .x_y(x, y)
    //     .color(color::rgb_u32(TEXT_COLOR));    
}

fn to_binary_string(n_qubits: usize, index: usize) -> String {
    assert!(index < 2usize.pow(n_qubits as u32));
    let mut binary_string = format!("{:b}", index);
    while binary_string.len() < n_qubits {
        binary_string = format!("0{}", binary_string);
    }
    binary_string
}


#[cfg(test)]
mod test_visualization {
    use super::*;

    #[test]
    fn test_to_binary_string() {
        
        assert_eq!(to_binary_string(1, 0), "0");
        assert_eq!(to_binary_string(1, 1), "1");
        assert_eq!(to_binary_string(2, 0), "00");
        assert_eq!(to_binary_string(2, 1), "01");
        assert_eq!(to_binary_string(2, 2), "10");
        assert_eq!(to_binary_string(2, 3), "11");
    }
}