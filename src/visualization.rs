use nalgebra::{Complex, ComplexField};
use nannou::{prelude::*, color};

use crate::{quantum_circuit::QuantumCircuit, quantum_gate::{QuantumGate, QuantumRegister}};

const BACKGROUND_COLOR: u32 = 0x121212;
const TEXT_COLOR: u32 = 0x03DAC5;

const BASIS_DELTA_Y: f32 = 15.0;
const COMPLEX_RECT_HEIGHT: f32 = 10.;
const BASIS_OFFSET_X: f32 = 20.0;
const LEGEND_DELTA_Y: f32 = 15.0;
const LEGEND_ENTRY_DELTA_X: f32 = 100.0;
const LEGEND_ENTRIES: usize = 8;
const PADDING: f32 = 10.0;
const GATE_TO_REGISTER_DELTA_X_PER_BASIS: f32 = 20.;
const REGISTER_TO_GATE_DELTA_X: f32 = 65.;

pub struct Model {
    window: window::Id,
    circuit: QuantumCircuit,
    input: QuantumRegister,
}

pub fn model(app: &App) -> Model {
    let window = app.new_window().view(view).build().unwrap();

    // Instantiation
    // let mut circuit = QuantumCircuit::new(2);
    // circuit.add_gate(QuantumGate::hadamard(), vec![0]);
    // circuit.add_gate(QuantumGate::global_rotation(1, TAU/4.), vec![0]);
    // circuit.add_gate(QuantumGate::hadamard(), vec![1]);
    // circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);

    let mut circuit = QuantumCircuit::fourier_transform(3);

    let input = QuantumRegister::basis(3, 1);
    
    Model { window, circuit, input }
}

pub fn update(_app: &App, _model: &mut Model, _update: Update) {}

pub fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(color::rgb_u32(BACKGROUND_COLOR));

    draw_legend(&draw, vec2(0., app.window_rect().top()));
    
    let mut xy = Point2::new(app.window_rect().left() + COMPLEX_RECT_HEIGHT / 2.0 + PADDING, 0.);
    draw_register(&draw, &model.input, xy);

    let mut result = model.input.clone();
    for gate in model.circuit.get_gates() {
        xy = xy + vec2(REGISTER_TO_GATE_DELTA_X, 0.);
        draw_gate(&draw, &gate.clone(), xy);
        xy = xy + vec2(GATE_TO_REGISTER_DELTA_X_PER_BASIS * (2usize.pow(gate.n_qubits() as u32) as f32), 0.);
        result = gate.apply(result);
        draw_register(&draw, &result, xy);
        
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_legend(draw: &Draw, xy: Vec2) {
    let start_x = xy.x - ((LEGEND_ENTRIES as f32) * LEGEND_ENTRY_DELTA_X) / 2.;
    for i in 0..LEGEND_ENTRIES {
        let z = Complex::exp(Complex::i() * TAU * i as f32 / LEGEND_ENTRIES as f32);
        let legend_xy = vec2(start_x + (i as f32) * LEGEND_ENTRY_DELTA_X, xy.y - PADDING);
        draw.text(&pretty_print_complex(z))
            .xy(legend_xy)
            .color(color::rgb_u32(TEXT_COLOR));
        draw_coefficient(draw, z, legend_xy + vec2(0., -LEGEND_DELTA_Y));
    }
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

fn draw_gate(draw: &Draw, gate: &QuantumGate, xy: Vec2) {
    let x = xy.x;
    let y = xy.y;

    let n_bases = 2usize.pow(gate.n_qubits() as u32);
    
    let bounding_size = (BASIS_DELTA_Y * (n_bases as f32));
    let bounding_xy = xy + vec2(bounding_size / 2. - COMPLEX_RECT_HEIGHT / 2., -bounding_size / 2. + COMPLEX_RECT_HEIGHT / 2.);
    draw.rect()
    .w_h(bounding_size + 0.5 * PADDING, bounding_size + 0.5 * PADDING)
    .xy(bounding_xy)
    .color(GRAY);
    draw.rect()
    .w_h(bounding_size, bounding_size)
    .xy(bounding_xy)
    .color(color::rgb_u32(BACKGROUND_COLOR));
    for row in 0..n_bases {
        for col in 0..n_bases {
            let coefficient = gate.get_coefficient(row, col);
            draw_coefficient(draw, coefficient, vec2(x + col as f32 * BASIS_DELTA_Y, y - row as f32 * BASIS_DELTA_Y));
        }
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

fn pretty_print_complex(z: Complex<f32>) -> String {
    if z.abs() < 0.0001 {
        return "0".to_string();
    }

    let mut real_result = String::new();
    if z.re.abs() > 0.0001 {
        real_result = pretty_print_real_number(z.re);
    }

    let mut imaginary_result = String::new();
    if z.im.abs() > 0.0001 {
        if (z.im.abs() - 1.0).abs() < 0.0001 {
            imaginary_result = "i".to_string();
        } else {
            imaginary_result = format!("{}i", pretty_print_real_number(z.im.abs()));
        }
    }

    let result = if real_result == "" {
        if z.im < 0.0 {
            format!("-{}", imaginary_result)
        } else {
            imaginary_result
        }
    } else if imaginary_result == "" {
        real_result
    } else {
        if z.im < 0.0 {
            format!("{} - {}", real_result, imaginary_result)
        } else {
            format!("{} + {}", real_result, imaginary_result)
        }
    };
    return result;
}

fn pretty_print_real_number(x: f32) -> String {
    let x = (x * 1000.).round() / 1000.;
    return format!("{}", x);
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

    #[test]
    fn test_pretty_print_complex() {

        assert_eq!(pretty_print_complex(Complex::new(0., 0.)), "0");
        assert_eq!(pretty_print_complex(Complex::new(1., 0.)), "1");
        assert_eq!(pretty_print_complex(Complex::new(0., 1.)), "i");
        assert_eq!(pretty_print_complex(Complex::new(0., -1.)), "-i");
        assert_eq!(pretty_print_complex(Complex::new(1., 1.)), "1 + i");
        assert_eq!(pretty_print_complex(Complex::new(1., -1.)), "1 - i");
        assert_eq!(pretty_print_complex(Complex::new(-1.2, 1.)), "-1.2 + i");
        assert_eq!(pretty_print_complex(Complex::new(-1.2, -3.4)), "-1.2 - 3.4i");
        assert_eq!(pretty_print_complex(Complex::exp(Complex::i()/8.*TAU)), "0.707 + 0.707i");
        
    }
}