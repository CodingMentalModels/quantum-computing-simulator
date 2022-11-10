use crate::qubit::Qubit;
use crate::quantum_gate::QuantumGate;
use crate::quantum_gate::QuantumRegister;

use std::f32::consts::{TAU, SQRT_2};

use nalgebra::{Complex, ComplexField};

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumCircuit {
    n_qubits: usize,
    gates: Vec<QuantumGate>,
}

impl QuantumCircuit {

    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits, gates: Vec::new() }
    }

    pub fn add_gate(&mut self, gate: QuantumGate, initial_qubit_index: usize) {
        assert!(initial_qubit_index + gate.n_qubits() <= self.n_qubits);

        let left_identity_size = initial_qubit_index;
        let right_identity_size = self.n_qubits - initial_qubit_index - gate.n_qubits();
        match (left_identity_size, right_identity_size) {
            (0, 0) => self.gates.push(gate),
            (0, _) => self.gates.push(gate.tensor_product(QuantumGate::identity(right_identity_size))),
            (_, 0) => self.gates.push(QuantumGate::identity(left_identity_size).tensor_product(gate)),
            (_, _) => self.gates.push(QuantumGate::identity(left_identity_size).tensor_product(gate).tensor_product(QuantumGate::identity(right_identity_size))),
        };
    }

    pub fn run(&self, register: impl Into<QuantumRegister>) -> QuantumRegister {
        let mut intermediate_register = register.into().clone();
        for gate in &self.gates {
            intermediate_register = gate.apply(intermediate_register);
        }
        return intermediate_register;
    }

    pub fn reverse(&self) -> Self {
        let mut reversed_gates = self.gates.iter().map(|gate| gate.clone().reverse()).collect::<Vec<_>>();
        reversed_gates.reverse();
        Self { n_qubits: self.n_qubits, gates: reversed_gates }
    }

}


#[cfg(test)]
mod test_quantum_circuit {

    use std::f32::consts::SQRT_2;

    use nalgebra::Complex;
    use num_traits::{Zero, One};

    use super::*;

    #[test]
    fn test_identity_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::identity(1), 0);

        assert!(circuit.run(Qubit::basis_0()).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(Qubit::basis_1()).almost_equals(Qubit::basis_1()));
        assert!(circuit.run(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)).almost_equals(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)));

    }

    #[test]
    fn test_serial_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::hadamard(), 0);
        circuit.add_gate(QuantumGate::not(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 0);

        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_0())).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_1())).almost_equals(Qubit::from_coefficients_normalize(Complex::zero(), -Complex::one())));
        assert!(circuit.run(QuantumRegister::singleton((Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 1.)))).almost_equals(Qubit::from_coefficients_normalize(Complex::one()/SQRT_2, -Complex::one()/SQRT_2)));
    }

    #[test]
    fn test_two_qubit_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::cnot(), 0);

        let zero_zero = QuantumRegister::basis(2, 0);
        let zero_one = QuantumRegister::basis(2, 1);
        let one_zero = QuantumRegister::basis(2, 2);
        let one_one = QuantumRegister::basis(2, 3);

        assert!(circuit.run(zero_zero.clone()).almost_equals(zero_zero.clone()));
        assert!(circuit.run(zero_one.clone()).almost_equals(zero_one.clone()));
        assert!(circuit.run(one_zero.clone()).almost_equals(one_one.clone()));
        assert!(circuit.run(one_one.clone()).almost_equals(one_zero.clone()));
        
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::cnot(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 1);

        let expected_zero_zero = QuantumRegister::from_vec_normalize(vec![Complex::one(), Complex::one(), Complex::one(), Complex::one()]);
        let expected_zero_one = QuantumRegister::from_vec_normalize(vec![Complex::one(), -Complex::one(), Complex::one(), -Complex::one()]);
        let expected_one_zero = QuantumRegister::from_vec_normalize(vec![Complex::one(), -Complex::one(), -Complex::one(), Complex::one()]);
        let expected_one_one = QuantumRegister::from_vec_normalize(vec![Complex::one(), Complex::one(), -Complex::one(), -Complex::one()]);


        assert!(circuit.run(zero_zero.clone()).almost_equals(expected_zero_zero), "{:?}", circuit.run(zero_zero.clone()));
        assert!(circuit.run(zero_one.clone()).almost_equals(expected_zero_one), "{:?}", circuit.run(zero_one.clone()));
        assert!(circuit.run(one_zero.clone()).almost_equals(expected_one_zero), "{:?}", circuit.run(one_zero.clone()));
        assert!(circuit.run(one_one.clone()).almost_equals(expected_one_one), "{:?}", circuit.run(one_one.clone()));
    }

    #[test]
    fn test_gate_expands() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::identity(2), 0);
        assert!(circuit.run(QuantumRegister::all_1s(2)).almost_equals(QuantumRegister::all_1s(2)));

        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::hadamard(), 0);
        // |00> -> 1/sqrt(2) * (|00> + |10>)
        // |01> -> 1/sqrt(2) * (|01> + |11>)
        // |10> -> 1/sqrt(2) * (|00> - |10>)
        // |11> -> 1/sqrt(2) * (|01> - |11>)

        // [1/sqrt(2), 0, 1/sqrt(2), 0]
        // [0, 1/sqrt(2), 0, 1/sqrt(2)]
        // [1/sqrt(2), 0, -1/sqrt(2), 0]
        // [0, 1/sqrt(2), 0, -1/sqrt(2)]

        let expected = QuantumRegister::from_vec_normalize(
            vec![Complex::zero(), Complex::one()/SQRT_2, Complex::zero(), -Complex::one()/SQRT_2]
        );
        assert!(circuit.run(QuantumRegister::all_1s(2)).almost_equals(expected));
        
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::hadamard(), 1);
        // |00> -> 1/sqrt(2) * (|00> + |01>)
        // |01> -> 1/sqrt(2) * (|00> - |01>)
        // |10> -> 1/sqrt(2) * (|10> + |11>)
        // |11> -> 1/sqrt(2) * (|10> - |11>)

        // [1/sqrt(2), 1/sqrt(2), 0, 0]
        // [1/sqrt(2), -1/sqrt(2), 0, 0]
        // [0, 0, 1/sqrt(2), 1/sqrt(2)]
        // [0, 0, 1/sqrt(2), -1/sqrt(2)]

        let expected = QuantumRegister::from_vec_normalize(
            vec![Complex::zero(), Complex::zero(), Complex::one()/SQRT_2, -Complex::one()/SQRT_2]
        );
        assert!(circuit.run(QuantumRegister::all_1s(2)).almost_equals(expected));
        
    }

    #[test]
    fn test_trivial_quantum_circuit() {

        let circuit = QuantumCircuit::new(1);
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_0())).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_1())).almost_equals(Qubit::basis_1()));
        
    }

    #[test]
    fn test_quantum_circuit_reverses() {

        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::hadamard(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 1);
        circuit.add_gate(QuantumGate::cnot(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 0);
        circuit.add_gate(QuantumGate::hadamard(), 1);

        let mut reversed = circuit.reverse();

        let mut expected = QuantumCircuit::new(2);
        expected.add_gate(QuantumGate::hadamard(), 1);
        expected.add_gate(QuantumGate::hadamard(), 0);
        expected.add_gate(QuantumGate::cnot(), 0);
        expected.add_gate(QuantumGate::hadamard(), 1);
        expected.add_gate(QuantumGate::hadamard(), 0);

        for basis in QuantumRegister::all_bases(2) {
            assert!(reversed.run(basis.clone()).almost_equals(expected.run(basis)));
        }
    }


    #[test]
    fn test_measurement_interference() {

        // Circuit with measurement in the middle:
        // |0> -> H -> Measure -> H => |0> with 50% probability, |1> with 50% probability
        
        let hadamard_gate = QuantumGate::hadamard();
        let mut hadamard_circuit = QuantumCircuit::new(1);
        hadamard_circuit.add_gate(hadamard_gate.clone(), 0);
        
        let register = QuantumRegister::from(Qubit::basis_0());
        
        let hadamarded_0 = hadamard_circuit.run(register.clone());
        
        let mut count_0s = 0;
        let mut count_1s = 1;
        
        for _ in 0..1000 {
            let (measurement, after_measurement) = hadamarded_0.measure();

            assert!(after_measurement.almost_equals(Qubit::basis_0()) || after_measurement.almost_equals(Qubit::basis_1()));

            let (second_measurement, after_second_measurement) = hadamard_circuit.run(after_measurement).measure();

            assert!(after_second_measurement.almost_equals(Qubit::basis_0()) || after_second_measurement.almost_equals(Qubit::basis_1()));
            if second_measurement == 0 {
                count_0s += 1;
            } else {
                count_1s += 1;
            }
        }

        assert!(count_0s > 400); // Should be 50% = 500, so should be safe
        assert!(count_1s > 400); // Should be 50% = 500, so should be safe

        // Circuit without measurement in the middle:
        // |0> -> H -> H => |0> with 100% probability, |1> with 0% probability

        let mut full_circuit = hadamard_circuit.clone();
        full_circuit.add_gate(hadamard_gate, 0);
        let (measurement, after_measurement) = full_circuit.run(register).measure();

        assert_eq!(measurement, 0);
        assert!(after_measurement.almost_equals(Qubit::basis_0()));
        

    }

    #[test]
    fn test_fourier_and_inverse_fourier_are_inverses() {
        
        let ft = QuantumCircuit::fourier_transform(4);
        let ift = QuantumCircuit::inverse_fourier_transform(4);

        for i in 0..15 {
            let input = QuantumRegister::from_int(i, 4);
            let output = ft.run(input.clone());
            let output2 = ift.run(output);
            assert!(output2.almost_equals(input));
        }
    }

    #[test]
    fn test_inverse_fourier_transform() {

        // omega = 0.101
        // => x1 = 1
        // => x2 = 0
        // => x3 = 1
        // => 0.x3 = 0.1 = 1/2
        // => 0.x2x3 = 0.01 = 1/4
        // => 0.x1x2x3 = 0.101 = 5/8
        let omega = 5.0 * TAU / 8.0;
        
        let ift = QuantumCircuit::inverse_fourier_transform(3);

        let mut rotated_bases = Vec::new();
        for idx in 0..8 {
            let basis = QuantumRegister::basis(3, idx);
            let rotated = basis.rotate((idx as f32) * omega);
            rotated_bases.push(rotated.clone());
        };

        let input = QuantumRegister::mixture(rotated_bases);

        assert!(ift.run(input.clone()).almost_equals(QuantumRegister::from_int(3, 5)));
    }

}