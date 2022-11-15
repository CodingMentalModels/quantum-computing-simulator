use crate::qubit::Qubit;
use crate::quantum_gate::QuantumGate;
use crate::quantum_gate::QuantumRegister;

use std::f32::consts::{TAU, SQRT_2};
use std::fmt::Display;
use std::fmt::Formatter;

use nalgebra::{Complex, ComplexField};

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumCircuit {
    n_qubits: usize,
    gates: Vec<QuantumGate>,
}

impl Display for QuantumCircuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = self.gates.iter().map(|gate| gate.to_string()).collect::<Vec<String>>().join("\n");
        write!(f, "Circuit: \n\n{}", s)
    }
}


impl QuantumCircuit {

    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits, gates: Vec::new() }
    }

    pub fn add_gate(&mut self, gate: QuantumGate, input_qubits: Vec<usize>) {
        assert!(gate.n_qubits() > 0);
        assert!(gate.n_qubits() == input_qubits.len());

        let mut sorted_qubits = input_qubits.clone();
        sorted_qubits.sort();
        assert!(sorted_qubits.iter().zip(sorted_qubits.iter().skip(1)).all(|(a, b)| a != b));

        let mut next = 0;
        let mut completed_qubits = input_qubits.clone();
        while completed_qubits.len() < self.n_qubits {
            if completed_qubits.contains(&next) {
                next += 1;
            } else {
                completed_qubits.push(next);
                next += 1;
            }
        }
        
        let input_qubits_to_first_n = QuantumGate::permutation(completed_qubits);
        let first_n_to_input_qubits = input_qubits_to_first_n.reverse();
        let gate_to_add = if input_qubits.len() == self.n_qubits {
            gate
        } else {
            let identity_gate = QuantumGate::identity(self.n_qubits - input_qubits.len());
            identity_gate.clone().tensor_product(gate)
        };

        let gate_to_add = gate_to_add.compose(&input_qubits_to_first_n);
        let gate_to_add = first_n_to_input_qubits.compose(&gate_to_add);
        
        self.gates.push(gate_to_add);

    }

    pub fn run(&self, register: impl Into<QuantumRegister>) -> QuantumRegister {
        let inner_register = register.into();
        assert_eq!(self.n_qubits, inner_register.n_qubits());
        let mut intermediate_register = inner_register.clone();
        for gate in &self.gates {
            intermediate_register = gate.apply(intermediate_register);
        }
        return intermediate_register;
    }

    pub fn as_gate(&self) -> QuantumGate {
        let mut gate = QuantumGate::identity(self.n_qubits);
        for g in &self.gates {
            gate = gate.compose(g);
        }
        return gate;
    }

    pub fn reverse(&self) -> Self {
        let mut reversed_gates = self.gates.iter().map(|gate| gate.clone().reverse()).collect::<Vec<_>>();
        reversed_gates.reverse();
        Self { n_qubits: self.n_qubits, gates: reversed_gates }
    }

    pub fn extend(&mut self, other: &Self) {
        assert_eq!(self.n_qubits, other.n_qubits);
        self.gates.extend(other.gates.iter().map(|gate| gate.clone()));
    }

    pub fn fourier_transform(n_qubits: usize) -> Self {
        let mut to_return = Self::new(n_qubits);
        for i in 0..n_qubits {
            let partial = Self::partial_fourier_transform(n_qubits, i);
            to_return.extend(&partial);
        }
        to_return.add_gate(QuantumGate::permutation((0..n_qubits).rev().collect()), (0..n_qubits).collect());
        return to_return;
    }

    fn partial_fourier_transform(n_qubits: usize, start_idx: usize) -> Self {
        let mut circuit = Self::new(n_qubits);
        circuit.add_gate(QuantumGate::hadamard(), vec![start_idx]);
        for i in (start_idx + 1)..(n_qubits - start_idx) {
            let k = 2usize.pow((i + 1 - start_idx) as u32);
            let phase_shift_gate = QuantumGate::controlled_phase_shift(TAU / (k as f32));
            circuit.add_gate(phase_shift_gate, vec![start_idx + i, start_idx]);
        }
        return circuit;
    }
    
    pub fn inverse_fourier_transform(n_qubits: usize) -> Self {
        Self::fourier_transform(n_qubits).reverse()
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
        circuit.add_gate(QuantumGate::identity(1), vec![0]);

        assert!(circuit.run(Qubit::basis_0()).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(Qubit::basis_1()).almost_equals(Qubit::basis_1()));
        assert!(circuit.run(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)).almost_equals(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)));

    }

    #[test]
    fn test_serial_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);
        circuit.add_gate(QuantumGate::not(), vec![0]);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);

        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_0())).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_1())).almost_equals(Qubit::from_coefficients_normalize(Complex::zero(), -Complex::one())));
        assert!(circuit.run(QuantumRegister::singleton((Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 1.)))).almost_equals(Qubit::from_coefficients_normalize(Complex::one()/SQRT_2, -Complex::one()/SQRT_2)));
    }

    #[test]
    fn test_quantum_circuit_order() {

        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);
        circuit.add_gate(QuantumGate::not(), vec![1]);

        // |00> -> |00> -> |10>
        // |01> -> |01> -> |11>
        // |10> -> |11> -> |01>
        // |11> -> |10> -> |00>

        assert!(circuit.clone().run(QuantumRegister::basis(2, 0)).almost_equals(QuantumRegister::basis(2, 2)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 1)).almost_equals(QuantumRegister::basis(2, 3)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 2)).almost_equals(QuantumRegister::basis(2, 1)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 3)).almost_equals(QuantumRegister::basis(2, 0)));


        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::not(), vec![1]);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);

        // |00> -> |10> -> |11>
        // |01> -> |11> -> |10>
        // |10> -> |00> -> |00>
        // |11> -> |01> -> |01>

        assert!(circuit.clone().run(QuantumRegister::basis(2, 0)).almost_equals(QuantumRegister::basis(2, 3)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 1)).almost_equals(QuantumRegister::basis(2, 2)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 2)).almost_equals(QuantumRegister::basis(2, 0)));
        assert!(circuit.clone().run(QuantumRegister::basis(2, 3)).almost_equals(QuantumRegister::basis(2, 1)));

        
        
    }


    #[test]
    fn test_quantum_circuit_adds_not_gates_to_the_top() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::not(), vec![0]);
        // |000> -> |001>
        // |001> -> |000>
        // |010> -> |011>
        // |011> -> |010>
        // |100> -> |101>
        // |101> -> |100>
        // |110> -> |111>
        // |111> -> |110>

        assert!(circuit.run(QuantumRegister::from_int(3, 0)).clone().almost_equals(QuantumRegister::from_int(3, 1)));
        assert!(circuit.run(QuantumRegister::from_int(3, 1)).clone().almost_equals(QuantumRegister::from_int(3, 0)));
        assert!(circuit.run(QuantumRegister::from_int(3, 2)).clone().almost_equals(QuantumRegister::from_int(3, 3)));
        assert!(circuit.run(QuantumRegister::from_int(3, 3)).clone().almost_equals(QuantumRegister::from_int(3, 2)));
        assert!(circuit.run(QuantumRegister::from_int(3, 4)).clone().almost_equals(QuantumRegister::from_int(3, 5)));
        assert!(circuit.run(QuantumRegister::from_int(3, 5)).clone().almost_equals(QuantumRegister::from_int(3, 4)));
        assert!(circuit.run(QuantumRegister::from_int(3, 6)).clone().almost_equals(QuantumRegister::from_int(3, 7)));
        assert!(circuit.run(QuantumRegister::from_int(3, 7)).clone().almost_equals(QuantumRegister::from_int(3, 6)));

    }

    #[test]
    fn test_quantum_circuit_adds_gates_to_the_top() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);
        // |000> -> 1/root(2)*(|000> + |001>)

        let expected = QuantumRegister::mixture(vec![QuantumRegister::from_int(3, 0), QuantumRegister::from_int(3, 1)]);
        let actual = circuit.run(QuantumRegister::all_0s(3));
        assert!(actual.clone().almost_equals(expected), "{:?}", actual);

    }

    #[test]
    fn test_quantum_circuit_adds_not_gates_at_the_end() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::not(), vec![1]);
        // |00> -> |10>
        // |01> -> |11>
        // |10> -> |00>
        // |11> -> |01>

        assert!(circuit.clone().run(QuantumRegister::from_int(2, 0)).clone().almost_equals(QuantumRegister::from_int(2, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(2, 1)).clone().almost_equals(QuantumRegister::from_int(2, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 1)));
        assert!(circuit.clone().run(QuantumRegister::from_int(2, 2)).clone().almost_equals(QuantumRegister::from_int(2, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(2, 3)).clone().almost_equals(QuantumRegister::from_int(2, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 3)));

    }
    
    
    #[test]
    fn test_quantum_circuit_adds_not_gates_at_the_end_3() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::not(), vec![2]);
        // |000> -> |100>
        // |001> -> |101>
        // |010> -> |110>
        // |011> -> |111>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 0)).clone().almost_equals(QuantumRegister::from_int(3, 4)), "{}\n\n{}", circuit, circuit.clone().run(QuantumRegister::from_int(3, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 1)).clone().almost_equals(QuantumRegister::from_int(3, 5)), "{}\n\n{}", circuit, circuit.clone().run(QuantumRegister::from_int(3, 1)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 2)).clone().almost_equals(QuantumRegister::from_int(3, 6)), "{}\n\n{}", circuit, circuit.clone().run(QuantumRegister::from_int(3, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 3)).clone().almost_equals(QuantumRegister::from_int(3, 7)), "{}\n\n{}", circuit, circuit.clone().run(QuantumRegister::from_int(3, 3)));

    }
    
    #[test]
    fn test_quantum_circuit_adds_not_gates_in_the_middle() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::not(), vec![1]);
        // |000> -> |010>
        // |001> -> |011>
        // |010> -> |000>
        // |011> -> |001>
        // |100> -> |110>
        // |101> -> |111>
        // |110> -> |100>
        // |111> -> |101>

        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 0)).clone().almost_equals(QuantumRegister::from_int(3, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 1)).clone().almost_equals(QuantumRegister::from_int(3, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 1)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 2)).clone().almost_equals(QuantumRegister::from_int(3, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 3)).clone().almost_equals(QuantumRegister::from_int(3, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 3)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 4)).clone().almost_equals(QuantumRegister::from_int(3, 6)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 4)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 5)).clone().almost_equals(QuantumRegister::from_int(3, 7)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 5)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 6)).clone().almost_equals(QuantumRegister::from_int(3, 4)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 6)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 7)).clone().almost_equals(QuantumRegister::from_int(3, 5)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 7)));
        
    }
    
    
    #[test]
    fn test_quantum_circuit_adds_gates_in_the_middle() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::hadamard(), vec![1]);
        // |000> -> 1/root(2)*(|000> + |010>)

        let expected = QuantumRegister::mixture(vec![QuantumRegister::from_int(3, 0), QuantumRegister::from_int(3, 2)]);
        let actual = circuit.run(QuantumRegister::all_0s(3));
        assert!(actual.clone().almost_equals(expected), "{:?}", actual);

    }

    #[test]
    fn test_quantum_circuit_adds_two_qubit_gates_at_the_beginning() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);
        // |000> -> |000>
        // |001> -> |001>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 0)).clone().almost_equals(QuantumRegister::from_int(3, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 1)).clone().almost_equals(QuantumRegister::from_int(3, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 1)));

        // |010> -> |011>
        // |011> -> |010>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 2)).clone().almost_equals(QuantumRegister::from_int(3, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 3)).clone().almost_equals(QuantumRegister::from_int(3, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 3)));

        // |100> -> |100>
        // |101> -> |101>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 4)).clone().almost_equals(QuantumRegister::from_int(3, 4)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 4)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 5)).clone().almost_equals(QuantumRegister::from_int(3, 5)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 5)));

        // |110> -> |111>
        // |111> -> |110>

        assert!(circuit.clone().run(QuantumRegister::from_int(3, 6)).clone().almost_equals(QuantumRegister::from_int(3, 7)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 6)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 7)).clone().almost_equals(QuantumRegister::from_int(3, 6)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 7)));

    }

    #[test]
    fn test_quantum_circuit_adds_two_qubit_gates_at_the_end() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::cnot(), vec![1, 2]);
        // |000> -> |000>
        // |001> -> |001>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 0)).clone().almost_equals(QuantumRegister::from_int(3, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 1)).clone().almost_equals(QuantumRegister::from_int(3, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 1)));

        // |010> -> |010>
        // |011> -> |011>

        assert!(circuit.clone().run(QuantumRegister::from_int(3, 2)).clone().almost_equals(QuantumRegister::from_int(3, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 3)).clone().almost_equals(QuantumRegister::from_int(3, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 3)));
        
        // |100> -> |110>
        // |101> -> |111>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 4)).clone().almost_equals(QuantumRegister::from_int(3, 6)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 4)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 5)).clone().almost_equals(QuantumRegister::from_int(3, 7)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 5)));
        
        // |110> -> |100>
        // |111> -> |101>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 6)).clone().almost_equals(QuantumRegister::from_int(3, 4)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 6)));
        assert!(circuit.clone().run(QuantumRegister::from_int(3, 7)).clone().almost_equals(QuantumRegister::from_int(3, 5)), "{:?}", circuit.clone().run(QuantumRegister::from_int(3, 7)));
        
    }

    
        #[test]
        fn test_quantum_circuit_reverses_cnot_gate_inputs() {
            
            let mut circuit = QuantumCircuit::new(2);
            circuit.add_gate(QuantumGate::cnot(), vec![1, 0]);
            
            // |00> -> |00>
            // |01> -> |11>
            
            assert!(circuit.clone().run(QuantumRegister::from_int(2, 0)).clone().almost_equals(QuantumRegister::from_int(2, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 0)));
            assert!(circuit.clone().run(QuantumRegister::from_int(2, 1)).clone().almost_equals(QuantumRegister::from_int(2, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 1)));
    
            // |10> -> |10>
            // |11> -> |01>
    
            assert!(circuit.clone().run(QuantumRegister::from_int(2, 2)).clone().almost_equals(QuantumRegister::from_int(2, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 2)));
            assert!(circuit.clone().run(QuantumRegister::from_int(2, 3)).clone().almost_equals(QuantumRegister::from_int(2, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(2, 3)));
            
        }
    
    #[test]
    fn test_quantum_circuit_adds_two_qubit_gates_in_the_middle() {
        let mut circuit = QuantumCircuit::new(4);
        circuit.add_gate(QuantumGate::cnot(), vec![1, 2]);
        // |0000> -> |0000>
        // |0001> -> |0001>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 0)).clone().almost_equals(QuantumRegister::from_int(4, 0)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 0)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 1)).clone().almost_equals(QuantumRegister::from_int(4, 1)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 1)));
        
        // |0010> -> |0010>
        // |0011> -> |0011>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 2)).clone().almost_equals(QuantumRegister::from_int(4, 2)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 2)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 3)).clone().almost_equals(QuantumRegister::from_int(4, 3)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 3)));

        // |0100> -> |0110>
        // |0101> -> |0111>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 4)).clone().almost_equals(QuantumRegister::from_int(4, 6)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 4)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 5)).clone().almost_equals(QuantumRegister::from_int(4, 7)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 5)));

        // |0110> -> |0100>
        // |0111> -> |0101>

        assert!(circuit.clone().run(QuantumRegister::from_int(4, 6)).clone().almost_equals(QuantumRegister::from_int(4, 4)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 6)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 7)).clone().almost_equals(QuantumRegister::from_int(4, 5)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 7)));
        
        // |1000> -> |1000>
        // |1001> -> |1001>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 8)).clone().almost_equals(QuantumRegister::from_int(4, 8)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 8)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 9)).clone().almost_equals(QuantumRegister::from_int(4, 9)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 9)));
        
        // |1010> -> |1010>
        // |1011> -> |1011>
        
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 10)).clone().almost_equals(QuantumRegister::from_int(4, 10)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 10)));
        assert!(circuit.clone().run(QuantumRegister::from_int(4, 11)).clone().almost_equals(QuantumRegister::from_int(4, 11)), "{:?}", circuit.clone().run(QuantumRegister::from_int(4, 11)));
        
    }
    #[test]
    fn test_two_qubit_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);

        let zero_zero = QuantumRegister::basis(2, 0);
        let zero_one = QuantumRegister::basis(2, 1);
        let one_zero = QuantumRegister::basis(2, 2);
        let one_one = QuantumRegister::basis(2, 3);

        assert!(circuit.run(zero_zero.clone()).almost_equals(zero_zero.clone()));
        assert!(circuit.run(zero_one.clone()).almost_equals(zero_one.clone()));
        assert!(circuit.run(one_zero.clone()).almost_equals(one_one.clone()));
        assert!(circuit.run(one_one.clone()).almost_equals(one_zero.clone()));
        
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);
        circuit.add_gate(QuantumGate::hadamard(), vec![1]);

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
    fn test_trivial_quantum_circuit() {

        let circuit = QuantumCircuit::new(1);
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_0())).almost_equals(Qubit::basis_0()));
        assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_1())).almost_equals(Qubit::basis_1()));
        
    }

    #[test]
    fn test_quantum_circuit_reverses() {

        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);
        circuit.add_gate(QuantumGate::hadamard(), vec![1]);
        circuit.add_gate(QuantumGate::cnot(), vec![0, 1]);
        circuit.add_gate(QuantumGate::hadamard(), vec![0]);
        circuit.add_gate(QuantumGate::hadamard(), vec![1]);

        let mut reversed = circuit.reverse();

        let mut expected = QuantumCircuit::new(2);
        expected.add_gate(QuantumGate::hadamard(), vec![1]);
        expected.add_gate(QuantumGate::hadamard(), vec![0]);
        expected.add_gate(QuantumGate::cnot(), vec![0, 1]);
        expected.add_gate(QuantumGate::hadamard(), vec![1]);
        expected.add_gate(QuantumGate::hadamard(), vec![0]);

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
        hadamard_circuit.add_gate(hadamard_gate.clone(), vec![0]);
        
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
        full_circuit.add_gate(hadamard_gate, vec![0]);
        let (measurement, after_measurement) = full_circuit.run(register).measure();

        assert_eq!(measurement, 0);
        assert!(after_measurement.almost_equals(Qubit::basis_0()));
        

    }

    #[test]
    fn test_fourier_and_inverse_fourier_are_inverses() {
        
        let ft = QuantumCircuit::fourier_transform(4);
        let ift = QuantumCircuit::inverse_fourier_transform(4);

        for i in 0..15 {
            let input = QuantumRegister::from_int(4, i);
            let output = ft.run(input.clone());
            let output2 = ift.run(output);
            assert!(output2.almost_equals(input));
        }
    }

    #[test]
    fn test_fourier_transform_trivial() {
        
        let ft = QuantumCircuit::fourier_transform(1);

        let input = QuantumRegister::from_int(1, 0);
        let expected = Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 1.);
        assert!(ft.clone().run(input).almost_equals(expected));

        let input = QuantumRegister::from_int(1, 1);

        let expected = Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., -1.);
        assert!(ft.clone().run(input.clone()).almost_equals(expected.clone()), "Actual:\n{}\n\nExpected:\n{}", ft.clone().run(input.clone()), QuantumRegister::singleton(expected.clone()));
    }


    #[test]
    fn test_fourier_transform_simple() {
        let ft = QuantumCircuit::fourier_transform(2);
        println!("{}", ft);
        println!("Circuit as Gate:\n{}", ft.as_gate());
        let m = 2usize.pow(2);

        let actual_0 = ft.run(QuantumRegister::from_int(2, 0));
        let expected_0 = QuantumRegister::mixture(QuantumRegister::all_bases(2));
        assert!(actual_0.almost_equals(expected_0));

        let actual_1 = ft.run(QuantumRegister::from_int(2, 1));
        let expected_1 = QuantumRegister::mixture(
            vec![
                QuantumRegister::basis(2, 0),
                QuantumRegister::basis(2, 1).rotate(TAU/m as f32),
                QuantumRegister::basis(2, 2).rotate(2.*TAU/m as f32),
                QuantumRegister::basis(2, 3).rotate(3.*TAU/m as f32),
                ]
        );
        assert!(actual_1.clone().almost_equals(expected_1.clone()), "\n\n{}\n\n vs. \n\n{}", actual_1.clone(), expected_1.clone());
    }


    #[test]
    fn test_fourier_transform() {

        let x_0 = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., Complex::exp(TAU * 3./4. * Complex::i())));
        let x_1 = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., Complex::exp(TAU * 1./2. * Complex::i())));
        
        let expected = x_1.tensor_product(&x_0);

        assert!(
            QuantumCircuit::fourier_transform(2).run(QuantumRegister::from_int(2, 3)).almost_equals(expected.clone()),
            "\n\n{}\n\n vs. \n\n{}",
            QuantumCircuit::fourier_transform(2).run(QuantumRegister::from_int(2, 3)),
            expected
        );

    }

    #[test]
    fn test_fourier_transform_3() {

        let ft = QuantumCircuit::fourier_transform(3);
        println!("{}", ft);
        let m = 2usize.pow(3);

        let actual_0 = ft.run(QuantumRegister::from_int(3, 0));
        let expected_0 = QuantumRegister::mixture(QuantumRegister::all_bases(3));
        assert!(actual_0.almost_equals(expected_0));

        for x in 0..m {
            let mut rotated_bases = Vec::new();
            for y in 0..m {
                let y_basis = QuantumRegister::basis(3, y);
                let rotated = y_basis.rotate(TAU * (y as f32) * (x as f32) / (m as f32));
                rotated_bases.push(rotated.clone());
            };
    
            let expected = QuantumRegister::mixture(rotated_bases);
            let actual = ft.clone().run(QuantumRegister::basis(3, x));
    
            assert!(actual.clone().almost_equals(expected.clone()), "x = {}: \n\n{}\n\n vs. \n\n{}", x, actual, expected);

        }

    }

    #[test]
    fn test_inverse_fourier_transform_trivial() {
        let ift = QuantumCircuit::inverse_fourier_transform(1);
        
        let non_rotated = QuantumRegister::basis(1, 0);
        let rotated = QuantumRegister::mixture(vec![QuantumRegister::basis(1, 0), QuantumRegister::basis(1, 1)]);

        let mut hadamard_only = QuantumCircuit::new(1);
        hadamard_only.add_gate(QuantumGate::hadamard(), vec![0]);

        assert!(hadamard_only.run(non_rotated.clone()).almost_equals(ift.run(non_rotated)));
        assert!(hadamard_only.run(rotated.clone()).almost_equals(ift.run(rotated)));
    }

    #[test]
    fn test_inverse_fourier_transform_simple() {
        let ift = QuantumCircuit::inverse_fourier_transform(2);
        println!("{}", ift);
        let m = 2usize.pow(2);

        let actual_0 = ift.run(QuantumRegister::from_int(2, 0));
        let expected_0 = QuantumRegister::mixture(QuantumRegister::all_bases(2));
        assert!(actual_0.almost_equals(expected_0));

        let actual_1 = ift.run(QuantumRegister::from_int(2, 1));
        let expected_1 = QuantumRegister::mixture(
            vec![
                QuantumRegister::basis(2, 0),
                QuantumRegister::basis(2, 1).rotate(-TAU/m as f32),
                QuantumRegister::basis(2, 2).rotate(-2.*TAU/m as f32),
                QuantumRegister::basis(2, 3).rotate(-3.*TAU/m as f32),
                ]
        );
        assert!(actual_1.clone().almost_equals(expected_1.clone()), "\n\n{}\n\n vs. \n\n{}", actual_1.clone(), expected_1.clone());
    }

    #[test]
    fn test_inverse_fourier_transform() {

        let ift = QuantumCircuit::inverse_fourier_transform(3);
        println!("{}", ift);
        let m = 2usize.pow(3);

        let actual_0 = ift.run(QuantumRegister::from_int(3, 0));
        let expected_0 = QuantumRegister::mixture(QuantumRegister::all_bases(3));
        assert!(actual_0.almost_equals(expected_0));

        for x in 0..m {
            let mut rotated_bases = Vec::new();
            for y in 0..m {
                let y_basis = QuantumRegister::basis(3, y);
                let rotated = y_basis.rotate(-TAU * (y as f32) * (x as f32) / (m as f32));
                rotated_bases.push(rotated.clone());
            };
    
            let expected = QuantumRegister::mixture(rotated_bases);
            let actual = ift.clone().run(QuantumRegister::basis(3, x));
    
            assert!(actual.clone().almost_equals(expected.clone()), "x = {}: \n\n{}\n\n vs. \n\n{}", x, actual, expected);

        }

    }

}