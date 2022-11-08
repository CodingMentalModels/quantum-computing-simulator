use crate::qubit::Qubit;
use crate::quantum_gate::QuantumGate;
use crate::quantum_gate::QuantumRegister;


#[cfg(test)]
mod test_quantum_circuit {

    use super::*;

    // #[test]
    // fn test_trivial_quantum_circuit() {
    //     let mut circuit = QuantumCircuit::new(1);
    //     circuit.add_gate(QuantumGate::identity(1));

    //     assert!(circuit.run(Qubit::basis_0()).almost_equals(&Qubit::basis_0()));
    //     assert!(circuit.run(Qubit::basis_1()).almost_equals(&Qubit::basis_1()));
    //     assert!(circuit.run(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)).almost_equals(&Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)));

    // }

    // #[test]
    // fn test_serial_quantum_circuit() {
    //     let mut circuit = QuantumCircuit::new(1);
    //     circuit.add_gate(QuantumGate::hadamard(), 0);
    //     circuit.add_gate(QuantumGate::not(), 0);
    //     circuit.add_gate(QuantumGate::hadamard(), 0);

    //     assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_0())).almost_equals(Qubit::basis_1()));
    //     assert!(circuit.run(QuantumRegister::singleton(Qubit::basis_1())).almost_equals(Qubit::basis_0()));
    //     assert!(circuit.run(QuantumRegister::singleton((Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 1.)))).almost_equals(Qubit::mix(Qubit::basis_1(), Qubit::basis_0(), 1., 1.)));
    // }

    // #[test]
    // fn test_two_qubit_quantum_circuit() {
    //     let mut circuit = QuantumCircuit::new(2);
    //     circuit.add_gate(QuantumGate::hadamard(), 0);
    //     circuit.add_gate(QuantumGate::hadamard(), 1);
    //     circuit.add_gate(QuantumGate::cnot(), 0);

    //     assert!(circuit.run(QuantumRegister::fill(Qubit::basis_0(), 2)).almost_equals(&QuantumRegister::fill(Qubit::basis_0(), 2)));
    //     assert!(circuit.run(QuantumRegister::fill(Qubit::basis_1(), 2)).almost_equals(&QuantumRegister::fill(Qubit::basis_1(), 2)));
    // }

    #[test]
    fn test_quantum_circuit_initializes() {
        unimplemented!();
    }

    #[test]
    fn test_quantum_circuit_reverses() {
        unimplemented!();
    }

}