use crate::{quantum_circuit::QuantumCircuit};
use crate::quantum_register::QuantumRegister;



#[derive(Debug)]
pub struct OrderFindingAlgorithm {
    capital_n: u32,
    circuit: QuantumCircuit,
}

impl OrderFindingAlgorithm {

    pub fn new(capital_n: u32) -> Self {
        Self {capital_n, circuit: QuantumCircuit::order_finding(capital_n as usize)}
    }

    pub fn get_circuit(&self) -> &QuantumCircuit {
        &self.circuit
    }

    pub fn run(&mut self) -> u8 {
        let n_qubits = self.circuit.n_qubits() / 2;
        let control_register = QuantumRegister::basis(n_qubits, 0);
        let target_register = QuantumRegister::basis(n_qubits, 1);

        let register = control_register.tensor_product(&target_register);

        let output = self.circuit.run(register);

        // MEASURE JUST THE CONTROL REGISTER
        let (measurement, _) = output.measure();
        return measurement;

    }

}

#[cfg(test)]
mod test_quantum_algorithm {
    use super::*;

    #[test]
    fn test_order_finding_algorithm() {
        
        let mut algorithm = OrderFindingAlgorithm::new(6);
        let result = algorithm.run();
        assert_eq!(result, 3);

    }
}