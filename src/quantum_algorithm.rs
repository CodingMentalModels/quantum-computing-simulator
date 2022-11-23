use crate::{quantum_circuit::QuantumCircuit};
use crate::quantum_register::QuantumRegister;



#[derive(Debug)]
pub struct OrderFindingAlgorithm {
    capital_n: u32,
    circuit: QuantumCircuit,
}

impl OrderFindingAlgorithm {

    pub fn new(capital_n: u32) -> Self {
        let n_qubits = 2 * (capital_n.log(2) + 1) as usize;
        Self {capital_n, circuit: QuantumCircuit::order_finding(n_qubits)}
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
        
        let mut algorithm = OrderFindingAlgorithm::new(15);
        let result = algorithm.run();
        assert_eq!(result, 3);

    }
}