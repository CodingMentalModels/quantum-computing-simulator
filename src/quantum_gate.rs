use std::f32::consts::SQRT_2;

use nalgebra::{Vector2, DVector, Complex, Unit};
use num_traits::One;

use crate::qubit::{Qubit, Measurement};
use crate::matrix::SquareMatrix;

#[derive(Clone, Debug, PartialEq)]
pub struct QuantumRegister {
    register: Unit<DVector<Complex<f32>>>,
}

impl From<Qubit> for QuantumRegister {
    fn from(qubit: Qubit) -> Self {
        Self::singleton(qubit)
    }
}

impl QuantumRegister {

    pub fn new(register: Unit<DVector<Complex<f32>>>) -> Self {
        Self { register }
    }

    pub fn new_normalize(register: DVector<Complex<f32>>) -> Self {
        Self::new(Unit::<DVector<Complex<f32>>>::new_normalize(register))
    }

    pub fn singleton(qubit: Qubit) -> Self {
        Self::new_normalize(DVector::from_vec(vec![qubit.get_state().into_inner().x, qubit.get_state().into_inner().y]))
    }

    pub fn fill(qubit: Qubit, size: usize) -> Self {
        let (q_0, q_1) = (qubit.get_state().into_inner().x, qubit.get_state().into_inner().y);
        let mut register_vec = Vec::new();
        for _ in 0..size {
            register_vec.push(q_0);
            register_vec.push(q_1);
        }
        Self::new_normalize(DVector::from(register_vec))
    }

    pub fn len(&self) -> usize {
        self.register.len() / 2
    }

    pub fn get_qubit(&self, index: usize) -> Qubit {
        Qubit::new(Unit::new_normalize(Vector2::new(self.register[2 * index], self.register[2 * index + 1])))
    }

    pub fn get_vector(&self) -> Unit<DVector<Complex<f32>>> {
        self.register.clone()
    }

    pub fn almost_equals(&self, rhs: impl Into<Self>) -> bool {
        (self.register.clone().into_inner() - rhs.into().register.clone().into_inner()).norm() < 0.0001
    }

    pub fn measure(&self) -> (Vec<Measurement>, Self) {
        let rng = rand::thread_rng();
        let mut resulting_register = Vec::new();
        let mut resulting_measurements = Vec::new();

        for i in 0..self.len() {
            let qubit = self.get_qubit(i);
            let (measurement, new_qubit) = qubit.measure();
            resulting_measurements.push(measurement);
            resulting_register.push(new_qubit.get_state().into_inner().x);
            resulting_register.push(new_qubit.get_state().into_inner().y);
        }

        return (resulting_measurements, Self::new_normalize(DVector::from(resulting_register)));
    }

}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGate {
    matrix: SquareMatrix,
}

impl QuantumGate {

    pub fn new(matrix: SquareMatrix) -> Self {
        Self { matrix }
    }

    pub fn identity(size: usize) -> Self {
        Self::new(SquareMatrix::one(2*size))
    }

    pub fn not() -> Self {
        Self::new(SquareMatrix::from_vec_normalize(
            2,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ]
        ))
    }

    pub fn hadamard() -> Self {
        Self::new(
            SquareMatrix::from_vec_normalize(
                2,
                vec![
                    Complex::one() * 1./SQRT_2, Complex::one() * 1./SQRT_2,
                    Complex::one() * 1./SQRT_2, Complex::one() * -1./SQRT_2,
                ]
            )
        )
    }

    pub fn almost_equals(&self, rhs: &Self) -> bool {
        self.matrix.almost_equals(&rhs.matrix)
    }

    pub fn apply(&self, register: impl Into<QuantumRegister>) -> QuantumRegister {
        QuantumRegister::new_normalize(self.matrix.clone() * register.into().get_vector())
    }

    pub fn n_qubits(&self) -> usize {
        self.matrix.size() / 2
    }

}


#[cfg(test)]
mod test_quantum_gate {

    use super::*;

    #[test]
    fn test_register_initializes() {

        let singleton = QuantumRegister::singleton(Qubit::basis_1());
        assert_eq!(singleton.len(), 1);
        assert_eq!(singleton.get_qubit(0), Qubit::basis_1());

        let singleton = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.));
        assert_eq!(singleton.len(), 1);
        assert_eq!(singleton.get_qubit(0), Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.));

        let register = QuantumRegister::fill(Qubit::basis_0(), 2);
        assert_eq!(register.len(), 2);
        assert_eq!(register.get_qubit(0), Qubit::basis_0());
        assert_eq!(register.get_qubit(1), Qubit::basis_0());

        
    }

    #[test]
    fn test_quantum_gate_initializes() {
        
        let gate = QuantumGate::new(
            SquareMatrix::identity(2)
        );

        let identity_gate = QuantumGate::identity(1);
        assert!(gate.almost_equals(&identity_gate));
        assert_eq!(gate.n_qubits(), 1);

        let basis_0 = Qubit::basis_0();
        let basis_1 = Qubit::basis_1();

        let mixed_state = Qubit::mix(basis_0, basis_1, 1., 3.);
        let result = gate.apply(QuantumRegister::singleton(mixed_state));

        assert!(result.almost_equals(QuantumRegister::singleton(mixed_state)), "{:?} != {:?}", result, QuantumRegister::singleton(mixed_state));

        let hadamard_gate = QuantumGate::hadamard();

        assert!(
            hadamard_gate.apply(QuantumRegister::singleton(basis_0)).almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one() * 0.5_f32.sqrt(), Complex::one() * 0.5_f32.sqrt())))
            )
        );
        assert!(
            hadamard_gate.apply(QuantumRegister::singleton(basis_1)).almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one() * 0.5_f32.sqrt(), -Complex::one() * 0.5_f32.sqrt())))
            )
        );

        let result = hadamard_gate.apply(QuantumRegister::singleton(mixed_state));
        assert!(
            result.almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one() * (0.25_f32.sqrt() + 0.75_f32.sqrt()), Complex::one() * (0.25_f32.sqrt() - 0.75_f32.sqrt()))))
            ),
            "{:?}",
            result,
        );

    }

    #[test]
    fn test_not_gate() {

        assert!(
            QuantumGate::not().apply(Qubit::basis_0()).almost_equals(Qubit::basis_1())
        );
        assert!(
            QuantumGate::not().apply(Qubit::basis_1()).almost_equals(Qubit::basis_0())
        );
        assert!(
            QuantumGate::not().apply(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.)).almost_equals(
                Qubit::mix(Qubit::basis_1(), Qubit::basis_0(), 1., 3.)
            )
        );
        
    }

    #[test]
    fn test_measurement_interference() {

        // Circuit with measurement in the middle:
        // |0> -> H -> Measure -> H => |0> with 50% probability, |1> with 50% probability
        
        let hadamard_gate = QuantumGate::hadamard();
        
        let basis_0 = Qubit::basis_0();
        
        let hadamarded_0 = hadamard_gate.apply(basis_0);
        
        let mut count_0s = 0;
        let mut count_1s = 1;
        
        for _ in 0..1000 {
            let (measurement, after_measurement) = hadamarded_0.measure();

            assert!(after_measurement.almost_equals(Qubit::basis_0()) || after_measurement.almost_equals(Qubit::basis_1()));

            let (second_measurement, after_second_measurement) = hadamard_gate.apply(after_measurement).measure();

            assert!(after_second_measurement.almost_equals(Qubit::basis_0()) || after_second_measurement.almost_equals(Qubit::basis_1()));
            if second_measurement[0] == 0 {
                count_0s += 1;
            } else {
                count_1s += 1;
            }
        }

        assert!(count_0s > 400); // Should be 50% = 500, so should be safe
        assert!(count_1s > 400); // Should be 50% = 500, so should be safe

        // Circuit without measurement in the middle:
        // |0> -> H -> H => |0> with 100% probability, |1> with 0% probability

        let double_hadamarded_0 = hadamard_gate.apply(hadamarded_0);
        let (measurement, after_measurement) = double_hadamarded_0.measure();

        assert!(double_hadamarded_0.almost_equals(Qubit::basis_0()));
        assert_eq!(measurement[0], 0);
        assert!(after_measurement.almost_equals(Qubit::basis_0()));
        

    }
}