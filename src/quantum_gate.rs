use std::f32::consts::SQRT_2;

use nalgebra::{Vector2, DVector, Complex, Unit, Normed};
use num_traits::{One, Zero};
use rand::Rng;

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

    pub fn basis(n_qubits: usize, i: usize) -> Self {
        let state_vector_length = 2_usize.pow(n_qubits as u32);
        let mut register = DVector::zeros(state_vector_length);
        register[i] = Complex::one();
        Self::new_normalize(register)
    }

    pub fn all_0s(n_qubits: usize) -> Self {
        Self::basis(n_qubits, 0)
    }

    pub fn all_1s(n_qubits: usize) -> Self {
        let state_vector_length = 2_usize.pow(n_qubits as u32);
        Self::basis(n_qubits, state_vector_length - 1)
    }

    pub fn n_qubits(&self) -> usize {
        self.register.len().checked_log2().expect("Register size is not a power of 2") as usize
    }

    pub fn len(&self) -> usize {
        self.register.len()
    }

    pub fn get_vector(&self) -> Unit<DVector<Complex<f32>>> {
        self.register.clone()
    }

    pub fn almost_equals(&self, rhs: impl Into<Self>) -> bool {
        (self.register.clone().into_inner() - rhs.into().register.clone().into_inner()).norm() < 0.0001
    }

    pub fn get_coefficient(&self, i: usize) -> Complex<f32> {
        self.register[i]
    }

    pub fn get_probability(&self, i: usize) -> f32 {
        self.get_coefficient(i).norm().powi(2)
    }

    pub fn measure(&self) -> (Measurement, Self) {
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(0.0_f32..1.0);
        let mut probably_so_far = 0.;
        
        for i in 0..self.len() {
            let coefficient = self.get_coefficient(i);
            probably_so_far += self.get_probability(i);
            if random_number <= probably_so_far {
                // This is the basis state to collapse to
                let new_register = Self::basis(self.len(), i);
                return (i as u8, new_register);
            }
        }

        panic!("The measurement vector is unitary and so probability_so_far should count up to 1.0 > random_number");
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
                Complex::zero(),
                Complex::one(),
                Complex::one(),
                Complex::zero(),
            ]
        ))
    }

    pub fn cnot() -> Self {
        Self::new(SquareMatrix::from_vec_normalize(
            4,
            vec![
                Complex::one(), // 00
                Complex::zero(),
                Complex::zero(),
                Complex::zero(),
                Complex::zero(), // 10
                Complex::one(), // 11
                Complex::zero(),
                Complex::zero(),
                Complex::zero(), // 20
                Complex::zero(),
                Complex::zero(),
                Complex::one(), // 23
                Complex::zero(), // 30
                Complex::zero(),
                Complex::one(), // 32
                Complex::zero(),
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
        self.matrix.size().log2() as usize
    }

}


#[cfg(test)]
mod test_quantum_gate {

    use super::*;

    #[test]
    fn test_register_initializes() {

        let singleton = QuantumRegister::singleton(Qubit::basis_1());
        assert_eq!(singleton.n_qubits(), 1);
        assert_eq!(singleton.len(), 2);
        assert_eq!(singleton.get_coefficient(0), Complex::zero());
        assert_eq!(singleton.get_coefficient(1), Complex::one());

        let singleton = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.));
        assert_eq!(singleton.n_qubits(), 1);
        assert_eq!(singleton.len(), 2);
        assert_eq!(singleton.get_coefficient(0), Complex::one() * 0.5);
        assert_eq!(singleton.get_coefficient(1), Complex::one() * 0.75_f32.sqrt());

        let register = QuantumRegister::all_0s(2);
        assert_eq!(register.n_qubits(), 2);
        assert_eq!(register.len(), 4);
        assert_eq!(register.get_coefficient(0), Complex::one());
        assert_eq!(register.get_coefficient(1), Complex::zero());
        assert_eq!(register.get_coefficient(2), Complex::zero());
        assert_eq!(register.get_coefficient(3), Complex::zero());

        let register = QuantumRegister::all_1s(2);
        assert_eq!(register.n_qubits(), 2);
        assert_eq!(register.len(), 4);
        assert_eq!(register.get_coefficient(0), Complex::zero());
        assert_eq!(register.get_coefficient(1), Complex::zero());
        assert_eq!(register.get_coefficient(2), Complex::zero());
        assert_eq!(register.get_coefficient(3), Complex::one());

        
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
    fn test_cnot_gate() {
        
        let cnot = QuantumGate::cnot();

        let zero: Complex<f32> = Complex::zero();
        let one = Complex::one();

        assert_eq!(cnot.matrix.get(0, 0), one.clone());
        assert_eq!(cnot.matrix.get(1, 1), one.clone());
        assert_eq!(cnot.matrix.get(2, 3), one.clone());
        assert_eq!(cnot.matrix.get(3, 2), one.clone());

        let zero_zero = QuantumRegister::basis(2, 0);
        let zero_one = QuantumRegister::basis(2, 1);
        let one_one = QuantumRegister::basis(2, 2);
        let one_zero = QuantumRegister::basis(2, 3);
        
        assert_eq!(cnot.clone().apply(zero_zero.clone()), zero_zero.clone());
        assert_eq!(cnot.clone().apply(zero_one.clone()), zero_one.clone());
        assert_eq!(cnot.clone().apply(one_zero.clone()), one_one.clone());
        assert_eq!(cnot.clone().apply(one_one.clone()), one_zero.clone());
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

        let double_hadamarded_0 = hadamard_gate.apply(hadamarded_0);
        let (measurement, after_measurement) = double_hadamarded_0.measure();

        assert!(double_hadamarded_0.almost_equals(Qubit::basis_0()));
        assert_eq!(measurement, 0);
        assert!(after_measurement.almost_equals(Qubit::basis_0()));
        

    }
}