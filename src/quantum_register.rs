use std::fmt::{Formatter, Display, Debug};

use nalgebra::{Unit, DVector, Complex, ComplexField, Normed};
use num_traits::One;
use rand::Rng;

use crate::qubit::{Qubit, Measurement};





#[derive(Clone, PartialEq)]
pub struct QuantumRegister {
    register: Unit<DVector<Complex<f32>>>,
}

impl Debug for QuantumRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.register)
    }
}

impl Display for QuantumRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = self.register.iter().map(|x| format!("{:?}", x)).collect::<Vec<String>>().join("\n");
        write!(f, "{}", s)
    }
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

    pub fn from_vec_normalize(vec: Vec<Complex<f32>>) -> Self {
        Self::new_normalize(DVector::from_vec(vec))
    }

    pub fn from_int(n_qubits: usize, value: usize) -> Self {
        let size = 2usize.pow(n_qubits as u32);
        assert!(value < size);
        let mut register = DVector::zeros(size);
        register[value] = Complex::one();
        Self::new_normalize(register)
    }

    pub fn singleton(qubit: Qubit) -> Self {
        Self::new_normalize(DVector::from_vec(vec![qubit.get_state().into_inner().x, qubit.get_state().into_inner().y]))
    }

    pub fn basis(n_qubits: usize, i: usize) -> Self {
        Self::from_int(n_qubits, i)
    }

    pub fn all_bases(n_qubits: usize) -> Vec<Self> {
        let mut bases = Vec::new();
        for i in 0..2_usize.pow(n_qubits as u32) {
            bases.push(Self::basis(n_qubits, i));
        }
        return bases;
    }

    pub fn basis_subspace_indices(n_qubits: usize, qubit_indices: Vec<usize>) -> Vec<usize> {
        assert!(n_qubits > 0);
        assert!(qubit_indices.len() <= n_qubits);

        let mut sorted_qubits = qubit_indices.clone();
        sorted_qubits.sort();
        assert!(sorted_qubits.iter().zip(sorted_qubits.iter().skip(1)).all(|(a, b)| a != b));
        
        let mut basis_indices = Vec::new();
        for qubit_idx in sorted_qubits {
            let new_bit_value = 2usize.pow(qubit_idx as u32);
            if basis_indices.len() == 0 {
                basis_indices.push(0);
                basis_indices.push(new_bit_value);
            } else {
                for idx in 0..basis_indices.len() {
                    basis_indices.push(basis_indices[idx] + new_bit_value);
                }
            }
        }
        return basis_indices;
    }

    pub fn tensor_product(&self, other: &Self) -> Self {
        Self::new_normalize(self.register.kronecker(&other.register))
    }

    pub fn basis_subspace(n_qubits: usize, qubit_indices: Vec<usize>) -> Vec<Self> {
        Self::basis_subspace_indices(n_qubits, qubit_indices).iter()
            .map(|&i| Self::basis(n_qubits, i))
            .collect()
    }

    pub fn all_0s(n_qubits: usize) -> Self {
        Self::basis(n_qubits, 0)
    }

    pub fn all_1s(n_qubits: usize) -> Self {
        let state_vector_length = 2_usize.pow(n_qubits as u32);
        Self::basis(n_qubits, state_vector_length - 1)
    }

    pub fn mixture(registers: Vec<Self>) -> Self {
        assert!(registers.len() > 0);
        let n_qubits = registers[0].n_qubits();
        assert!(registers.iter().all(|x| x.n_qubits() == n_qubits));
        let mut register = DVector::zeros(registers[0].register.len());
        for r in registers {
            register = register + r.register.into_inner();
        }
        Self::new_normalize(register)
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
        let inner_rhs = rhs.into();
        assert_eq!(self.n_qubits(), inner_rhs.clone().n_qubits());
        (self.register.clone().into_inner() - inner_rhs.register.clone().into_inner()).norm() < 0.0001
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
            probably_so_far += self.get_probability(i);
            if random_number <= probably_so_far {
                // This is the basis state to collapse to
                let new_register = Self::basis(self.n_qubits(), i);
                return (i as u8, new_register);
            }
        }

        panic!("The measurement vector is unitary and so probability_so_far should count up to 1.0 > random_number");
    }

    pub fn partially_measure(&self, qubits: Vec<usize>) -> (Measurement, Self) {
        let (measurement, new_register) = self.measure();
        let mut to_return = 0;
        for i in qubits {
            to_return += ((measurement >> i) & 1) << i;
        }

        return (to_return as u8, new_register);
    }

    pub fn rotate(&self, phase: f32) -> Self {
        let new_register =
            self.register.clone().into_inner() * Complex::exp(phase * Complex::i());
        return Self::new_normalize(new_register);
    }

}

#[cfg(test)]
mod test_quantum_register {
    use std::iter;

    use nannou::prelude::TAU;
    use num_traits::Zero;

    use super::*;


    #[test]
    fn test_register_initializes() {

        let singleton = QuantumRegister::singleton(Qubit::basis_1());
        assert_eq!(singleton.n_qubits(), 1);
        assert_eq!(singleton.len(), 2);
        assert_eq!(singleton.get_coefficient(0), Complex::zero());
        assert_eq!(singleton.get_coefficient(1), Complex::one());

        let singleton = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.0_f32.sqrt()));
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
    fn test_quantum_register_instantiates_bases() {
        assert!(
            QuantumRegister::from_int(3, 2).almost_equals(
                QuantumRegister::from_vec_normalize(
                    vec![
                        Complex::zero(),
                        Complex::zero(),
                        Complex::one(),
                        Complex::zero(),
                        Complex::zero(),
                        Complex::zero(),
                        Complex::zero(),
                        Complex::zero()
                    ]
                )
            ),
            "{:?}",
            QuantumRegister::from_int(3, 2)
        );
    }


    #[test]
    fn test_register_measures() {

        let register = QuantumRegister::from_int(
            3,
            4
        );

        assert_eq!(register.len(), 8);
        assert_eq!(register.n_qubits(), 3);
        
        let (measurement, after_measurement) = register.measure();
        assert_eq!(measurement, 4);
        assert!(after_measurement.almost_equals(register));
        
    }

    #[test]
    fn test_quantum_register_partially_measures() {
        
        let bases = QuantumRegister::all_bases(4);

        let measured: Vec<u8> = bases.clone().into_iter().map(|x| x.partially_measure(vec![0])).map(|(m, r)| m).collect();
        assert_eq!(measured, iter::repeat(vec![0, 1]).take(8).into_iter().flatten().collect::<Vec<_>>());
        
        let measured: Vec<u8> = bases.clone().into_iter().map(|x| x.partially_measure(vec![2])).map(|(m, r)| m).collect();
        assert_eq!(measured, vec![0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4]);

        let measured: Vec<u8> = bases.clone().into_iter().map(|x| x.partially_measure(vec![1, 3])).map(|(m, r)| m).collect();
        assert_eq!(measured, vec![0, 0, 2, 2, 0, 0, 2, 2, 8, 8, 10, 10, 8, 8, 10, 10]);
    }

    #[test]
    fn test_quantum_register_gets_basis_subspace_indices() {
        assert_eq!(
            QuantumRegister::basis_subspace_indices(1, vec![0]),
            vec![0, 1]
        );
        
        assert_eq!(
            QuantumRegister::basis_subspace_indices(3, vec![1]),
            vec![0, 2]
        );
    }

    #[test]
    fn test_quantum_register_gets_basis_subspace() {
        
        assert!(
            QuantumRegister::basis_subspace(3, vec![0, 1, 2]).into_iter().zip(
                QuantumRegister::all_bases(3)
            ).all(|(x, y)| x.almost_equals(y))
        );

        assert!(
            QuantumRegister::basis_subspace(3, vec![0]).into_iter().zip(
                vec![
                    QuantumRegister::from_int(3, 0),
                    QuantumRegister::from_int(3, 1)
                ]
            ).all(|(x, y)| x.almost_equals(y))
        );
        
        assert!(
            QuantumRegister::basis_subspace(3, vec![1]).into_iter().zip(
                vec![
                    QuantumRegister::from_int(3, 0),
                    QuantumRegister::from_int(3, 2)
                ]
            ).all(|(x, y)| x.almost_equals(y))
        );

        assert!(
            QuantumRegister::basis_subspace(3, vec![0, 1]).into_iter().zip(
                vec![
                    QuantumRegister::from_int(3, 0),
                    QuantumRegister::from_int(3, 1),
                    QuantumRegister::from_int(3, 2),
                    QuantumRegister::from_int(3, 3),
                ]
            ).all(|(x, y)| x.almost_equals(y))
        );
        
        // |00> -> |000>
        // |01> -> |001>
        // |10> -> |100>
        // |11> -> |101>
        assert!(
            QuantumRegister::basis_subspace(3, vec![0, 2]).into_iter().zip(
                vec![
                    QuantumRegister::from_int(3, 0),
                    QuantumRegister::from_int(3, 1),
                    QuantumRegister::from_int(3, 4),
                    QuantumRegister::from_int(3, 5),
                ]
            ).all(|(x, y)| x.almost_equals(y))
        );
    }

    #[test]
    fn test_quantum_register_tensor_products() {

        // |0> x |0> = |00>
        // |0> x |1> = |01>
        // |1> x |0> = |10>
        // |1> x |1> = |11>

        assert!(QuantumRegister::basis(1, 0).tensor_product(&QuantumRegister::basis(1, 0)).almost_equals(QuantumRegister::basis(2, 0)));
        assert!(QuantumRegister::basis(1, 0).tensor_product(&QuantumRegister::basis(1, 1)).almost_equals(QuantumRegister::basis(2, 1)));
        assert!(QuantumRegister::basis(1, 1).tensor_product(&QuantumRegister::basis(1, 0)).almost_equals(QuantumRegister::basis(2, 2)));
        assert!(QuantumRegister::basis(1, 1).tensor_product(&QuantumRegister::basis(1, 1)).almost_equals(QuantumRegister::basis(2, 3)));
        
        let x_0 = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., Complex::exp(TAU * 3./4. * Complex::i())));
        let x_1 = QuantumRegister::singleton(Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., Complex::exp(TAU * 1./2. * Complex::i())));
        
        let actual = x_0.tensor_product(&x_1);
        let expected = QuantumRegister::from_vec_normalize(
            vec![
                x_0.get_coefficient(0) * x_1.get_coefficient(0),
                x_0.get_coefficient(0) * x_1.get_coefficient(1),
                x_0.get_coefficient(1) * x_1.get_coefficient(0),
                x_0.get_coefficient(1) * x_1.get_coefficient(1),
            ]
        );

        assert!(actual.almost_equals(expected));
    }

}