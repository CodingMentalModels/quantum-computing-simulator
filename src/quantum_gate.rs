use std::f32::consts::{SQRT_2, TAU};
use std::fmt::{Display, Formatter, Debug};

use nalgebra::{Vector2, DVector, Complex, Unit, Normed, ComplexField};
use num_traits::{One, Zero};
use rand::Rng;

use crate::qubit::{Qubit, Measurement};
use crate::matrix::SquareMatrix;

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
        let new_register = self.register.clone().into_inner() * Complex::exp(phase * Complex::i());
        return Self::new_normalize(new_register);
    }

}

#[derive(Clone, PartialEq)]
pub struct QuantumGate {
    matrix: SquareMatrix,
}

impl Debug for QuantumGate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.matrix)
    }
}

impl Display for QuantumGate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

impl QuantumGate {

    pub fn new(matrix: SquareMatrix) -> Self {
        Self { matrix }
    }

    pub fn get_coefficient(&self, i: usize, j: usize) -> Complex<f32> {
        self.matrix.get_coefficient(i, j)
    }

    pub fn identity(n_qubits: usize) -> Self {
        assert!(n_qubits > 0);
        Self::new(SquareMatrix::one(2usize.pow(n_qubits as u32)))
    }

    pub fn global_rotation(n_qubits: usize, phase: f32) -> Self {
        assert!(n_qubits > 0);
        Self { matrix: SquareMatrix::identity(2usize.pow(n_qubits as u32)).scale(Complex::exp(phase * Complex::i())) }
    }

    pub fn permutation(qubit_permutation: &Vec<usize>) -> Self {
        assert!(qubit_permutation.len() > 0);
        let basis_size = 2usize.pow(qubit_permutation.len() as u32);
        let mut permutation = Vec::new();
        for i in 0..basis_size {
            let mut new_i = 0;
            for (qubit_start, qubit_end) in qubit_permutation.iter().enumerate() {
                let start_value = i & (1 << qubit_start as u32);
                let start_bit = start_value >> (qubit_start as u32);
                assert!(start_bit == 0 || start_bit == 1);
                let end_value = (start_bit << qubit_end);
                new_i = new_i | end_value;
            }
            permutation.push(new_i);
        }
        Self::new(SquareMatrix::permutation(permutation))
    }

    pub fn reverse_permutation(reverse_qubit_permutation: &Vec<usize>) -> Self {
        let mut qubit_permutation = vec![0; reverse_qubit_permutation.len()];
        for (i, &j) in reverse_qubit_permutation.iter().enumerate() {
            qubit_permutation[j] = i;
        }
        Self::permutation(&qubit_permutation)
    }

    pub fn tensor_product(&self, rhs: Self) -> Self {
        Self::new(self.matrix.tensor_product(&rhs.matrix))
    }

    pub fn compose(&self, rhs: &Self) -> Self {
        Self::new(self.matrix.clone() * rhs.matrix.clone())
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

    pub fn controlled_phase_shift(phase: f32) -> Self {
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
                Complex::one(), // 22
                Complex::zero(),
                Complex::zero(), // 30
                Complex::zero(),
                Complex::zero(),
                Complex::exp(phase * Complex::i()), // 33
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

    pub fn reverse(&self) -> Self {
        Self::new(self.matrix.clone().invert())
    }

    pub fn swap_bases(&mut self, i: usize, j: usize) {
        let new_matrix = self.matrix.clone().swap_columns(i, j);
        self.matrix = new_matrix;
    }

}


#[cfg(test)]
mod test_quantum_gate {

    use std::iter;

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

        let mixed_state = Qubit::mix(basis_0, basis_1, 1.0_f32.sqrt(), 3.0_f32.sqrt());
        let result = gate.apply(QuantumRegister::singleton(mixed_state));

        assert!(result.almost_equals(QuantumRegister::singleton(mixed_state)), "{:?} != {:?}", result, QuantumRegister::singleton(mixed_state));

        let hadamard_gate = QuantumGate::hadamard();

        assert!(
            hadamard_gate.apply(QuantumRegister::singleton(basis_0)).almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one(), Complex::one())))
            )
        );
        assert!(
            hadamard_gate.apply(QuantumRegister::singleton(basis_1)).almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one(), -Complex::one())))
            )
        );

        let result = hadamard_gate.apply(QuantumRegister::singleton(mixed_state));
        assert!(
            result.almost_equals(
                QuantumRegister::singleton(Qubit::new_normalize(Vector2::new(Complex::one() * (0.25_f32.sqrt() + 0.75_f32.sqrt()), Complex::one() * (0.25_f32.sqrt() - 0.75_f32.sqrt()))))
            ),
            "{}",
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
    fn test_identity_gate() {
        let gate = QuantumGate::identity(8);
        assert_eq!(gate.n_qubits(), 8);
        assert!(gate.apply(QuantumRegister::basis(8, 0)).almost_equals(QuantumRegister::basis(8, 0)));
        assert!(gate.apply(QuantumRegister::basis(8, 1)).almost_equals(QuantumRegister::basis(8, 1)));
        assert!(gate.apply(QuantumRegister::basis(8, 2)).almost_equals(QuantumRegister::basis(8, 2)));
        assert!(gate.apply(QuantumRegister::basis(8, 3)).almost_equals(QuantumRegister::basis(8, 3)));
        assert!(gate.apply(QuantumRegister::basis(8, 4)).almost_equals(QuantumRegister::basis(8, 4)));
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
        
        // |00> -> |00>
        // |01> -> |01>
        // |10> -> |11>
        // |11> -> |10>
        assert_eq!(cnot.clone().apply(zero_zero.clone()), zero_zero.clone());
        assert_eq!(cnot.clone().apply(zero_one.clone()), zero_one.clone());
        assert_eq!(cnot.clone().apply(one_zero.clone()), one_one.clone());
        assert_eq!(cnot.clone().apply(one_one.clone()), one_zero.clone());
    }

    #[test]
    fn test_controlled_phase_shift_gate() {
        assert!(
            QuantumGate::controlled_phase_shift(TAU).almost_equals(&QuantumGate::identity(2))
        );
        let cphase = QuantumGate::controlled_phase_shift(TAU / 4.);

        let expected = QuantumGate::new(
            SquareMatrix::from_vec_normalize(4, vec![
                Complex::one(), Complex::zero(), Complex::zero(), Complex::zero(),
                Complex::zero(), Complex::one(), Complex::zero(), Complex::zero(),
                Complex::zero(), Complex::zero(), Complex::one(), Complex::zero(),
                Complex::zero(), Complex::zero(), Complex::zero(), Complex::i(),
            ])
        );
        assert!(
            cphase.clone().almost_equals(&expected),
            "{}",
            cphase
        );

        let swap_gate = QuantumGate::permutation(&vec![1, 0]);
        let reversed_cphase = swap_gate.compose(
            &QuantumGate::controlled_phase_shift(TAU / 2.)
                .compose(&swap_gate));
        assert!(
            QuantumGate::controlled_phase_shift(TAU / 2.).almost_equals(&reversed_cphase)
        )

    }

    #[test]
    fn test_quantum_gate_swaps_bases() {
        
        let mut gate = QuantumGate::cnot();
        gate.swap_bases(0, 0);
        assert!(gate.almost_equals(&QuantumGate::cnot()));

        gate.swap_bases(0, 1);

        let zero_zero = QuantumRegister::basis(2, 0);
        let zero_one = QuantumRegister::basis(2, 1);
        let one_one = QuantumRegister::basis(2, 2);
        let one_zero = QuantumRegister::basis(2, 3);

        assert!(gate.apply(zero_zero.clone()).almost_equals(zero_one.clone()));
        assert!(gate.apply(zero_one.clone()).almost_equals(zero_zero.clone()));
        assert!(gate.apply(one_zero.clone()).almost_equals(one_one.clone()));
        assert!(gate.apply(one_one.clone()).almost_equals(one_zero.clone()));
    }

    #[test]
    fn test_permutation_matrix() {
        assert!(QuantumGate::permutation(&vec![0, 1]).almost_equals(&QuantumGate::identity(2)));
        assert_eq!(QuantumGate::permutation(&vec![0, 1, 2]).n_qubits(), 3);
        assert_eq!(QuantumGate::identity(3).n_qubits(), 3);
        assert!(QuantumGate::permutation(&vec![0, 1, 2]).almost_equals(&QuantumGate::identity(3)));

        // |00> -> |00>
        // |01> -> |10>
        // |10> -> |01>
        // |11> -> |11>
        assert!(QuantumGate::permutation(&vec![1, 0]).almost_equals(
            &QuantumGate::new(
                SquareMatrix::from_vec_normalize(4, vec![
                    Complex::one(), Complex::zero(), Complex::zero(), Complex::zero(),
                    Complex::zero(), Complex::zero(), Complex::one(), Complex::zero(),
                    Complex::zero(), Complex::one(), Complex::zero(), Complex::zero(),
                    Complex::zero(), Complex::zero(), Complex::zero(), Complex::one(),
                ])
            )
        ));

        
        let rotation = QuantumGate::permutation(&vec![2, 0, 1]);
        // |000> -> |000>
        // |001> -> |010>
        // |010> -> |100>
        // |011> -> |110>
        // |100> -> |001>
        // |101> -> |011>
        // |110> -> |101>
        // |111> -> |111>

        assert!(rotation.apply(QuantumRegister::basis(3, 0)).almost_equals(QuantumRegister::basis(3, 0)));
        assert!(rotation.apply(QuantumRegister::basis(3, 1)).almost_equals(QuantumRegister::basis(3, 2)));
        assert!(rotation.apply(QuantumRegister::basis(3, 2)).almost_equals(QuantumRegister::basis(3, 4)));
        assert!(rotation.apply(QuantumRegister::basis(3, 3)).almost_equals(QuantumRegister::basis(3, 6)));
        assert!(rotation.apply(QuantumRegister::basis(3, 4)).almost_equals(QuantumRegister::basis(3, 1)));
        assert!(rotation.apply(QuantumRegister::basis(3, 5)).almost_equals(QuantumRegister::basis(3, 3)));
        assert!(rotation.apply(QuantumRegister::basis(3, 6)).almost_equals(QuantumRegister::basis(3, 5)));
        assert!(rotation.apply(QuantumRegister::basis(3, 7)).almost_equals(QuantumRegister::basis(3, 7)));
        
    }

    #[test]
    fn test_reverse_permutation_matrix_is_inverse_of_permutation_matrix() {
        
        let permutation = vec![3, 1, 2, 0];

        let expected = QuantumGate::permutation(&permutation).reverse();

        assert!(QuantumGate::reverse_permutation(&permutation).almost_equals(&expected));
    }

    #[test]
    fn test_quantum_gates_compose() {
        
        let n_qubits = 12;
        let permutation = (0..n_qubits).collect();
        let permutation_gate = QuantumGate::permutation(&permutation);
        let reverse_permutation_gate = QuantumGate::reverse_permutation(&permutation);
        let actual = permutation_gate.clone().compose(&reverse_permutation_gate.clone());

        let identity = QuantumGate::identity(n_qubits);
        assert!(actual.almost_equals(&identity));

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