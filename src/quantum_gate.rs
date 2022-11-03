use std::f32::consts::SQRT_2;

use nalgebra::{Matrix2, Vector2, Complex, Unit};
use num_traits::One;

use crate::qubit::Qubit;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct QuantumGate {
    matrix: Unit<Matrix2<Complex<f32>>>,
}

impl QuantumGate {

    pub fn new(matrix: Unit<Matrix2<Complex<f32>>>) -> Self {
        Self { matrix }
    }

    pub fn new_normalize(matrix: Matrix2<Complex<f32>>) -> Self {
        Self::new(Unit::new_normalize(matrix))
    }

    pub fn identity() -> Self {
        Self::new_normalize(Matrix2::one())
    }

    pub fn hadamard() -> Self {
        Self::new_normalize(Matrix2::new(
            Complex::one() * 1./SQRT_2, Complex::one() * 1./SQRT_2,
            Complex::one() * 1./SQRT_2, Complex::one() * -1./SQRT_2,
        ))
    }

    pub fn approx_equals(&self, rhs: &Self) -> bool {
        (self.matrix.into_inner() - rhs.matrix.into_inner()).norm() < 0.0001
    }

    pub fn apply(&self, qubit: Qubit) -> Qubit {
        Qubit::new_normalize(self.matrix.into_inner() * qubit.get_state().into_inner())
    }

}


#[cfg(test)]
mod test_quantum_gate {

    use super::*;

    #[test]
    fn test_quantum_gate_initializes() {
        
        let gate = QuantumGate::new_normalize(
            Matrix2::one()
        );

        let identity_gate = QuantumGate::identity();
        assert!(gate.approx_equals(&identity_gate));

        let basis_0 = Qubit::basis_0();
        let basis_1 = Qubit::basis_1();

        let mixed_state = Qubit::mix(basis_0, basis_1, 1., 3.);
        let result = gate.apply(mixed_state);

        assert!(result.almost_equals(&mixed_state));

        let hadamard_gate = QuantumGate::hadamard();

        assert!(
            hadamard_gate.apply(basis_0).almost_equals(
                &Qubit::new_normalize(Vector2::new(Complex::one() * 0.5_f32.sqrt(), Complex::one() * 0.5_f32.sqrt()))
            )
        );
        assert!(
            hadamard_gate.apply(basis_1).almost_equals(
                &Qubit::new_normalize(Vector2::new(Complex::one() * 0.5_f32.sqrt(), -Complex::one() * 0.5_f32.sqrt()))
            )
        );

        let result = hadamard_gate.apply(mixed_state);
        assert!(
            result.almost_equals(
                &Qubit::new_normalize(Vector2::new(Complex::one() * (0.25_f32.sqrt() + 0.75_f32.sqrt()), Complex::one() * (0.25_f32.sqrt() - 0.75_f32.sqrt())))
            ),
            "{:?}",
            result,
        );

    }
}