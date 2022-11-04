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

            assert!(after_measurement.almost_equals(&Qubit::basis_0()) || after_measurement.almost_equals(&Qubit::basis_1()));

            let (second_measurement, after_second_measurement) = hadamard_gate.apply(after_measurement).measure();

            assert!(after_second_measurement.almost_equals(&Qubit::basis_0()) || after_second_measurement.almost_equals(&Qubit::basis_1()));
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

        assert!(double_hadamarded_0.almost_equals(&Qubit::basis_0()));
        assert_eq!(measurement, 0);
        assert!(after_measurement.almost_equals(&Qubit::basis_0()));
        

    }
}