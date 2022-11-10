use std::ops::{Mul};

use nalgebra::ComplexField;

use nalgebra::{UnitVector2, Complex, Vector2};
use num_traits::identities::One;
use num_traits::Zero;

use rand::prelude::*;

pub type Measurement = u8;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Qubit {
    state: UnitVector2<Complex<f32>>,
}

impl Qubit {
    
    pub fn new(state: UnitVector2<Complex<f32>>) -> Self {
        Self { state }
    }

    pub fn new_normalize(state: Vector2<Complex<f32>>) -> Self {
        Self::new(UnitVector2::new_normalize(state))
    }

    pub fn from_coefficients_normalize(a: Complex<f32>, b: Complex<f32>) -> Self {
        Self::new_normalize(Vector2::new(a, b))
    }

    pub fn basis_0() -> Self {
        Self::new(UnitVector2::new_normalize(Vector2::new(Complex::one(), Complex::zero())))
    }

    pub fn basis_1() -> Self {
        Self::new(UnitVector2::new_normalize(Vector2::new(Complex::zero(), Complex::one())))
    }

    pub fn mix(lhs: Self, rhs: Self, lhs_weight: impl Into<Complex<f32>> + Copy, rhs_weight: impl Into<Complex<f32>> + Copy) -> Self {
        let lhs_coefficient = (lhs_weight.into() / (lhs_weight.into() + rhs_weight.into())).sqrt();
        let rhs_coefficient = (rhs_weight.into() / (lhs_weight.into() + rhs_weight.into())).sqrt();
        Self::new_normalize(lhs.state.into_inner().mul(lhs_coefficient) + rhs.state.into_inner().mul(rhs_coefficient))
    }

    pub fn almost_equals(&self, rhs: &Self) -> bool {
        (self.state.into_inner() - rhs.state.into_inner()).norm() < 0.0001
    }

    pub fn get_state(&self) -> UnitVector2<Complex<f32>> {
        self.state
    }

    pub fn measure(&self) -> (Measurement, Self) {
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(0.0_f32..1.0);
        let probability_of_0 = self.state.into_inner().x.abs().powi(2);
        if random_number < probability_of_0 {
            (0, Self::basis_0())
        } else {
            (1, Self::basis_1())
        }
    }

}

#[cfg(test)]
mod test_qubit {

    use super::*;

    #[test]
    fn test_qubit_initializes() {
        
        assert_eq!(Qubit::basis_0(), Qubit::new(UnitVector2::new_normalize(Vector2::new(Complex::one(), Complex::zero()))));
        assert_eq!(Qubit::basis_1(), Qubit::new(UnitVector2::new_normalize(Vector2::new(Complex::zero(), Complex::one()))));

    }

    #[test]
    fn test_qubit_mixes() {

        let mixed_state = Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.);

        assert!(mixed_state.almost_equals(&Qubit::new(UnitVector2::new_normalize(Vector2::new(Complex::one() * 0.25_f32.sqrt(), Complex::one() * 0.75_f32.sqrt())))));
        
    }

    #[test]
    fn test_qubit_measures() {
        
        let (basis_0_measurement, new_basis_0_qubit) = Qubit::basis_0().measure();
        assert_eq!(basis_0_measurement, 0);
        assert_eq!(new_basis_0_qubit, Qubit::basis_0());

        let (basis_1_measurement, new_basis_1_qubit) = Qubit::basis_1().measure();
        assert_eq!(basis_1_measurement, 1);
        assert_eq!(new_basis_1_qubit, Qubit::basis_1());

        let mixed_state = Qubit::mix(Qubit::basis_0(), Qubit::basis_1(), 1., 3.);

        let (mixed_state_measurement, new_mixed_state_qubit) = mixed_state.measure();

        assert!(mixed_state_measurement == 0 || mixed_state_measurement == 1);
        assert!(new_mixed_state_qubit.almost_equals(&Qubit::basis_0()) || new_mixed_state_qubit.almost_equals(&Qubit::basis_1()));

        let mut count_0s = 0;
        let mut count_1s = 0;
        for _ in 0..1000 {
            let (mixed_state_measurement, new_mixed_state_qubit) = mixed_state.measure();
            if mixed_state_measurement == 0 {
                count_0s += 1;
            } else {
                count_1s += 1;
            }
        }

        assert!(count_0s > 200); // Expect 1/4 = 250, so this should be pretty safe
        assert!(count_1s > 650); // Expect 3/4 = 750, so this should be pretty safe

    }

}