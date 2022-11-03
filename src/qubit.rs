use std::ops::{Mul, Add};

use nalgebra::{UnitVector2, Complex, Vector2};
use num_traits::identities::One;
use num_traits::Zero;


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

    pub fn basis_0() -> Self {
        Self::new(UnitVector2::new_normalize(Vector2::new(Complex::one(), Complex::zero())))
    }

    pub fn basis_1() -> Self {
        Self::new(UnitVector2::new_normalize(Vector2::new(Complex::zero(), Complex::one())))
    }

    pub fn mix(lhs: Self, rhs: Self, lhs_weight: f32, rhs_weight: f32) -> Self {
        let lhs_coefficient = (lhs_weight / (lhs_weight + rhs_weight)).sqrt();
        let rhs_coefficient = (rhs_weight / (lhs_weight + rhs_weight)).sqrt();
        Self::new_normalize(lhs.state.into_inner().mul(Complex::one() * lhs_coefficient) + rhs.state.into_inner().mul(Complex::one() * rhs_coefficient))
    }

    pub fn almost_equals(&self, rhs: &Self) -> bool {
        (self.state.into_inner() - rhs.state.into_inner()).norm() < 0.0001
    }

    pub fn get_state(&self) -> UnitVector2<Complex<f32>> {
        self.state
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
}