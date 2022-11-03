use std::ops::Mul;

use nalgebra::{UnitVector2, Complex, Vector2};

#[cfg(test)]
mod test_nalgebra {
    use std::f32::consts::SQRT_2;

    use float_cmp::{assert_approx_eq, approx_eq};
    use nalgebra::{Complex, Vector2};

    use super::*;

    #[test]
    fn test_instantiates_complex_vector() {
        let v = UnitVector2::new_normalize(
            Vector2::new(Complex::from(1.0), Complex::from(1.0))
        );
        assert!((v.x.re - 1./SQRT_2).abs() < 0.0001, "{}", v.x.re);
        assert!(v.x.im.abs() < 0.0001, "{}", v.x.im);
        assert!((v.y.re - 1./SQRT_2).abs() < 0.0001, "{}", v.y.re);
        assert!(v.y.im.abs() < 0.0001, "{}", v.y.im);
    }

    #[test]
    fn test_can_multiply_complex_numbers_by_complex_vectors() {

        let unnormed_v = Vector2::new(Complex::from(1.0_f32), Complex::from(2.0_f32));
        let v = UnitVector2::new_normalize(
            unnormed_v
        );
        let normalizer = unnormed_v.norm();

        let v = v.mul(Complex::new(3.0, 2.0));

        assert!((v.x.re - 3./normalizer).abs() < 0.0001, "{}", v.x.re);
        assert!((v.x.im - 2./normalizer).abs() < 0.0001, "{}", v.x.im);
        assert!((v.y.re - 6./normalizer).abs() < 0.0001, "{}", v.y.re);
        assert!((v.y.im - 4./normalizer).abs() < 0.0001, "{}", v.y.im);
        
    }
}