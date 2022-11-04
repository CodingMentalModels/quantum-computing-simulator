use std::ops::Mul;

use nalgebra::{UnitVector2, Complex, Vector2, DMatrix, Unit, Scalar, ComplexField, RealField, DVector};

#[derive(Clone, Debug)]
pub struct SquareMatrix {
    matrix: Unit<DMatrix<Complex<f32>>>,
}

impl PartialEq for SquareMatrix {
    fn eq(&self, rhs: &Self) -> bool {
        (self.matrix.clone().into_inner() - rhs.matrix.clone().into_inner()).norm() < 0.0001
    }
}

impl Mul<Unit<DVector<Complex<f32>>>> for SquareMatrix {
    type Output = DVector<Complex<f32>>;

    fn mul(self, rhs: Unit<DVector<Complex<f32>>>) -> Self::Output {
        self.matrix.into_inner() * rhs.into_inner()
    }
}

impl SquareMatrix {
    
    pub fn new(matrix: Unit<DMatrix<Complex<f32>>>) -> Self {
        Self { matrix }
    }
    
    pub fn new_normalize(matrix: DMatrix<Complex<f32>>) -> Self {
        Self::new(Unit::<DMatrix<Complex<f32>>>::new_normalize(matrix))
    }

    pub fn from_vec_normalize(size: usize, vec: Vec<Complex<f32>>) -> Self {
        Self::new_normalize(DMatrix::from_vec(size, size, vec))
    }
    
    pub fn identity(size: usize) -> Self {
        Self::new_normalize(DMatrix::identity(size, size))
    }

    pub fn one(size: usize) -> Self {
        Self::identity(size)
    }

    pub fn almost_equals(&self, rhs: &Self) -> bool {
        (self.matrix.clone().into_inner() - rhs.matrix.clone().into_inner()).norm() < 0.0001
    }

    pub fn size(&self) -> usize {
        self.matrix.ncols()
    }
    
}

impl Mul<SquareMatrix> for SquareMatrix {
    type Output = SquareMatrix;
    
    fn mul(self, rhs: SquareMatrix) -> Self::Output {
        SquareMatrix::new_normalize(self.matrix.into_inner() * rhs.matrix.into_inner())
    }
}

#[cfg(test)]
mod test_nalgebra {
    use std::f32::consts::SQRT_2;

    use float_cmp::{assert_approx_eq, approx_eq};
    use nalgebra::{Complex, Vector2, DMatrix};

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

    #[test]
    fn test_square_matrix_instantiates() {
        
        let m = SquareMatrix::new_normalize(
            DMatrix::from_vec(
                2,
                2,
                vec![
                    Complex::from(1.0_f32), Complex::from(2.0_f32),
                    Complex::from(3.0_f32), Complex::from(4.0_f32),
                ]
            )
        );

        let id = SquareMatrix::one(2);
        assert!(m.clone().almost_equals(&(id * m)));
    }
}