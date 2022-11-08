use std::ops::Mul;

use nalgebra::{UnitVector2, Complex, Vector2, DMatrix, Unit, Scalar, ComplexField, RealField, DVector, Normed};
use num_traits::One;

#[derive(Clone, Debug)]
pub struct SquareMatrix {
    matrix: DMatrix<Complex<f32>>,
}

impl PartialEq for SquareMatrix {
    fn eq(&self, rhs: &Self) -> bool {
        (self.matrix.clone() - rhs.matrix.clone()).norm() < 0.0001
    }
}

impl Mul<SquareMatrix> for SquareMatrix {
    type Output = SquareMatrix;
    
    fn mul(self, rhs: SquareMatrix) -> Self::Output {
        SquareMatrix::new_unitary(self.matrix * rhs.matrix)
    }
}

impl Mul<Unit<DVector<Complex<f32>>>> for SquareMatrix {
    type Output = DVector<Complex<f32>>;

    fn mul(self, rhs: Unit<DVector<Complex<f32>>>) -> Self::Output {
        self.matrix * rhs.into_inner()
    }
}

impl SquareMatrix {
    
    pub fn new_unitary(matrix: DMatrix<Complex<f32>>) -> Self {
        // Determinants are homogenous, meaning that:
        // Det(cA) = c^N * Det(A), where N is the dimension of the matrix
        // => to make Det(cA) = 1, we want Det(cA) = c^N * Det(A) = 1 => c = 1 / Det(A)^(1/N)
        assert!(matrix.is_square());
        let determinant_norm: f32 = matrix.determinant().norm();
        let normalizer = 1./(determinant_norm.powf(1./(matrix.nrows() as f32)));
        let normalized_matrix = matrix.scale(normalizer);
        assert!((normalized_matrix.determinant().norm() - 1.).abs() < 0.0001, "Matrix is not unitary, {}", normalized_matrix.determinant().norm());
        Self {matrix: normalized_matrix }
    }

    pub fn from_vec_normalize(size: usize, vec: Vec<Complex<f32>>) -> Self {
        Self::new_unitary(DMatrix::from_vec(size, size, vec))
    }
    
    pub fn identity(size: usize) -> Self {
        Self::new_unitary(DMatrix::identity(size, size))
    }

    pub fn one(size: usize) -> Self {
        Self::identity(size)
    }

    pub fn almost_equals(&self, rhs: &Self) -> bool {
        (self.matrix.clone() - rhs.matrix.clone()).norm() < 0.0001
    }

    pub fn size(&self) -> usize {
        self.matrix.ncols()
    }

    pub fn get(&self, i: usize, j: usize) -> Complex<f32> {
        let index = i * self.size() + j;
        self.matrix[index]
    }

    pub fn scale(&self, scalar: Complex<f32>) -> Self {
        Self::new_unitary(self.matrix.clone().mul(scalar))
    }
    
    pub fn tensor_product(&self, rhs: &Self) -> Self {
        Self::new_unitary(self.matrix.kronecker(&rhs.matrix.clone()))
    }

    pub fn invert(&self) -> Self {
        Self::new_unitary(self.matrix.clone().try_inverse().expect("All unitary square matrices are invertible"))
    }

    fn is_unitary(&self) -> bool {
        (self.matrix.determinant().norm() - 1.).abs() < 0.0001
    }
    
}

#[cfg(test)]
mod test_nalgebra {
    use std::f32::consts::SQRT_2;

    use float_cmp::{assert_approx_eq, approx_eq};
    use nalgebra::{Complex, Vector2, DMatrix, Normed};
    use num_traits::{Zero, One};

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
    fn test_cnot_matrix_is_already_unitary() {
        let cnot = DMatrix::<Complex<f32>>::from_vec(
            4,
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
        );

        assert!((cnot.determinant().norm() - 1.).abs() < 0.0001, "{}", cnot.determinant().norm());
    }

    #[test]
    fn test_square_matrix_instantiates() {
        
        let m = SquareMatrix::new_unitary(
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
        let result = id.clone() * m.clone();
        assert!(result.is_unitary(), "{}", result.matrix.determinant().norm());
        assert!(m.clone().almost_equals(&(result.clone())), "{:?} * {:?} = {:?}", id, m.clone(), result.clone());
    }

    #[test]
    fn test_square_matrix_tensor_products() {

        let m = SquareMatrix::new_unitary(
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

        let result = id.clone().tensor_product(&m.clone());
        assert!(result.is_unitary(), "{}", result.matrix.determinant().norm());

        let expected = SquareMatrix::new_unitary(
            DMatrix::from_vec(
                4,
                4,
                vec![
                    Complex::from(1.0_f32), Complex::from(2.0_f32), Complex::zero(), Complex::zero(),
                    Complex::from(3.0_f32), Complex::from(4.0_f32), Complex::zero(), Complex::zero(),
                    Complex::zero(), Complex::zero(), Complex::from(1.0_f32), Complex::from(2.0_f32),
                    Complex::zero(), Complex::zero(), Complex::from(3.0_f32), Complex::from(4.0_f32),
                ]
            )
        );

        assert!(result.clone().almost_equals(&(expected.clone())), "{:?} * {:?} = {:?}", id, m.clone(), result.clone());
        
    }
}