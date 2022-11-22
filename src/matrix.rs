use std::{ops::Mul, fmt::{Debug, Display, Formatter}, collections::HashMap};

use nalgebra::{Complex, ComplexField, DMatrix, Unit, DVector, Normed};
use num_traits::{One, Zero};
use std::f32::consts::SQRT_2;

type SparseMatrixRepresentation = HashMap<usize, HashMap<usize, Complex<f32>>>;

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    size: usize,
    data: SparseMatrixRepresentation,
}

impl From<DMatrix<Complex<f32>>> for SparseMatrix {
    fn from(matrix: DMatrix<Complex<f32>>) -> Self {
        let mut sparse_matrix: SparseMatrixRepresentation = HashMap::new();
        for (i, row) in matrix.row_iter().enumerate() {
            for (j, coefficient) in row.iter().enumerate() {
                if !coefficient.is_zero() {
                    if sparse_matrix.contains_key(&i) {
                        sparse_matrix.get_mut(&i).unwrap().insert(j, *coefficient);
                    } else {
                        let mut row = HashMap::new();
                        row.insert(j, *coefficient);
                        sparse_matrix.insert(i, row);
                    }
                }
            }
        }
        SparseMatrix::new(matrix.nrows(), sparse_matrix)
    }
}

impl From<SparseMatrix> for DMatrix<Complex<f32>> {
    fn from(sparse_matrix: SparseMatrix) -> Self {
        DMatrix::from_fn(
            sparse_matrix.size,
            sparse_matrix.size,
            |i, j| {
                if sparse_matrix.data.contains_key(&i) {
                    if sparse_matrix.data.get(&i).unwrap().contains_key(&j) {
                        return *sparse_matrix.data.get(&i).unwrap().get(&j).unwrap();
                    }
                }
                Complex::zero()
            }
        )
    }
}

impl Mul<SparseMatrix> for SparseMatrix {
    type Output = SparseMatrix;

    fn mul(self, rhs: SparseMatrix) -> Self::Output {
        let mut result: SparseMatrixRepresentation = HashMap::new();
        for (i, row) in self.data.iter() {
            for (j, lhs_coefficient) in row.iter() {
                if rhs.data.contains_key(j) {
                    for (k, rhs_coefficient) in rhs.data.get(j).unwrap().iter() {
                        if result.contains_key(i) {
                            if result.get_mut(i).unwrap().contains_key(k) {
                                *result.get_mut(i).unwrap().get_mut(k).unwrap() += lhs_coefficient * rhs_coefficient;
                            } else {
                                result.get_mut(i).unwrap().insert(*k, lhs_coefficient * rhs_coefficient);
                            }
                        } else {
                            let mut new_row = HashMap::new();
                            new_row.insert(*k, lhs_coefficient * rhs_coefficient);
                            result.insert(*i, new_row);
                        }
                    }
                }
            }
        }
        SparseMatrix::new(self.size, result)
    }

}

impl SparseMatrix {

    pub fn new(size: usize, data: SparseMatrixRepresentation) -> Self {
        SparseMatrix {
            size,
            data,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> Complex<f32> {
        match self.data.get(&i) {
            Some(row) => match row.get(&j) {
                Some(coefficient) => *coefficient,
                None => Complex::zero(),
            },
            None => Complex::zero(),
        }
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn identity(size: usize) -> Self {
        DMatrix::identity(size, size).into()
    }

    pub fn almost_equals(&self, other: &Self) -> bool {
        for (i, row) in self.data.iter() {
            for (j, coefficient) in row.iter() {
                if !((other.get(*i, *j) - coefficient).abs() < 0.0001) {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Clone)]
pub struct SquareMatrix {
    matrix: DMatrix<Complex<f32>>,
}

impl Display for SquareMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

impl Debug for SquareMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

impl PartialEq for SquareMatrix {
    fn eq(&self, rhs: &Self) -> bool {
        (self.matrix.clone() - rhs.matrix.clone()).norm() < 0.0001
    }
}

impl Mul<SquareMatrix> for SquareMatrix {
    type Output = SquareMatrix;
    
    fn mul(self, rhs: SquareMatrix) -> Self::Output {
        SquareMatrix::new_unchecked(self.matrix * rhs.matrix)
    }
}

impl Mul<Unit<DVector<Complex<f32>>>> for SquareMatrix {
    type Output = DVector<Complex<f32>>;

    fn mul(self, rhs: Unit<DVector<Complex<f32>>>) -> Self::Output {
        self.matrix * rhs.into_inner()
    }
}

impl SquareMatrix {

    pub fn new_unchecked(matrix: DMatrix<Complex<f32>>) -> Self {
        Self { matrix }
    }
    
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

    pub fn get_coefficient(&self, row: usize, column: usize) -> Complex<f32> {
        self.matrix[(row, column)]
    }
    
    pub fn identity(size: usize) -> Self {
        Self::new_unchecked(DMatrix::identity(size, size))
    }

    pub fn one(size: usize) -> Self {
        Self::identity(size)
    }

    pub fn permutation(permutation: Vec<usize>) -> Self {
        assert!(permutation.len() > 0);
        assert!(permutation.iter().zip(permutation.iter().skip(1)).all(|(a, b)| a != b));

        let size = permutation.len();

        let mut matrix = DMatrix::from_element(size, size, Complex::zero());
        for (i, j) in permutation.iter().enumerate() {
            matrix[(i, *j)] = Complex::one();
        }
        Self::new_unchecked(matrix)
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
        Self::new_unchecked(self.matrix.kronecker(&rhs.matrix.clone()))
    }

    pub fn invert(&self) -> Self {
        Self::new_unchecked(self.matrix.clone().try_inverse().expect("All unitary square matrices are invertible"))
    }

    pub fn swap_columns(&self, i: usize, j: usize) -> Self {
        assert!(i < self.size());
        assert!(j < self.size());

        let mut new_matrix = self.matrix.clone();
        new_matrix.swap_columns(i, j);
        Self::new_unitary(new_matrix)
    }

    fn is_unitary(&self) -> bool {
        (self.matrix.determinant().norm() - 1.).abs() < 0.0001
    }
    
}

#[cfg(test)]
mod test_nalgebra {

    use super::*;

    #[test]
    fn test_square_matrix_identity() {
        
        let matrix = SquareMatrix::identity(2usize.pow(5));
        assert_eq!(matrix.size(), 2usize.pow(5));

        let v = Unit::<DVector<Complex<f32>>>::new_normalize(DVector::from_element(2usize.pow(5), Complex::one()));
        assert!(Unit::<DVector<_>>::new_normalize(matrix * v.clone()).norm() - v.clone().norm() < 0.0001);

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

    #[test]
    fn test_swaps_columns() {
        
        let matrix = SquareMatrix::new_unitary(
            DMatrix::from_vec(
                2,
                2,
                vec![
                    Complex::from(1.0_f32),
                    Complex::from(3.0_f32),
                    Complex::from(2.0_f32),
                    Complex::from(4.0_f32),
                ]
            )
        );

        assert!(
            matrix.clone().swap_columns(0, 1).almost_equals(
                &SquareMatrix::new_unitary(
                    DMatrix::from_vec(
                        2,
                        2,
                        vec![
                            Complex::from(2.0_f32),
                            Complex::from(4.0_f32),
                            Complex::from(1.0_f32),
                            Complex::from(3.0_f32),
                        ]
                    )
                )
            ),
            "{}",
            matrix.swap_columns(0, 1)
        );

    }

    #[test]
    fn test_sparse_matrix_from_dmatrix() {
        
        let matrix = DMatrix::from_vec(
            2,
            2,
            vec![
                Complex::from(1.0_f32),
                Complex::from(3.0_f32),
                Complex::from(2.0_f32),
                Complex::from(4.0_f32),
            ]
        );

        let sparse = SparseMatrix::from(matrix.clone());

        assert_eq!(sparse.get_size(), 2);
        assert_eq!(sparse.get(0, 0), Complex::from(1.0_f32));
        assert_eq!(sparse.get(0, 1), Complex::from(2.0_f32));
        assert_eq!(sparse.get(1, 0), Complex::from(3.0_f32));
        assert_eq!(sparse.get(1, 1), Complex::from(4.0_f32));

        let original = DMatrix::<_>::from(sparse);
        assert!((original - matrix).norm() < 0.0001);

    }

    #[test]
    fn test_sparse_matrix_multiplication() {

        let lhs = SparseMatrix::from(
            DMatrix::from_vec(
                2,
                2,
                vec![
                    Complex::from(1.0_f32),
                    Complex::from(3.0_f32),
                    Complex::from(2.0_f32),
                    Complex::from(4.0_f32),
                ]
            )
        );
        let rhs = SparseMatrix::identity(2);

        // 1 2
        // 3 4

        // 5 6
        // 7 8

        // 23 22
        // 43 50

        let result = lhs.clone() * rhs.clone();
        assert!(result.almost_equals(&lhs));

        let rhs = SparseMatrix::from(
            DMatrix::from_vec(
                2,
                2,
                vec![
                    Complex::from(5.0_f32),
                    Complex::from(7.0_f32),
                    Complex::from(6.0_f32),
                    Complex::from(8.0_f32),
                ]
            )
        );

        let result = lhs.clone() * rhs.clone();
        assert!(
            result.almost_equals(
                &SparseMatrix::from(
                    DMatrix::from_vec(
                        2,
                        2,
                        vec![
                            Complex::from(19.0_f32),
                            Complex::from(43.0_f32),
                            Complex::from(22.0_f32),
                            Complex::from(50.0_f32),
                        ]
                    )
                )
            ),
            "{}",
            DMatrix::from(result)
        );
        
    }

    #[test]
    fn test_sparse_matrix_multiplication_big() {
        
        let size = 2usize.pow(10);
        let lhs = SparseMatrix::from(DMatrix::from_diagonal(&DVector::from_vec((0..size).map(|x| Complex::from(x as f32)).collect())));

        let result = lhs.clone() * lhs;

        assert_eq!(result.get_size(), size);
    }
}