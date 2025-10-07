use std::ops::{Add, Mul};
use num_traits::{CheckedAdd, CheckedMul, Num, One, Zero};
use crate::matrix::{Matrix, MatrixError};

// Dynamic runtime matrix.
pub struct DMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> DMatrix<T> {
    // Constructors.
    
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> DMatrix<T> {
        assert_eq!(rows * cols, data.len(), "DMatrix::new: data length does not match dimensions.");
        DMatrix { rows, cols, data }
    }
    
    pub fn from_val(rows: usize, cols: usize, val: T) -> DMatrix<T> where T: Clone {
        let data = vec![val; rows * cols];
        DMatrix { rows, cols, data }
    }
    
    pub fn zeros(rows: usize, cols: usize) -> DMatrix<T> where T: Zero + Clone {
        Self::from_val(rows, cols, T::zero())
    }
    
    pub fn identity(n: usize) -> DMatrix<T> where T: Zero + One + Clone {
        // make an n x n all zero matrix.
        let mut res = Self::zeros(n, n);
        for i in 0..n {
            res.data[i * n + i] = T::one();
        };
        res
    }

    // Getters

    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    // Math
}

impl<T> Matrix for DMatrix<T> {
    type Item = T;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i >= self.rows || j >= self.cols {
            None
        } else {
            Some(&self.data[i * self.cols + j])
        }
    }

    fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i >= self.rows || j >= self.cols {
            None
        } else {
            Some(&mut self.data[i * self.cols + j])
        }
    }

    fn set(&mut self, i: usize, j: usize, value: &T) -> Result<(), MatrixError> where T: Clone {
        if i >= self.rows || j >= self.cols {
            Err(MatrixError::OutOfBounds(i, j, self.rows, self.cols))
        } else {
            self.data[i * self.cols + j] = value.clone();
            Ok(())
        }
    }
}

impl<T> Add<&Self> for &DMatrix<T> where for <'a> &'a T: Add<&'a T, Output = T> {
    type Output = DMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("DMatrix dimensions do not match for addition.")
    }
}

impl<T> CheckedAdd for DMatrix<T>
    where for <'a> &'a T: Add<&'a T, Output = T>{
    fn checked_add(&self, other: &Self) -> Option<Self> {
        if (self.rows != other.rows || self.cols != other.cols) {
            None
        } else {
            let data = self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            Some(DMatrix { rows: self.rows, cols: self.cols, data })
        }
    }
}

impl<T> DMatrix<T> {
    pub fn mul_scalar<K: Copy>(&self, k: K) -> DMatrix<T> where T: Mul<K, Output = T> + Clone {
        let data = self.data.iter()
            .map(|a| a.clone() * k)
            .collect();
        DMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul<M, O>(&self, rhs: &M) -> Result<DMatrix<O>, MatrixError>
    where
        M: Matrix,
        for<'a> &'a T: Mul<&'a M::Item, Output = O>,
        O: Add<Output = O> + Zero
    {
        if self.cols != rhs.rows() {
            return Err(MatrixError::DimensionMismatch(self.rows, self.cols, rhs.rows(), rhs.cols()));
        }
        // actually do matrix mult
        // initiate a resulting vec we will just flat store stuff. as ez to calculate.
        let mut vec = Vec::<O>::with_capacity(self.rows * rhs.cols());
        for i in 0..self.rows {
            for j in 0..rhs.cols() {
                let mut sum = O::zero();
                for k in 0..self.cols() {
                    let a = self.get(i, j).unwrap();
                    let b = rhs.get(k, j).unwrap();
                    sum = sum + (a * b)
                }
                vec.push(sum);
            }
        }
        Ok(DMatrix {rows: self.rows, cols: rhs.cols(), data: vec})
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Imports DMatrix, Matrix trait, etc.

    #[test]
    fn test_new_constructor() {
        let m = DMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic]
    fn test_new_constructor_panic() {
        // Data length (5) does not match rows * cols (6)
        DMatrix::new(2, 3, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_zeros_constructor() {
        let m: DMatrix<i32> = DMatrix::zeros(2, 2);
        assert_eq!(m.data, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_identity_constructor() {
        let m = DMatrix::<i32>::identity(3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_get_and_set() {
        let mut m = DMatrix::new(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(m.get(0, 1), Some(&2));
        assert_eq!(m.get(1, 0), Some(&3));
        assert_eq!(m.get(2, 0), None); // Out of bounds

        let res = m.set(1, 1, &99);
        assert!(res.is_ok());
        assert_eq!(m.get(1, 1), Some(&99));

        let err_res = m.set(2, 2, &0);
        assert!(err_res.is_err());
    }

    #[test]
    fn test_checked_add() {
        let a = DMatrix::new(2, 2, vec![1, 2, 3, 4]);
        let b = DMatrix::new(2, 2, vec![5, 6, 7, 8]);
        let result = a.checked_add(&b).unwrap();
        assert_eq!(result.data, vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_checked_add_mismatched_dims() {
        let a = DMatrix::new(2, 2, vec![1, 2, 3, 4]);
        let b = DMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let result = a.checked_add(&b);
        assert!(result.is_none());
    }

    #[test]
    fn test_add_operator() {
        let a = DMatrix::new(2, 2, vec![1, 2, 3, 4]);
        let b = DMatrix::new(2, 2, vec![5, 6, 7, 8]);
        // This tests the `Add` trait impl for references
        let result = &a + &b;
        assert_eq!(result.data, vec![6, 8, 10, 12]);
        // This tests the `Add` trait impl for owned values
        let result2 = a + b;
        assert_eq!(result2.data, vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_mul_scalar() {
        let m = DMatrix::new(2, 2, vec![1, -2, 3, -4]);
        let result = m.mul_scalar(3);
        assert_eq!(result.data, vec![3, -6, 9, -12]);
    }

    #[test]
    fn test_matrix_multiplication() {
        // A is 2x3
        let a = DMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        // B is 3x2
        let b = DMatrix::new(3, 2, vec![7, 8, 9, 10, 11, 12]);
        // Result should be 2x2
        let result = a.mul(&b).unwrap();

        // Expected result:
        // [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ] = [ 58, 64 ]
        // [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ] = [ 139, 154 ]
        let expected = DMatrix::new(2, 2, vec![58, 64, 139, 154]);

        assert_eq!(result.rows, expected.rows);
        assert_eq!(result.cols, expected.cols);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_mul_by_identity() {
        let m = DMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let identity = DMatrix::<i32>::identity(3);
        let result = m.mul(&identity).unwrap();

        assert_eq!(result.data, m.data);
    }

    #[test]
    fn test_mul_mismatched_dims() {
        let a = DMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let b = DMatrix::new(2, 2, vec![1, 2, 3, 4]); // Inner dimensions 3 != 2
        let result = a.mul(&b);

        assert!(result.is_err());
        match result.err().unwrap() {
            MatrixError::DimensionMismatch(r1, c1, r2, c2) => {
                assert_eq!((r1, c1, r2, c2), (2, 3, 2, 2));
            }
            _ => panic!("Expected a DimensionMismatch error"),
        }
    }
}