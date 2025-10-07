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
        DMatrix { rows, cols, data }
    }
    
    pub fn empty(rows: usize, cols: usize) -> DMatrix<T> {
        let data = Vec::with_capacity(rows * cols);
        DMatrix { rows, cols, data }
    }
    
    pub fn zeros(rows: usize, cols: usize) -> DMatrix<T> where T: Zero + Clone {
        let data = vec![T::zero(); rows * cols];
        DMatrix { rows, cols, data }
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

impl<T> Add<Self> for DMatrix<T> where T: Add<T, Output = T> + Clone {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs).expect("DMatrix dimensions do not match for addition.")
    }
}

impl<T> CheckedAdd for DMatrix<T> where T: Add<Output= T> + Clone {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        if (self.rows != other.rows || self.cols != other.cols) {
            None
        } else {
            let data = self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a.clone() + b.clone())
                .collect();
            Some(DMatrix { rows: self.rows, cols: self.cols, data })
        }
    }
}

impl<T> Add<Self> for &DMatrix<T> where T: Add<T, Output = T> + Clone {
    type Output = DMatrix<T>;
    
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs).expect("DMatrix dimensions do not match for addition.")
    }
}

impl<T> DMatrix<T> {
    pub fn mul_scalar<K: Copy>(&self, k: K) -> DMatrix<T> where T: Mul<K, Output = T> + Clone {
        let data = self.data.iter()
            .map(|a| a.clone() * k)
            .collect();
        DMatrix { rows: self.rows, cols: self.cols, data }
    }
    
    pub fn mul<J, O>(&self, rhs: &DMatrix<J>) -> Result<DMatrix<O>, MatrixError> 
    where T: Mul<J, Output = O>, 
        J: Mul<J, Output = J> + Clone,
        O: Mul<J, Output = O> + Clone
    {
        
    }
}