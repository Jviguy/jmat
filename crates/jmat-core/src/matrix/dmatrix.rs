use std::cmp::{max, min};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Div, Mul, SubAssign};
use num_traits::{CheckedAdd, Float, Zero};
use crate::expr::Expr;
use crate::matrix::{Matrix, MatrixError};
use crate::scalar::{NumericScalar, Scalar};

// Dynamic runtime matrix.
// TODO: Check if Eq and Partial
#[derive(Debug, Clone, Eq, PartialEq)]
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
    
    pub fn zeros(rows: usize, cols: usize) -> DMatrix<T> where T: Scalar {
        Self::from_val(rows, cols, T::zero())
    }
    
    pub fn identity(n: usize) -> DMatrix<T> where T: Scalar {
        // make an n x n all zero matrix.
        let mut res = Self::zeros(n, n);
        for i in 0..n {
            res.data[i * n + i] = T::one();
        };
        res
    }


    // Utils No matter what T is.
    fn row_swap(&mut self, r1: usize, r2: usize) {
        if r1 == r2 || r1 >= self.rows || r2 >= self.rows {
            return;
        }

        let (rs, rl) = (min(r1, r2), max(r1, r2));
        let (first, second) = self.data.split_at_mut(rl * self.cols);
        let row1 = &mut first[(rs * self.cols)..(rs + 1) * self.cols];
        let row2 = &mut second[0..self.cols];
        row1.swap_with_slice(row2)
    }
}

impl DMatrix<Expr> {
    pub fn symbolic(name: char, m: usize, n: usize) -> DMatrix<Expr> {
        let mut data = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                data.push(Expr::Symbol(format!("{}_{},{}", name, i+1, j+1)));
            }
        }
        DMatrix { rows: m, cols: n, data }
    }

    pub fn symbolic_identity(n: usize) -> DMatrix<Expr> {
        let mut data = vec![Expr::Zero; n*n];
        for i in 0..n {
            data[i*n + i] = Expr::One;
        }

        DMatrix{ rows: n, cols:n, data }
    }
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

impl<T: Scalar> Add<Self> for &DMatrix<T> where for <'a> &'a T: Add<&'a T, Output = T> {
    type Output = DMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("DMatrix dimensions do not match for addition.")
    }
}

impl<T: Scalar> Add<DMatrix<T>> for DMatrix<T>
where
        for<'a> &'a T: Add<&'a T, Output = T>,
{
    type Output = DMatrix<T>;
    fn add(self, rhs: DMatrix<T>) -> Self::Output {
        (&self).checked_add(&rhs).expect("DMatrix dimensions do not match for addition.")
    }
}

impl<T: Scalar> CheckedAdd for DMatrix<T>
    where for <'a> &'a T: Add<&'a T, Output = T>{
    fn checked_add(&self, other: &Self) -> Option<DMatrix<T>> {
        if self.rows != other.rows || self.cols != other.cols {
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

impl<T: Scalar> DMatrix<T> {
    pub fn mul_scalar<K: Scalar>(&self, k: K) -> DMatrix<T> where T: Mul<K, Output = T> + Clone {
        let data = self.data.iter()
            .map(|a| a.clone() * k.clone())
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
                    let a = self.get(i, k).unwrap();
                    let b = rhs.get(k, j).unwrap();
                    sum = sum + (a * b)
                }
                vec.push(sum);
            }
        }
        Ok(DMatrix {rows: self.rows, cols: rhs.cols(), data: vec})
    }

    /// Ensures no row swapping for ease of div, no scaling either.
    /// This is used for symbolic ref. As we can't do any ordering.
    /// This is useful for users by finding determinant using the property
    /// det(A) = product of diagonals of u where u is ref of A.
    pub fn naive_row_echelon_form(&mut self)
    where
            for<'a> &'a T: Div<&'a T, Output = T>,
            for<'a> &'a T: Mul<&'a T, Output = T>,
            T: SubAssign<T>,
    {
        let (rows, cols) = (self.rows, self.cols);
        let mut pivot_row = 0;
        for pivot_col in 0..cols {
            if pivot_row >= rows { break; }

            // NOTE: The pivoting search and swap is removed.
            // We just need to find the first non-zero row to be the pivot.
            if self.get(pivot_row, pivot_col).unwrap().is_zero() {
                let first_non_zero = ((pivot_row+1)..rows).find(|r| !self.get(*r, pivot_col).unwrap().is_zero());
                if let Some(r) = first_non_zero {
                    self.row_swap(pivot_row, r);
                } else {
                    continue; // Column is all zeros, move to next
                }
            }

            let pivot_val = self.get(pivot_row, pivot_col).unwrap().clone();
            if pivot_val.is_zero() {
                continue;
            }

            for i in (pivot_row + 1)..rows {
                let current_val_ref = self.get(i, pivot_col).unwrap();
                let factor = current_val_ref / &pivot_val;

                for j in pivot_col..cols {
                    let val_from_pivot_row = self.get(pivot_row, j).unwrap().clone();
                    let product = &factor * &val_from_pivot_row;
                    self.data[i * cols + j].sub_assign(product);
                }
            }
            pivot_row += 1;
        }
    }
}

impl<T: NumericScalar> DMatrix<T> {
    pub fn row_echelon_form(&mut self)
    where
        for<'a> &'a T: Div<&'a T, Output = T>,
        for<'a> &'a T: Mul<T, Output = T>,
        T: SubAssign<T>,
    {
        let (rows, cols) = (self.rows, self.cols);
        let mut pivot_row = 0;
        for pivot_col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            let mut best_pivot_row = pivot_row;
            for i in (pivot_row + 1)..rows {
                // basically whatever has the greatest position number lol.
                // in terms of highest number at the greatest
                if self.get(i, pivot_col).unwrap() > self.get(best_pivot_row, pivot_col).unwrap() {
                    best_pivot_row = i;
                }
            }
            self.row_swap(pivot_row, best_pivot_row);
            let pivot_val = self.get(pivot_row, pivot_col).unwrap().clone();
            if pivot_val.is_zero() {
                continue;
            }

            for i in (pivot_row + 1)..rows {
                let current = self.get(i, pivot_col).unwrap();

                let factor = current / &pivot_val;

                for j in pivot_col..cols {
                    let piv_value = self.get(pivot_row, j).unwrap().clone();
                    let product = &factor * piv_value;
                    self.data[i * cols + j].sub_assign(product)
                }
            }
            pivot_row += 1;
        }
    }

    pub fn to_ref(&self) -> DMatrix<T>
    where
        for<'a> &'a T: Div<&'a T, Output = T>,
        for<'a> &'a T: Mul<T, Output = T>,
        T: SubAssign<T>,
    {
        let mut new = self.clone();
        new.row_echelon_form();
        new
    }
}

impl<T> Display for DMatrix<T>
where
// We constrain this to `Float` types because handling near-zero values
// and formatting decimals is a concept specific to floating-point numbers.
// `Display` is needed to format the numbers themselves.
    T: Float + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut max_width = 0;
        let mut formatted_nums = Vec::with_capacity(self.data.len());

        for val in &self.data {
            let s = if val.abs() <= T::epsilon() * T::from(100.0).unwrap() {
                "0.0".to_string()
            } else {
                // Format other numbers with a reasonable precision.
                format!("{:.4}", val)
            };

            if s.len() > max_width {
                max_width = s.len();
            }
            formatted_nums.push(s);
        }

        // Building the final string.
        writeln!(f)?; // Start with a newline for better spacing
        for i in 0..self.rows {
            write!(f, "[ ")?;
            for j in 0..self.cols {
                let s = &formatted_nums[i * self.cols + j];
                // Pad each number with spaces to align the columns.
                write!(f, "{: >width$}", s, width = max_width)?;
                if j < self.cols - 1 {
                    write!(f, "  ")?;
                }
            }
            writeln!(f, " ]")?;
        }

        Ok(())
    }
}


// Macros
#[macro_export]
macro_rules! dmatrix {
    () => {
        $crate::DMatrix::new(0, 0, vec![]);
    };
   ( $( $( $x:expr ),* );* ) => {
        {
            let data = vec![ $( $( $x ),* ),* ];

            let rows_arr = [ $( [ $( $x ),* ] ),* ];

            let row_count = rows_arr.len();
            let mut col_count = 0;

            if row_count > 0 {
                col_count = rows_arr[0].len();
            }
            for r in 1..row_count {
                assert_eq!(rows_arr[r].len(), col_count, "All rows in a dmatrix! \
                macro must have the same number of columns.");
            }
            $crate::DMatrix::new(row_count, col_count, data)
        }
    };
    ( $( $( $x:expr ),* );* ; ) => {
        dmatrix![ $( $( $x ),* );* ]
    };
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
    /// A helper function to compare floating-point matrices for approximate equality.
    /// Direct comparison with `==` can fail due to tiny floating-point inaccuracies.
    fn assert_matrices_approx_equal(a: &DMatrix<f64>, b: &DMatrix<f64>, epsilon: f64) {
        assert_eq!(a.rows(), b.rows(), "Matrix row counts do not match");
        assert_eq!(a.cols(), b.cols(), "Matrix column counts do not match");

        for i in 0..a.rows() {
            for j in 0..a.cols() {
                let val_a = a.get(i, j).unwrap();
                let val_b = b.get(i, j).unwrap();
                assert!(
                    (val_a - val_b).abs() < epsilon,
                    "Matrix values at ({}, {}) are not approximately equal: {} vs {}",
                    i, j, val_a, val_b
                );
            }
        }
    }

    #[test]
    fn test_row_echelon_form_standard() {
        // A standard 3x3 matrix that requires elimination.
        let mut matrix = DMatrix::new(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);

        // CORRECTED: This is the actual result of the implemented algorithm
        // after pivoting on the initial '7.0' and subsequent eliminations.
        let expected = DMatrix::new(3, 3, vec![
            7.0, 8.0, 9.0,
            0.0, 0.85714, 1.71428,
            0.0, 0.0, 0.0,
        ]);

        matrix.row_echelon_form();

        assert_matrices_approx_equal(&matrix, &expected, 1e-5);
    }

    #[test]
    fn test_row_echelon_form_with_pivoting() {
        // This matrix has a zero in the initial pivot position,
        // forcing the algorithm to swap rows.
        let mut matrix = DMatrix::new(3, 3, vec![
            0.0, 2.0, 1.0,
            1.0, -2.0, -3.0,
            -1.0, 1.0, 2.0,
        ]);

        // CORRECTED: This is the actual result. The last element is -0.5, not 0.5.
        let expected = DMatrix::new(3, 3, vec![
            1.0, -2.0, -3.0,
            0.0, 2.0,  1.0,
            0.0, 0.0,  -0.5,
        ]);

        matrix.row_echelon_form();

        assert_matrices_approx_equal(&matrix, &expected, 1e-5);
    }

    #[test]
    fn test_to_ref_non_mutating() {
        // Verifies that the out-of-place method `to_ref` works
        // and does not modify the original matrix.
        let original_matrix = DMatrix::new(2, 3, vec![
            1.0, 2.0, -1.0,
            0.0, 1.0, 1.0,
        ]);

        // Create a clone to verify the original is untouched.
        let original_clone = original_matrix.clone();

        let expected_ref = DMatrix::new(2, 3, vec![
            1.0, 2.0, -1.0,
            0.0, 1.0, 1.0,
        ]);

        // The matrix is already in REF, so the result should be the same.
        let ref_matrix = original_matrix.to_ref();

        // Check that the result is correct.
        assert_matrices_approx_equal(&ref_matrix, &expected_ref, 1e-5);
        // CRITICAL: Check that the original matrix was not changed.
        assert_eq!(&original_matrix, &original_clone, "The original matrix was modified!");
    }

    #[test]
    fn test_row_echelon_form_wide_matrix() {
        // A "wide" matrix (more columns than rows)
        let mut matrix = DMatrix::new(2, 4, vec![
            2.0, 1.0, -1.0, 8.0,
            -3.0, -1.0, 2.0, -11.0,
        ]);

        let expected = DMatrix::new(2, 4, vec![
            2.0, 1.0, -1.0, 8.0,
            0.0, 0.5, 0.5, 1.0,
        ]);

        matrix.row_echelon_form();

        assert_matrices_approx_equal(&matrix, &expected, 1e-5);
    }
    
    #[test]
    fn test_matrix_factory() {
        let matrix = DMatrix::symbolic('A', 2, 3);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);

        // Check a few symbol names to ensure correct generation
        assert_eq!(matrix.get(0, 0), Some(&Expr::Symbol("A_1,1".to_string())));
        assert_eq!(matrix.get(1, 2), Some(&Expr::Symbol("A_2,3".to_string())));
    }
}