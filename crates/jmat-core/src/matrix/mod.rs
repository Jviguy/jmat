use thiserror::Error;
mod smatrix;
mod dmatrix;

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("Cannot make a {0}x{1} matrix.")]
    InvalidDimensions(usize, usize),
    #[error("Matrix dimensions do not match, A: {0}x{1}, B: {2}x{3}")]
    DimensionMismatch(usize, usize, usize, usize),
    #[error("Out of bounds access: ({0}, {1}) not in {2}x{3} mat")]
    OutOfBounds(usize, usize, usize, usize),
}

pub trait Matrix {
    type Item;
 
    fn rows(&self) -> usize;
    
    fn cols(&self) -> usize;
    
    fn get(&self, i: usize, j: usize) -> Option<&Self::Item>;
    
    fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut Self::Item>;
    
    fn set(&mut self, i: usize, j: usize, x: &Self::Item) -> Result<(), MatrixError> where Self::Item: Clone;
}