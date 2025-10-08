use crate::DMatrix;
use crate::expr::Expr;

#[derive(Debug, Clone)]
pub enum JmatValue {
    Concrete(DMatrix<f64>),
    Symbolic(DMatrix<Expr>),
}