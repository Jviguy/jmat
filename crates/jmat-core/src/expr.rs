use num_traits::{One, Signed, Zero};
use crate::scalar::Scalar;

#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    Symbol(String),
    Num(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Zero,
    One
}

impl Zero for Expr {
    fn zero() -> Self { Expr::Zero }
    fn is_zero(&self) -> bool { matches!(self, Expr::Zero) }
}

impl One for Expr {
    fn one() -> Self { Expr::One }
}

macro_rules! impl_op_binary {
    ($t:ty, $op:ident, $method:ident, $variant:ident) => {
        impl<'a, 'b> std::ops::$op<&'b $t> for &'a $t {
            type Output = $t;

            fn $method(self, rhs: &'b $t) -> $t {
                <$t>::$variant(Box::new(self.clone()), Box::new(rhs.clone()))
            }
        }

        impl std::ops::$op for $t {
            type Output = $t;

            fn $method(self, rhs: $t) -> $t {
                <$t>::$variant(Box::new(self), Box::new(rhs))
            }
        }
    };
}

impl_op_binary!(Expr, Add, add, Add);
impl_op_binary!(Expr, Mul, mul, Mul);
impl_op_binary!(Expr, Div, div, Div);

macro_rules! impl_op_unary {
    ($t:ty, $op:ident, $method:ident, $variant:ident) => {
        impl<'a, 'b> std::ops::$op for &'a $t {
            type Output = $t;

            fn $method(self) -> $t {
                <$t>::$variant(Box::new(self.clone()))
            }
        }

        impl std::ops::$op for $t {
            type Output = $t;

            fn $method(self) -> $t {
                <$t>::$variant(Box::new(self.clone()))
            }
        }
    };
}

impl_op_unary!(Expr, Neg, neg, Neg);

impl PartialOrd for Expr {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        None
    }
}

impl Eq for Expr {}

impl Scalar for Expr {}