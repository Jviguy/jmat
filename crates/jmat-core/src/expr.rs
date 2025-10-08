use std::fmt::{Display, Formatter};
use num_traits::{One, Zero};
use crate::DMatrix;
use crate::scalar::{NumericScalar, Scalar};

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

impl Expr {
    pub fn make_num(n: f64) -> Self {
        if n.abs() < 1.0e-9 {
            Self::Zero
        } else if n == 1.0 {
            Self::One
        } else {
            Self::Num(n)
        }
    }
    
    pub fn matrix(name: char, m: usize, n: usize) -> DMatrix<Self> {
        let mut data: Vec<Self> = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                data.push(Self::Symbol(format!("{}_{},{}", name, i+1, j+1)));
            }
        }
        DMatrix::new(m, n, data)
    }
    
    pub fn simplify(&self) -> Self {
        match self {
            Expr::Num(n) => {
                Self::make_num(*n)
            }
            Expr::Add(a, b) => {
                let lhs = a.simplify();
                let rhs = b.simplify();
                // Addition we can only simplify addition by 0. and simple cases like 1 + 1 and nums.
                match (&lhs, &rhs) {
                    (Expr::Zero, _) => rhs,
                    (_, Expr::Zero) => lhs,
                    (Expr::One, Expr::One) => Self::make_num(2.0),
                    (Expr::Num(a), Expr::Num(b)) => Self::make_num(a + b),
                    _ => Self::Add(Box::new(lhs), Box::new(rhs))
                }
            }
            Expr::Mul(a, b) => {
                let lhs = a.simplify();
                let rhs = b.simplify();
                match (&lhs, &rhs) {
                    (Expr::Zero, _) => Self::Zero,
                    (_, Expr::Zero) => Self::Zero,
                    (Expr::One, Expr::One) => Self::One,
                    (Expr::One, _) => rhs,
                    (_, Expr::One) => lhs,
                    (Expr::Num(a), Expr::Num(b)) => Self::make_num(a * b),
                    _ => Self::Mul(Box::new(lhs), Box::new(rhs))
                }
            }
            Expr::Div(a, b) => {
                let lhs = a.simplify();
                let rhs = b.simplify();
                match (&lhs, &rhs) {
                    (Expr::Zero, _) => Self::Zero,
                    (_, Expr::Zero) => panic!("division by zero symbolically."),
                    (_, Expr::One) => lhs,
                    (Expr::Num(a), Expr::Num(b)) => Self::make_num(a / b),
                    _ => Self::Div(Box::new(lhs), Box::new(rhs))
                }
            }
            Expr::Neg(a) => {
                let new = a.simplify();
                match new {
                    
                    Expr::Num(n) => Self::make_num(-n),
                    _ => new.clone()
                }
            }
            _ => self.clone(),
        }
    }
    
    pub fn substitute(&mut self, name: &String, val: f64) {
        match self { 
            // always replace symbols if name is same.
            Expr::Symbol(s) => { 
                if (*s).eq(name) {
                    *self = Expr::make_num(val)
                }
            }
            Expr::Add(a, b) => {
                a.substitute(&name, val);
                b.substitute(&name, val);
            }
            Expr::Mul(a, b) => {
                a.substitute(&name, val);
                b.substitute(&name, val);
            }
            Expr::Div(a, b) => {
                a.substitute(&name, val);
                b.substitute(&name, val);
            }
            Expr::Neg(a) => a.substitute(&name, val),
            /* Things we don't care about skip only want to search recursive or replace symbols. */
            _ => (),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Symbol(s) => write!(f, "{}", s),
            Expr::Num(n) => write!(f, "{}", n),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Neg(a) => write!(f, "-{}", a),
            Expr::Zero => write!(f, "0"),
            Expr::One => write!(f, "1"),
        }
    }
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

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use super::*;

    #[test]
    fn test_make_num_factory() {
        // Test the zero case with a very small number
        assert_eq!(Expr::make_num(1e-10), Expr::Zero);
        assert_eq!(Expr::make_num(-1e-12), Expr::Zero);

        // Test the one case
        assert_eq!(Expr::make_num(1.0), Expr::One);

        // Test a standard number case
        assert_eq!(Expr::make_num(5.5), Expr::Num(5.5));
        assert_eq!(Expr::make_num(-3.14), Expr::Num(-3.14));
    }

    #[test]
    fn test_matrix_factory() {
        let matrix = Expr::matrix('A', 2, 3);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);

        // Check a few symbol names to ensure correct generation
        assert_eq!(matrix.get(0, 0), Some(&Expr::Symbol("A_1,1".to_string())));
        assert_eq!(matrix.get(1, 2), Some(&Expr::Symbol("A_2,3".to_string())));
    }

    #[test]
    fn test_display_formatting() {
        let x = Expr::Symbol("x".to_string());
        let two = Expr::Num(2.0);
        let five = Expr::Num(5.0);

        // Build a complex expression: ((x * 2) + 5)
        let expr = Expr::Add(
            Box::new(Expr::Mul(Box::new(x), Box::new(two))),
            Box::new(five)
        );

        let formatted_string = format!("{}", expr);
        assert_eq!(formatted_string, "((x * 2) + 5)");

        // Test negation
        let neg_expr = Expr::Neg(Box::new(expr));
        let formatted_neg_string = format!("{}", neg_expr);
        assert_eq!(formatted_neg_string, "-((x * 2) + 5)");
    }

    #[test]
    fn test_operator_macros() {
        let a = Expr::Symbol("a".to_string());
        let b = Expr::Symbol("b".to_string());

        // Test reference-based addition
        let add_expr = &a + &b;
        let expected_add = Expr::Add(Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(add_expr, expected_add);

        // Test owned multiplication
        let mul_expr = a.clone() * b.clone();
        let expected_mul = Expr::Mul(Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(mul_expr, expected_mul);

        // Test negation
        let neg_expr = -&a; 
        let expected_neg = Expr::Neg(Box::new(a.clone()));
        assert_eq!(neg_expr, expected_neg);

        assert_eq!(a, Expr::Symbol("a".to_string()));
    }

    #[test]
    fn test_zero_and_one_traits() {
        let zero = Expr::zero();
        let one = Expr::one();

        assert_eq!(zero, Expr::Zero);
        assert_eq!(one, Expr::One);

        assert!(zero.is_zero());
        assert!(!one.is_zero());

        let num_expr = Expr::Num(5.0);
        assert!(!num_expr.is_zero());
    }
}
