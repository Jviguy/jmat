use std::ops::{Add, Div, Mul, Neg};
use num_traits::{One, Signed, Zero};

pub trait Scalar:
Clone
+ Zero
+ One
+ Add<Output = Self>
+ Mul<Output = Self>
+ Div<Output = Self>
+ Neg<Output = Self> {}

pub trait NumericScalar: Scalar + PartialOrd + Signed {}

macro_rules! impl_scalar_for {
    ($($t:ty),*) => {
        $(
            impl Scalar for $t {}

            impl NumericScalar for $t {}
        )*
    };
}


impl_scalar_for!(f32, f64, i8, i16, i32, i64, isize);