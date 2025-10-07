
// A compile time static matrix on the stack.
pub struct SMatrix<T, const M: usize, const N: usize> {
    data: [[T; N]; M],
}