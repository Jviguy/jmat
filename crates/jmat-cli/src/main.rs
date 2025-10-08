use jmat_core::{dmatrix};

fn main() {
    let mut matrix = dmatrix![25.0, -3.0, 3.0; 5.0, 9.0, 6.0; 0.0, -15.0, -9.0];
    matrix.naive_row_echelon_form();
    println!("Ref: {}", matrix);
}
