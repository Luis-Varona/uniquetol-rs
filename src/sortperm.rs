pub fn sortperm(arr: &[f64]) -> Vec<usize> {
    let mut perm = (0..arr.len()).collect::<Vec<usize>>();
    // change unwrap to expect?
    perm.sort_by(|&i, &j| arr[i].partial_cmp(&arr[j]).unwrap());
    return perm;
}