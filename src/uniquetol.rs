#[path = "isapprox.rs"] mod isapprox;
#[path = "test_arr.rs"] mod test_arr;

use isapprox::{Tols, isapprox};

pub fn uniquetol(arr: &[f64], tols: &Tols, occurrence: &Occurrence) -> UniqueTolArray {
    let n = arr.len();
    
    if n == 0 {
        return UniqueTolArray {
            arr_unique: Vec::new(),
            indices_unique: Vec::new(),
            inverse_unique: Vec::new(),
            counts_unique: Vec::new(),
        };
    }
    
    let mut perm_sorted: Vec<usize> = (0..n).collect();
    perm_sorted.sort_by(|&i, &j| arr[i].total_cmp(&arr[j]));
    
    let mut arr_sorted: Vec<f64> = Vec::with_capacity(n);
    let mut c = arr[perm_sorted[0]];
    arr_sorted.push(c);
    
    let mut indices_unique: Vec<usize> = Vec::with_capacity(n);
    indices_unique.push(0);
    let mut num_unique = 1;
    
    for i in 1..n {
        let next = arr[perm_sorted[i]];
        arr_sorted.push(next);
        
        if !isapprox(c, next, &tols) {
            c = next;
            indices_unique.push(i);
            num_unique += 1;
        }
    }
    
    let mut arr_unique: Vec<f64> = Vec::with_capacity(num_unique);
    let mut inverse_unique: Vec<usize> = vec![0; n]; // (1)
    // let mut inverse_unique: Vec<usize> = Vec::with_capacity(n); (1)
    let mut counts_unique: Vec<usize> = Vec::with_capacity(num_unique);
    
    indices_unique = indices_unique.into_boxed_slice().into_vec();
    let index_last = indices_unique[num_unique - 1];
    let count_last = n - index_last;
    
    for j in 0..count_last {
        inverse_unique[perm_sorted[index_last + j]] = num_unique - 1;
    }
    
    for i in 0..(num_unique - 1) {
        let index = indices_unique[i];
        let count = indices_unique[i + 1] - index;
        counts_unique.push(count);
        
        for j in 0..count {
            inverse_unique[perm_sorted[index + j]] = i;
        }
    }
    
    counts_unique.push(count_last);
    
    if *occurrence == Occurrence::Highest {
        for i in 0..(num_unique) {
            indices_unique[i] = perm_sorted[indices_unique[i]];
        }
        
        indices_unique[num_unique - 1] = n - 1;
    }
    
    for i in 0..num_unique {
        let index = perm_sorted[indices_unique[i]];
        arr_unique.push(arr[index]);
        indices_unique[i] = index;
    }
    
    return UniqueTolArray {
        arr_unique,
        indices_unique,
        inverse_unique,
        counts_unique,
    };
}

pub struct UniqueTolArray {
    pub arr_unique: Vec<f64>,
    pub indices_unique: Vec<usize>,
    pub inverse_unique: Vec<usize>,
    pub counts_unique: Vec<usize>,
}

#[derive(PartialEq)]
pub enum Occurrence {
    Lowest,
    Highest,
}

impl Default for Occurrence {
    fn default() -> Occurrence {
        Occurrence::Lowest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_arr::TEST_ARR;
    
    #[test]
    fn test_uniquetol() {
        let uniquetol_arr = uniquetol(
            &TEST_ARR,
            &Tols::default(),
            &Occurrence::default());
        
        let n = TEST_ARR.len();
        let k = uniquetol_arr.arr_unique.len();
        assert_eq!(k, 179);
        
        println!("Length of the original input array: {}\n", n);
        println!("Number of unique elements within tolerance: {}\n", k);
        println!("Unique elements: {:?}\n", uniquetol_arr.arr_unique);
        println!(
            "Indices of the unique elements in the original array: {:?}\n",
            uniquetol_arr.indices_unique
        );
        println!(
            "Indices of the original elements in the unique array: {:?}\n",
            uniquetol_arr.inverse_unique
        );
        println!("Counts of unique elements: {:?}\n", uniquetol_arr.counts_unique);
    }
}