#[path = "isapprox.rs"] mod isapprox;
#[path = "test_arr.rs"] mod test_arr;

use isapprox::{Tols, isapprox};

#[derive(Default, PartialEq)]
pub enum Occurrence {
    #[default]
    Lowest,
    Highest,
}

pub struct UniqueTolArray {
    pub arr_unique: Vec<f64>,
    pub indices_unique: Vec<usize>,
    pub inverse_unique: Vec<usize>,
    pub counts_unique: Vec<usize>,
}

fn sortperm(arr: &[f64], reverse: bool) -> Vec<usize> {
    let n = arr.len();
    let mut perm_sorted: Vec<usize> = (0..n).collect();
    
    match reverse {
        false => perm_sorted.sort_by(|&i, &j| arr[i].total_cmp(&arr[j])),
        true => perm_sorted.sort_by(|&i, &j| arr[j].total_cmp(&arr[i])),
    }
    
    perm_sorted
}

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
    
    let perm_sorted = match *occurrence {
        Occurrence::Lowest => sortperm(arr, false),
        Occurrence::Highest => sortperm(arr, true),
    };
    
    let mut indices_unique = Vec::with_capacity(n);
    let mut inverse_unique = vec![0; n];
    let mut counts_unique = Vec::with_capacity(n);
    
    let mut idx_curr = 0;
    let mut cnt_curr: usize = 1;
    let mut val_curr = arr[perm_sorted[0]];
    
    for (i, &idx) in perm_sorted.iter().enumerate().skip(1) {
        let val = arr[idx];
        
        match isapprox(val_curr, val, tols) {
            true => cnt_curr += 1,
            false => {
                indices_unique.push(perm_sorted[idx_curr]);
                counts_unique.push(cnt_curr);
                
                for j in 0..cnt_curr {
                    inverse_unique[perm_sorted[idx_curr + j]] = indices_unique.len() - 1;
                }
                
                idx_curr = i;
                cnt_curr = 1;
                val_curr = val;
            }
        }
    }
    
    indices_unique.push(idx_curr);
    counts_unique.push(cnt_curr);
    indices_unique.shrink_to_fit();
    counts_unique.shrink_to_fit();
    let num_unique = indices_unique.len();
    
    for j in 0..cnt_curr {
        inverse_unique[perm_sorted[idx_curr + j]] = num_unique - 1;
    }
    
    let mut arr_unique: Vec<f64> = Vec::with_capacity(num_unique);
    
    for &idx in indices_unique.iter() {
        arr_unique.push(arr[idx]);
    }
    
    UniqueTolArray {
        arr_unique,
        indices_unique,
        inverse_unique,
        counts_unique,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_arr::TEST_ARR;
    
    #[test]
    fn test_uniquetol_lowest() {
        let uniquetol_arr = uniquetol(
            &TEST_ARR,
            &Tols::default(),
            &Occurrence::default(),
        );
        
        let n = TEST_ARR.len();
        let k = uniquetol_arr.arr_unique.len();
        assert_eq!(n, 729);
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
    
    #[test]
    fn test_uniquetol_highest() {
        let uniquetol_arr = uniquetol(
            &TEST_ARR,
            &Tols::default(),
            &Occurrence::Highest,
        );
        
        let n = TEST_ARR.len();
        let k = uniquetol_arr.arr_unique.len();
        assert_eq!(n, 729);
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