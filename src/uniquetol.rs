#[path = "isapprox.rs"] mod isapprox;
#[path = "test_arr.rs"] mod test_arr;

use crate::isapprox::{EqualNan, Tols, isapprox};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Occurrence {
    #[default]
    Lowest,
    Highest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniqueTolArray {
    pub arr_unique: Vec<f64>,
    pub indices_unique: Vec<usize>,
    pub inverse_unique: Vec<usize>,
    pub counts_unique: Vec<usize>,
}

impl UniqueTolArray {
    #[inline]
    pub fn remap_to_original(&self) -> Vec<f64> {
        self.inverse_unique.iter().map(|&idx| self.arr_unique[idx]).collect()
    }
    
    #[inline]
    pub fn get_len_unique(&self) -> usize {
        self.arr_unique.len()
    }
    
    #[inline]
    pub fn get_len_original(&self) -> usize {
        self.inverse_unique.len()
    }
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

pub fn uniquetol(arr: &[f64], tols: Tols, equal_nan: EqualNan, occurrence: Occurrence) -> UniqueTolArray {
    let n = arr.len();
    
    if n == 0 {
        return UniqueTolArray {
            arr_unique: Vec::new(),
            indices_unique: Vec::new(),
            inverse_unique: Vec::new(),
            counts_unique: Vec::new(),
        };
    }
    
    let perm_sorted = match occurrence {
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
        
        match isapprox(val_curr, val, tols, equal_nan) {
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
    
    indices_unique.push(perm_sorted[idx_curr]);
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
    
    fn test_uniquetol(occurrence: Occurrence) {
        let n = TEST_ARR.len();
        let k: usize = 179;
        
        let tols = Tols::default();
        let equal_nan = EqualNan::default();
        
        let uniquetol_arr = uniquetol(
            &TEST_ARR,
            tols,
            equal_nan,
            occurrence,
        );
        
        assert_eq!(uniquetol_arr.get_len_unique(), k);
        assert_eq!(uniquetol_arr.get_len_original(), n);
        assert_eq!(uniquetol_arr.counts_unique.iter().sum::<usize>(), n);
        
        let arr_unique = &uniquetol_arr.arr_unique;
        match occurrence {
            Occurrence::Lowest => assert!(arr_unique.is_sorted()),
            Occurrence::Highest => assert!(arr_unique.iter().rev().is_sorted()),
        }
        
        let is_unique = arr_unique
            .windows(2)
            .all(|w| !isapprox(w[0], w[1], tols, equal_nan));
        assert!(is_unique);
        
        let mapped_correctly = uniquetol_arr.indices_unique.iter().zip(arr_unique.iter())
            .all(|(&idx, &x)| isapprox(TEST_ARR[idx], x, tols, equal_nan));
        assert!(mapped_correctly);
        
        let arr_remapped = uniquetol_arr.remap_to_original();
        let remapped_correctly = arr_remapped.len() == n
            && TEST_ARR.iter().zip(arr_remapped.iter())
                .all(|(&x, &y)| isapprox(x, y, tols, equal_nan));
        assert!(remapped_correctly);
    }
    
    #[test]
    fn test_uniquetol_lowest() {
        test_uniquetol(Occurrence::Lowest);
    }
    
    #[test]
    fn test_uniquetol_highest() {
        test_uniquetol(Occurrence::Highest);
    }
}