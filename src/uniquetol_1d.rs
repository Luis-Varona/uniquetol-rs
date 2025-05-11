// Copyright 2025 Luis M. B. Varona
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[path = "test_arr.rs"]
mod test_arr;

use num_traits::Float;
use std::fmt::{Debug, Display};

use crate::isapprox::{NanComparison, Tols, isapprox};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Occurrence {
    #[default]
    Lowest,
    Highest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniqueTolResult<F>
where
    F: Float + Display + Debug,
{
    pub arr_unique: Vec<F>,
    pub indices_unique: Vec<usize>,
    pub inverse_unique: Vec<usize>,
    pub counts_unique: Vec<usize>,
}

impl<F> UniqueTolResult<F>
where
    F: Float + Display + Debug,
{
    #[inline]
    pub fn remap_to_original(&self) -> Vec<F> {
        self.inverse_unique
            .iter()
            .map(|&idx| self.arr_unique[idx])
            .collect()
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

pub fn sortperm<F>(arr: &[F], reverse: bool) -> Vec<usize>
where
    F: Float + Display + Debug,
{
    let mut perm: Vec<usize> = (0..arr.len()).collect();

    match reverse {
        false => perm.sort_by(|&i, &j| {
            arr[i]
                .partial_cmp(&arr[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        true => perm.sort_by(|&i, &j| {
            arr[j]
                .partial_cmp(&arr[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
    }

    perm
}

pub fn uniquetol_1d<A, F>(
    arr: A,
    tols: Tols<F>,
    nan_cmp: NanComparison,
    occurrence: Occurrence,
) -> UniqueTolResult<F>
where
    A: AsRef<[F]>,
    F: Float + Display + Debug,
{
    let arr = arr.as_ref();
    let n = arr.len();

    if n == 0 {
        return UniqueTolResult {
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

        if isapprox(val_curr, val, tols, nan_cmp) {
            cnt_curr += 1;
        } else {
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

    indices_unique.push(perm_sorted[idx_curr]);
    counts_unique.push(cnt_curr);

    for j in 0..cnt_curr {
        inverse_unique[perm_sorted[idx_curr + j]] = indices_unique.len() - 1;
    }

    let arr_unique = indices_unique.iter().map(|&i| arr[i]).collect();

    UniqueTolResult {
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

    fn test_uniquetol_1d(occurrence: Occurrence) {
        let n = TEST_ARR.len();
        let k: usize = 179;

        let tols = Tols::default();
        let nan_cmp = NanComparison::default();

        let uniquetol_arr = uniquetol_1d(&TEST_ARR, tols, nan_cmp, occurrence);

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
            .all(|w| !isapprox(w[0], w[1], tols, nan_cmp));
        assert!(is_unique);

        let mapped_correctly = uniquetol_arr
            .indices_unique
            .iter()
            .zip(arr_unique.iter())
            .all(|(&idx, &x)| isapprox(TEST_ARR[idx], x, tols, nan_cmp));
        assert!(mapped_correctly);

        let arr_remapped = uniquetol_arr.remap_to_original();
        let remapped_correctly = arr_remapped.len() == n
            && TEST_ARR
                .iter()
                .zip(arr_remapped.iter())
                .all(|(&x, &y)| isapprox(x, y, tols, nan_cmp));
        assert!(remapped_correctly);
    }

    #[test]
    fn test_uniquetol_1d_lowest() {
        test_uniquetol_1d(Occurrence::Lowest);
    }

    #[test]
    fn test_uniquetol_1d_highest() {
        test_uniquetol_1d(Occurrence::Highest);
    }
}
