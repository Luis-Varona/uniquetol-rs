# uniquetol-rs

[![Build Status](https://github.com/Luis-Varona/uniquetol-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/Luis-Varona/uniquetol-rs/actions/workflows/rust.yml?query=branch%3Amain)

uniquetol is a Rust toolbox for isolating unique values in n-dimensional arrays
of imprecise floating-point data within a given tolerance. In the
one-dimensional case, it returns the largest subset in which no pairs of
elements are approximately equal. Here two numbers `x` and `y` are said to be
"approximately equal" within an absolute tolerance `atol` or a relative
tolerance `rtol` if and only if `|x - y| ≤ max(atol, rtol∙max(|x|, |y|))`.

This project was inspired by the `uniquetol` function in MATLAB and NumPy's
`unique` function in Python.

**(CURRENTLY UNDER DEVELOPMENT)**
