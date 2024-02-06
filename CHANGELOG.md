# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 (2024-02-06)

### ⚙️ CI/CD

- Update concurrency groups in workflows (#22)
- Run pytest on pull requests (#25)

### Other

- Increase maximum python version to 3.11 (#23)
- Remove numpy dependency and relax PyTorch version specification (#26)

## 0.4.0 (2023-10-08)

### ⚙️ CI/CD

- Add `.python-version` for consistency (#16)
- Update caching in workflows (#17)
- Update release workflow (#19)
- Update release workflow (#20)

### Other

- Update package versions (#18)

## 0.3.0 (2023-03-07)

### ❗️ Breaking Changes

- Remodel gathering of gradients (#13)

### 🚀 Features

- Add possibility to specify sup norm with `PNorm` (#8)

### 🐛 Fixes

- Remodel gathering of gradients (#13)

### 🧪 Tests

- Expand tests to cover checks for inputs on different devices (#12)

### ⚙️ CI/CD

- Update action workflows (#9)
- Change python version used in workflows (#10)
- Add tests tag to changelog config (#11)
- Change trigger for linting workflow (#14)

## 0.2.0 (2022-04-13)

### ❗️ BREAKING CHANGE

- move `target_layers` option from `GradientMetricCollector` to `GradientMetric`s

### 🚀 Features

- improve flexibility of defining `GradientMetric`s
- add `grad_transform` to `GradientMetric`s
