# Native classifier libraries — provenance

These two shared libraries are vendored, opaque build artifacts. They cannot be
rebuilt from this repository: the MATLAB project and the generated C sources that
produced them are not part of it (the generated sources were removed from the
tree; they remain in git history at commit `fcf3ec9`).

| Library | sha256 |
|---|---|
| `libProcessDataWristsQuat.so` | `2f8202d37d52b1909e0db6ae1df5acddb56a1aa2373f2257a5ddfbcdddd1e6f4` |
| `libPredictPericolosaWristsQuat.so` | `c257edf5ce2791ef0a79cd697c3200207f8ff19228b9ca0d94e9338394fa8030` |

Hashes were recorded with `sha256sum` before the files were moved from
`IMU_pipeline/data_stream/` and `IMU_pipeline/classifiers/stereotipy_classifier/`
(baseline `fcf3ec9`). A byte-identical copy of `libProcessDataWristsQuat.so`
(`IMU_pipeline/data_stream/ProcessDataWristsQuat/`) was dropped.
`tests/contract/test_provenance.py` checks that the files match this table.

## Common facts

- **Tool:** MATLAB Coder 25.1, C code generation.
- **Target architecture:** ELF 64-bit LSB shared objects, ARM aarch64 (Kria KV260,
  Ubuntu 22.04). They do not load on x86.
- **Licence:** the generated sources carry the MATLAB *Academic License* header:
  "for use in teaching, academic research, and meeting course requirements at
  degree granting institutions only. Not for government, commercial, or other
  organizational use."

## `libProcessDataWristsQuat.so` — feature extraction

- **Generated:** 03-Sep-2025 17:31:54.
- **Signature:**

  ```c
  void ProcessDataWristsQuat(
      float accX_R[150], float accY_R[150], float accZ_R[150],
      float gyrX_R[150], float gyrY_R[150], float gyrZ_R[150],
      float accX_L[150], float accY_L[150], float accZ_L[150],
      float gyrX_L[150], float gyrY_L[150], float gyrZ_L[150],
      const float quatRW_x[150], const float quatRW_y[150],
      const float quatRW_z[150], const float quatRW_w[150],
      const float quatLW_x[150], const float quatLW_y[150],
      const float quatLW_z[150], const float quatLW_w[150],
      float imuFeatures[18]);
  void ProcessDataWristsQuat_initialize(void);  /* also exported: ProcessDataWristsQuat_init */
  ```

- **Inputs:** acceleration in m/s², angular rate in dps, quaternions frame-aligned
  and scaled ×10000 (see `cpsa.imu.window.preprocess`). The acc/gyr arrays are
  non-const and may be modified by the library; the adapter passes private copies.
- **Window:** 150 samples, fixed by the ABI; the classifier was trained on a 75-sample hop.
- **State:** the function keeps internal state between calls. The runtime discards
  the first window before any call and discards the features of the first call.

## `libPredictPericolosaWristsQuat.so` — stereotypy classifier

- **Generated:** 13-Aug-2025 14:37:24.
- **Model:** MATLAB `CompactClassificationTree`, 185 nodes, 3 classes.
- **Signature:**

  ```c
  unsigned char Predict_Pericolosa_Wrists_Quat(const float x[18]);
  void c_Predict_Pericolosa_Wrists_Qua(void);   /* initializer called by the adapter */
  ```

- **Output:** 1 `NON_DANGEROUS`, 2 `DANGEROUS`, 3 `NON_STEREOTIPY`.
