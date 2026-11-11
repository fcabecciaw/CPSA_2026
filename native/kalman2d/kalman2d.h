/*
 * Frozen C ABI of the batched 2D keypoint Kalman kernel (plan F15).
 *
 * The Python side owns the masks, the re-initialization, the dropout counters
 * and the status; this library only runs the predict and measurement-update
 * arithmetic of `KeypointTracker2D` for every keypoint of one frame in one call.
 *
 * Every array is C-contiguous `float64` (flags `uint8`). The kernel keeps no
 * globals and allocates nothing, so it is thread-safe and the caller owns every
 * buffer. Inputs are never written.
 *
 * The header is frozen: the library reports `CPSA_KALMAN2D_ABI_VERSION` and the
 * Python adapter refuses a library that reports another version.
 */
#ifndef CPSA_KALMAN2D_H
#define CPSA_KALMAN2D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CPSA_KALMAN2D_ABI_VERSION 1
#define CPSA_KALMAN2D_STATE_DIM 6
#define CPSA_KALMAN2D_MEASUREMENT_DIM 2

/* Return the ABI version of the loaded library (currently 1). */
int32_t cpsa_kalman2d_abi_version(void);

/*
 * Run one batched predict/update step.
 *
 *   k          keypoints, >= 0
 *   F          (6, 6) transition matrix, row-major
 *   Q          (6, 6) process noise
 *   x          (k, 6) prior state
 *   P          (k, 6, 6) prior covariance
 *   predict    (k) flags: 1 = apply F and Q
 *   update     (k) flags: 1 = run the measurement update
 *   z          (k, 2) measured position in pixels
 *   r          (k) isotropic measurement variance, R_i = r_i * I2
 *   gate_chi2  two-degree-of-freedom chi-square gate
 *   x_pred     (k, 6) output, predicted state
 *   p_pred     (k, 6, 6) output, predicted covariance
 *   x_filt     (k, 6) output, filtered state
 *   p_filt     (k, 6, 6) output, filtered covariance
 *   accepted   (k) output, 1 where the measurement passed the gate
 *
 * Every output element is written for every keypoint. A keypoint with
 * `update[i]` whose measurement is rejected (singular or non-positive
 * innovation covariance, non-finite distance, or a distance above
 * `gate_chi2`) gets the prediction and `accepted[i] == 0`.
 *
 * Returns 0, or -1 when `k < 0` or a pointer is NULL (`k == 0` needs no arrays).
 */
int32_t cpsa_kalman2d_step(int32_t k, const double *F, const double *Q, const double *x,
                           const double *P, const uint8_t *predict, const uint8_t *update,
                           const double *z, const double *r, double gate_chi2, double *x_pred,
                           double *p_pred, double *x_filt, double *p_filt, uint8_t *accepted);

#ifdef __cplusplus
}
#endif

#endif /* CPSA_KALMAN2D_H */
