/*
 * Batched 2D keypoint Kalman kernel: the predict and measurement update of
 * `KeypointTracker2D._measurement_update` (commit 535ceef), for every keypoint
 * of a frame in one call.
 *
 * The arithmetic follows the frozen contract in kalman2d.h and matches the
 * NumPy reference `cpsa.vision.kalman_kernel.NumpyKalmanKernel`. Operation
 * order differs from NumPy's batched matmul, so results agree to about 1e-15
 * relative; never compile this file with -ffast-math, which would break that
 * agreement.
 *
 * Build (host tests, board deploy):
 *   cc -O2 -std=c99 -fPIC -shared -o libcpsa_kalman2d.so kalman2d.c -lm
 */
#include "kalman2d.h"

#include <math.h>
#include <stddef.h>

#define DIM CPSA_KALMAN2D_STATE_DIM

/* out = a * b, both (6, 6) row-major. */
static void mat6_mul(const double *a, const double *b, double *out) {
    int row, column, inner;
    for (row = 0; row < DIM; ++row) {
        for (column = 0; column < DIM; ++column) {
            double sum = 0.0;
            for (inner = 0; inner < DIM; ++inner) {
                sum += a[row * DIM + inner] * b[inner * DIM + column];
            }
            out[row * DIM + column] = sum;
        }
    }
}

/* out = a * b^T, used for F * P * F^T. */
static void mat6_mul_transposed(const double *a, const double *b, double *out) {
    int row, column, inner;
    for (row = 0; row < DIM; ++row) {
        for (column = 0; column < DIM; ++column) {
            double sum = 0.0;
            for (inner = 0; inner < DIM; ++inner) {
                sum += a[row * DIM + inner] * b[column * DIM + inner];
            }
            out[row * DIM + column] = sum;
        }
    }
}

/* x_pred = F x, P_pred = F P F^T + Q. */
static void predict_step(const double *F, const double *Q, const double *x, const double *P,
                         double *x_pred, double *p_pred) {
    double fp[DIM * DIM];
    double fpft[DIM * DIM];
    int row, inner;

    for (row = 0; row < DIM; ++row) {
        double sum = 0.0;
        for (inner = 0; inner < DIM; ++inner) {
            sum += F[row * DIM + inner] * x[inner];
        }
        x_pred[row] = sum;
    }
    mat6_mul(F, P, fp);
    mat6_mul_transposed(fp, F, fpft);
    for (row = 0; row < DIM * DIM; ++row) {
        p_pred[row] = fpft[row] + Q[row];
    }
}

/*
 * One position measurement update.
 *
 * `S = P[0:2, 0:2] + r I2` is inverted explicitly (the NumPy reference uses
 * `solve` and `pinv`, which coincide with this inverse for an invertible
 * symmetric 2x2); a non-finite or non-positive determinant, a non-finite
 * distance or a distance above the gate is a rejection. A rejection writes
 * nothing and returns 0.
 */
static int update_step(const double *x_pred, const double *p_pred, const double *z, double r,
                       double gate_chi2, double *x_filt, double *p_filt) {
    const double y0 = z[0] - x_pred[0];
    const double y1 = z[1] - x_pred[1];
    const double s00 = p_pred[0] + r;
    const double s01 = p_pred[1];
    const double s10 = p_pred[DIM];
    const double s11 = p_pred[DIM + 1] + r;
    const double det = s00 * s11 - s01 * s10;
    double i00, i01, i10, i11, distance;
    double gain[DIM * CPSA_KALMAN2D_MEASUREMENT_DIM];
    int row, column;

    if (!isfinite(det) || det <= 0.0) {
        return 0;
    }
    i00 = s11 / det;
    i01 = -s01 / det;
    i10 = -s10 / det;
    i11 = s00 / det;

    distance = y0 * (i00 * y0 + i01 * y1) + y1 * (i10 * y0 + i11 * y1);
    if (!isfinite(distance) || distance > gate_chi2) {
        return 0;
    }

    /* K = P_pred[:, 0:2] S^-1, (6, 2). */
    for (row = 0; row < DIM; ++row) {
        const double p0 = p_pred[row * DIM];
        const double p1 = p_pred[row * DIM + 1];
        gain[row * 2] = p0 * i00 + p1 * i10;
        gain[row * 2 + 1] = p0 * i01 + p1 * i11;
    }
    for (row = 0; row < DIM; ++row) {
        x_filt[row] = x_pred[row] + gain[row * 2] * y0 + gain[row * 2 + 1] * y1;
    }
    /* P_filt = P_pred - K P_pred[0:2, :] == (I - K H) P_pred, then symmetrized. */
    for (row = 0; row < DIM; ++row) {
        for (column = 0; column < DIM; ++column) {
            p_filt[row * DIM + column] =
                p_pred[row * DIM + column] -
                (gain[row * 2] * p_pred[column] + gain[row * 2 + 1] * p_pred[DIM + column]);
        }
    }
    for (row = 0; row < DIM; ++row) {
        for (column = row + 1; column < DIM; ++column) {
            const double mean = 0.5 * (p_filt[row * DIM + column] + p_filt[column * DIM + row]);
            p_filt[row * DIM + column] = mean;
            p_filt[column * DIM + row] = mean;
        }
    }
    return 1;
}

static void copy_state(const double *source, double *target) {
    int index;
    for (index = 0; index < DIM; ++index) {
        target[index] = source[index];
    }
}

static void copy_covariance(const double *source, double *target) {
    int index;
    for (index = 0; index < DIM * DIM; ++index) {
        target[index] = source[index];
    }
}

int32_t cpsa_kalman2d_abi_version(void) { return CPSA_KALMAN2D_ABI_VERSION; }

int32_t cpsa_kalman2d_step(int32_t k, const double *F, const double *Q, const double *x,
                           const double *P, const uint8_t *predict, const uint8_t *update,
                           const double *z, const double *r, double gate_chi2, double *x_pred,
                           double *p_pred, double *x_filt, double *p_filt, uint8_t *accepted) {
    int32_t index;

    if (k < 0) {
        return -1;
    }
    if (k == 0) {
        return 0;
    }
    if (F == NULL || Q == NULL || x == NULL || P == NULL || predict == NULL || update == NULL ||
        z == NULL || r == NULL || x_pred == NULL || p_pred == NULL || x_filt == NULL ||
        p_filt == NULL || accepted == NULL) {
        return -1;
    }

    for (index = 0; index < k; ++index) {
        const size_t offset = (size_t)index;
        const double *x_keypoint = x + offset * DIM;
        const double *p_keypoint = P + offset * DIM * DIM;
        const double *z_keypoint = z + offset * 2;
        double *x_pred_keypoint = x_pred + offset * DIM;
        double *p_pred_keypoint = p_pred + offset * DIM * DIM;
        double *x_filt_keypoint = x_filt + offset * DIM;
        double *p_filt_keypoint = p_filt + offset * DIM * DIM;

        if (predict[index]) {
            predict_step(F, Q, x_keypoint, p_keypoint, x_pred_keypoint, p_pred_keypoint);
        } else {
            copy_state(x_keypoint, x_pred_keypoint);
            copy_covariance(p_keypoint, p_pred_keypoint);
        }

        if (update[index] &&
            update_step(x_pred_keypoint, p_pred_keypoint, z_keypoint, r[index], gate_chi2,
                        x_filt_keypoint, p_filt_keypoint)) {
            accepted[index] = 1;
            continue;
        }
        copy_state(x_pred_keypoint, x_filt_keypoint);
        copy_covariance(p_pred_keypoint, p_filt_keypoint);
        accepted[index] = 0;
    }
    return 0;
}
