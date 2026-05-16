#ifndef KALMAN_H
#define KALMAN_H

#include <math.h>

// A minimal namespace for matrix operations required by the Kalman filter.
namespace SimpleMatrix {
    void multiply(const float* a, const float* b, int r1, int c1, int c2, float* out) {
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                out[i * c2 + j] = 0;
                for (int k = 0; k < c1; k++) {
                    out[i * c2 + j] += a[i * c1 + k] * b[k * c2 + j];
                }
            }
        }
    }

    void add(const float* a, const float* b, int r, int c, float* out) {
        for (int i = 0; i < r * c; i++) out[i] = a[i] + b[i];
    }

    void subtract(const float* a, const float* b, int r, int c, float* out) {
        for (int i = 0; i < r * c; i++) out[i] = a[i] - b[i];
    }

    void transpose(const float* a, int r, int c, float* out) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out[j * r + i] = a[i * c + j];
            }
        }
    }

    // Inverts a 3x3 matrix. Returns false if the matrix is singular.
    bool inverse3x3(const float* a, float* out) {
        float det = a[0]*(a[4]*a[8] - a[5]*a[7]) - a[1]*(a[3]*a[8] - a[5]*a[6]) + a[2]*(a[3]*a[7] - a[4]*a[6]);
        if (fabs(det) < 1e-9) return false;
        float inv_det = 1.0f / det;
        out[0] = (a[4]*a[8] - a[5]*a[7]) * inv_det;
        out[1] = (a[2]*a[7] - a[1]*a[8]) * inv_det;
        out[2] = (a[1]*a[5] - a[2]*a[4]) * inv_det;
        out[3] = (a[5]*a[6] - a[3]*a[8]) * inv_det;
        out[4] = (a[0]*a[8] - a[2]*a[6]) * inv_det;
        out[5] = (a[2]*a[3] - a[0]*a[5]) * inv_det;
        out[6] = (a[3]*a[7] - a[4]*a[6]) * inv_det;
        out[7] = (a[1]*a[6] - a[0]*a[7]) * inv_det;
        out[8] = (a[0]*a[4] - a[1]*a[3]) * inv_det;
        return true;
    }
}

class KalmanFilter {
public:
    // State vector: [px, py, pz, vx, vy, vz]
    float x[6] = {0, 0, 0, 0, 0, 0};
    // Covariance matrix P
    float P[36] = {1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1};

    // --- Tuning Parameters ---
    // Process noise: how much we trust our physics model (lower is more trust)
    float Q_accel_noise = 0.8f;
    // Measurement noise: how much we trust the UWB reading (lower is more trust)
    float R_uwb_noise = 0.1f;

    void predict(float ax, float ay, float az, float dt) {
        // State transition matrix A
        float A[36] = {
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
        };

        // Predict state: x_k = A * x_{k-1} + B * u_k
        x[0] += x[3] * dt + 0.5f * ax * dt * dt;
        x[1] += x[4] * dt + 0.5f * ay * dt * dt;
        x[2] += x[5] * dt + 0.5f * az * dt * dt;
        x[3] += ax * dt;
        x[4] += ay * dt;
        x[5] += az * dt;

        // Process noise covariance matrix Q
        float dt2 = dt*dt;
        float dt3 = dt2*dt;
        float dt4 = dt2*dt2;
        float Q[36] = {
            0.25f*dt4, 0, 0, 0.5f*dt3, 0, 0,
            0, 0.25f*dt4, 0, 0, 0.5f*dt3, 0,
            0, 0, 0.25f*dt4, 0, 0, 0.5f*dt3,
            0.5f*dt3, 0, 0, dt2, 0, 0,
            0, 0.5f*dt3, 0, 0, dt2, 0,
            0, 0, 0.5f*dt3, 0, 0, dt2
        };
        for(int i=0; i<36; ++i) Q[i] *= Q_accel_noise;

        // Predict covariance: P_k = A * P_{k-1} * A^T + Q
        float AP[36], AT[36];
        SimpleMatrix::transpose(A, 6, 6, AT);
        SimpleMatrix::multiply(A, P, 6, 6, 6, AP);
        SimpleMatrix::multiply(AP, AT, 6, 6, 6, P);
        SimpleMatrix::add(P, Q, 6, 6, P);
    }

    // This is not used in the dead-reckoning model, but is here for completeness.
    // The PC will calculate the correction and send it back.
    void update(const float* z) {
        // Measurement matrix H (we measure position x, y, z)
        float H[18] = {1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0};
        float HT[18];
        SimpleMatrix::transpose(H, 3, 6, HT);

        // Measurement noise R
        float R[9] = {R_uwb_noise,0,0, 0,R_uwb_noise,0, 0,0,R_uwb_noise};

        // Innovation covariance: S = H*P*H' + R
        float PHT[18], S[9];
        SimpleMatrix::multiply(P, HT, 6, 6, 3, PHT);
        SimpleMatrix::multiply(H, PHT, 3, 6, 3, S);
        SimpleMatrix::add(S, R, 3, 3, S);

        // Kalman gain: K = P*H'*inv(S)
        float S_inv[9];
        if (!SimpleMatrix::inverse3x3(S, S_inv)) return;
        float K[18];
        SimpleMatrix::multiply(PHT, S_inv, 6, 3, 3, K);

        // Update state: x = x + K*(z - H*x)
        float Hx[3] = {x[0], x[1], x[2]};
        float y[3];
        SimpleMatrix::subtract(z, Hx, 3, 1, y);
        float Ky[6];
        SimpleMatrix::multiply(K, y, 6, 3, 1, Ky);
        SimpleMatrix::add(x, Ky, 6, 1, x);

        // Update covariance: P = (I - K*H)*P
        float KH[36], I_KH[36];
        SimpleMatrix::multiply(K, H, 6, 3, 6, KH);
        for(int i=0; i<36; ++i) I_KH[i] = -KH[i];
        for(int i=0; i<6; ++i) I_KH[i*6+i] += 1.0f;

        float P_new[36];
        SimpleMatrix::multiply(I_KH, P, 6, 6, 6, P_new);
        for(int i=0; i<36; ++i) P[i] = P_new[i];
    }
};

#endif
