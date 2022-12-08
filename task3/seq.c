#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define U0(I, J, K) U0[(K)*N*N + (I)*N + (J)]
#define U1(I, J, K) U1[(K)*N*N + (I)*N + (J)]
#define U2(I, J, K) U2[(K)*N*N + (I)*N + (J)]

#define M_PI 3.14159265358979323846	

struct LValues {
    double x;
    double y;
    double z;
};

typedef struct LValues LValues;

struct HValues {
    double x;
    double y;
    double z;
};

double rtclock() {
    struct timeval Tp;
    int stat = gettimeofday (&Tp, NULL);
    if (stat != 0) 
        fprintf(stderr, "Error return from gettimeofday: %d\n", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

typedef struct HValues HValues;

double getAt(LValues L) {
    return M_PI * sqrt(1.0 / (L.x * L.x) +
                       1.0 / (L.y * L.y) +
                       4.0 / (L.z * L.z));
}

double u(double X, double Y, double Z, double T, LValues L) {
    return sin(M_PI * X / L.x) * sin(M_PI * Y / L.y)
            * sin(2 * M_PI * Z / L.z) * cos(getAt(L) * T + 2 * M_PI);
}

inline double phi(double X, double Y, double Z, LValues L) {
    return u(X, Y, Z, 0, L);
}

double fi(int I, int J, int K, HValues H, LValues L) {
    double Xi = I * H.x;
    double Yj = J * H.y;
    double Zk = K * H.z;
    return u(Xi, Yj, Zk, 0, L);
}

void initZeroLayer(double *U0, int N, HValues H, LValues L) {
    for (int I = 0; I <= N; I++) {
        double Xi = I * H.x;
        for (int J = 0; J <= N; J++) {
            double Yj = J * H.y;
            for (int K = 0; K <= N; K++) {
                double Zk = K * H.z;
                U0(I, J, K) = phi(Xi, Yj, Zk, L);
            }
        }
    }
}

void initFirstLayer(double *U1, double *U0, int N, double Tau, HValues H, LValues L) {
    // X border condition
    for (int J = 0; J <= N; J++) {
        for (int K = 0; K <= N; K++) {
            U1(0, J, K) = 0;
            U1(N, J, K) = 0;
        }
    }

    // Y border condition
    for (int I = 0; I <= N; I++) {
        for (int K = 0; K <= N; K++) {
            U1(I, 0, K) = 0;
            U1(I, N, K) = 0;
        }
    }

    // Z border condition
    for (int I = 0; I <= N; I++) {
        double Xi = I * H.x;
        for (int J = 0; J <= N; J++) {
            double Yj = J * H.y;
            U1(I, J, 0) = u(Xi, Yj, 0, Tau, L);
            U1(I, J, N) = u(Xi, Yj, N*H.z, Tau, L);
        }
    }

    // inner points
    for (int I = 1; I <= N-1; I++) {
        for (int J = 1; J <= N-1; J++) {
            for (int K = 1; K <= N-1; K++) {
                /*double Sx = (U0(I-1,J,K) - 2*U0(I,J,K) + U0(I+1, J, K)) / (H.x*H.x);
                double Sy = (U0(I,J-1,K) - 2*U0(I,J,K) + U0(I, J+1, K)) / (H.y*H.y);
                double Sz = (U0(I,J,K-1) - 2*U0(I,J,K) + U0(I, J, K+1)) / (H.z*H.z);*/
                double Sx = (fi(I-1,J,K,H,L) - 2*fi(I,J,K,H,L) + fi(I+1,J,K,H,L)) / (H.x*H.x);
                double Sy = (fi(I,J-1,K,H,L) - 2*fi(I,J,K,H,L) + fi(I,J+1,K,H,L)) / (H.y*H.y);
                double Sz = (fi(I,J,K-1,H,L) - 2*fi(I,J,K,H,L) + fi(I,J,K+1,H,L)) / (H.z*H.z);
                U1(I, J, K) = U0(I, J, K) + Tau*Tau*(Sx+Sy+Sz) / 2.0;
            }
        }
    }
}

void updateTop(double *U2, double *U1, double *U0, int N, double Tau, HValues H) {
    // X border condition
    for (int J = 0; J <= N; J++) {
        for (int K = 0; K <= N; K++) {
            U2(0, J, K) = 0;
            U2(N, J, K) = 0;
        }
    }

    // Y border condition
    for (int I = 0; I <= N; I++) {
        for (int K = 0; K <= N; K++) {
            U2(I, 0, K) = 0;
            U2(I, N, K) = 0;
        }
    }

    // inner points
    for (int I = 1; I <= N-1; I++) {
        for (int J = 1; J <= N-1; J++) {
            for (int K = 1; K <= N-1; K++) {
                double Sx = (U1(I-1,J,K) - 2*U1(I,J,K) + U1(I+1, J, K)) / (H.x*H.x);
                double Sy = (U1(I,J-1,K) - 2*U1(I,J,K) + U1(I, J+1, K)) / (H.y*H.y);
                double Sz = (U1(I,J,K-1) - 2*U1(I,J,K) + U1(I, J, K+1)) / (H.z*H.z);
                double Delta = Sx + Sy + Sz;
                U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
            }
        }
    }

    // Z border condition
    for (int I = 1; I <= N-1; I++) {
        for (int J = 1; J <= N-1; J++) {
            int K = N;
            double Sx = (U1(I-1,J,K) - 2*U1(I,J,K) + U1(I+1, J, K)) / (H.x*H.x);
            double Sy = (U1(I,J-1,K) - 2*U1(I,J,K) + U1(I, J+1, K)) / (H.y*H.y);
            double Sz = (U1(I,J,K-1) - 2*U1(I,J,K) + U1(I, J, 1)) / (H.z*H.z);
            double Delta = Sx + Sy + Sz;
            U2(I,J,N) = 2*U1(I,J,N) - U0(I,J,N) + Tau*Tau*Delta;
            U2(I,J,0) = U2(I,J,N);
        }
    }
}

double getDelta(double *U2, int N, double tn, LValues L, HValues H) {
    double MaxDelta = 0.0;
    for (int I = 0; I <= N; I++) {
        double Xi = I * H.x;
        for (int J = 0; J <= N; J++) {
            double Yj = J * H.y;
            for (int K = 0; K <= N; K++) {
                double Zk = K * H.z;
                double Delta = fabs(u(Xi,Yj,Zk,tn,L) - U2(I,J,K));
                if (Delta > MaxDelta)
                    MaxDelta = Delta;
            }
        }
    }
    return MaxDelta;
}

#ifdef DEBUG

void printMatrix(double *A, int N, int I) {\
    printf("[\n");
    for (int J = 0; J < N; J++) {
        printf("\t[ ");
        for (int K = 0; K < N; K++) {
            printf("%6.3lf ", A(I,J,K));
        }
        printf("]\n");
    }
    printf("]\n");
}

#endif

#ifdef PRINT
void generateDebugCube(double *U2, int N, LValues L, HValues H, double tn) {
    for (int I = 0; I <= N; I++) {
        double Xi = I * H.x;
        for (int J = 0; J <= N; J++) {
            double Yj = J * H.y;
            for (int K = 0; K <= N; K++) {
                double Zk = K * H.z;
                //fprintf(stderr, "%.3lf ", fabs(u(Xi,Yj,Zk,tn,L) - U2(I,J,K)));
                //fprintf(stderr, "%.3lf ", U2(I,J,K));
                fprintf(stderr, "%.3lf ", u(Xi,Yj,Zk,tn,L));
            }
        }
    }
    fprintf(stderr, "\n");
}
#endif

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Not enough arguments. Usage: ./task3 <Lx> <Ly> <Lz> <N> <T>");
        exit(1);
    }
    double Lx = atof(argv[1]);
    double Ly = atof(argv[2]);
    double Lz = atof(argv[3]);
    int N = atoi(argv[4]);
    double T = atof(argv[5]);
    int N1_3 = (N+1)*(N+1)*(N+1);
    double *ZeroLayer = malloc(N1_3 * sizeof(double));
    double *FirstLayer = malloc(N1_3 * sizeof(double));
    double *TopLayer = malloc(N1_3 * sizeof(double));

    double Hx = Lx / (double) N;
    double Hy = Ly / (double) N;
    double Hz = Lz / (double) N;
    HValues H = {Hx, Hy, Hz};
    LValues L = {Lx, Ly, Lz};

    double stabCond = 1 / (getAt(L) * sqrt(1/(Hx*Hx) + 1/(Hy*Hy) + 1/(Hz*Hz)));
    int K = T / stabCond;
    double Tau = T / (double) K;
    #ifdef DEBUG
    printf("Tau_opt <= %.5lf\n", stabCond);
    printf("%.5lf <= K_opt\n", T / stabCond);
    #endif

    double Time1 = rtclock();
    initZeroLayer(ZeroLayer, N, H, L);
    initFirstLayer(FirstLayer, ZeroLayer, N, Tau, H, L);
    double Delta = getDelta(ZeroLayer, N, Tau*0, L, H);
    printf("Max delta [%d]: %.15lf\n", 0, Delta);
    Delta = getDelta(FirstLayer, N, Tau*1, L, H);
    printf("Max delta [%d]: %.15lf\n", 1, Delta);
    double *Top = NULL;
    for (int n = 2; n <= 20; n++) {
        updateTop(TopLayer, FirstLayer, ZeroLayer, N, Tau, H);
        Delta = getDelta(TopLayer, N, Tau*n, L, H);
        printf("Max delta [%d]: %.15lf\n", n, Delta);

        Top = TopLayer;
        TopLayer = ZeroLayer; // will be rewritten on the next step
        ZeroLayer = FirstLayer;
        FirstLayer = Top;
    }
    #ifdef PRINT
    generateDebugCube(Top, N, L, H, Tau*20);
    #endif
    double Time2 = rtclock();
    printf("Total time: %0.6lf\n", Time2 - Time1);
    free(TopLayer);
    free(FirstLayer);
    free(ZeroLayer);
    return 0;
}