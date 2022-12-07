#include <mpich/mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include "common.h"

#define A(I, J, K) A[(K)*X*Y + (J)*X + (I)]
#define U0(I, J, K) U0[(K)*X*Y + (J)*X + (I)]
#define U1(I, J, K) U1[(K)*X*Y + (J)*X + (I)]
#define U2(I, J, K) U2[(K)*X*Y + (J)*X + (I)]
#define OwnYZ(J, K) OwnYZ[(K)*Y + (J)]
#define OtherYZ(J, K) OtherYZ[(K)*Y + (J)]
#define OwnXY(I, J) OwnXY[(J)*X + (I)]
#define OtherXY(I, J) OtherXY[(J)*X + (I)]
#define OwnXZ(I, K) OwnXZ[(K)*X + (I)]
#define OtherXZ(I, K) OtherXZ[(K)*X + (I)]

#define XI(I) ((I) + DI.Volume.RangeX.Min) * H.x;
#define YJ(J) ((J) + DI.Volume.RangeY.Min) * H.y;
#define ZK(K) ((K) + DI.Volume.RangeZ.Min) * H.z;

void printMatrix(double *A, DivisionInfo &DI, double tn, LValues &L, Values &Vals);

double inline getAt(LValues &L) {
    return M_PI * sqrt(1.0 / (L.x * L.x) +
                       1.0 / (L.y * L.y) +
                       4.0 / (L.z * L.z));
}

double u(double X, double Y, double Z, double T, LValues L) {
    double at = M_PI * std::sqrt(1.0 / (L.x * L.x) +
                                 1.0 / (L.y * L.y) +
                                 4.0 / (L.z * L.z));
    return std::sin(M_PI * X / L.x) * std::sin(M_PI * Y / L.y)
            * std::sin(2 * M_PI * Z / L.z) * std::cos(at * T + 2 * M_PI);
}

double inline phi(double X, double Y, double Z, LValues L) {
    return u(X, Y, Z, 0, L);
}

double *initZeroLayer(const DivisionInfo &DI, const LValues &L, const HValues &H,
                      const Values &Vals) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;

    double *A = new double[X * Y * Z];

    #pragma omp parallel for
    for (int I = 0; I < X; I++) {
        double Xi = XI(I);
        for (int J = 0; J < Y; J++) {
            double Yi = YJ(J);
            for (int K = 0; K < Z; K++) {
                double Zi = ZK(K);
                A(I, J, K) = phi(Xi, Yi, Zi, L);
            }
        }
    }  
    return A;
}

double *initFirstLayer(double *U0, const DivisionInfo &DI, const LValues &L,
                       const HValues &H, const Values &Vals, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;
    
    double *U1 = new double[X * Y * Z];

    // Border conditions for X
    double Xi0 = XI(0);
    double Xi1 = XI(X-1);
    if (DI.Coord.X == 0) {
        #pragma omp parallel for
        for (int J = 0; J < Y; J++) {
            double Yj = YJ(J);
            for (int K = 0; K < Z; K++) {
                U1(0, J, K) = 0;
                double Zk = ZK(K);
                U1(X-1, J, K) = u(Xi1,Yj,Zk,Tau,L);
            }
        }
    } else if (DI.Coord.X == DI.CubeNum.X-1) {
        #pragma omp parallel for
        for (int J = 0; J < Y; J++) {
            double Yj = YJ(J);
            for (int K = 0; K < Z; K++) {
                double Zk = ZK(K);
                U1(0, J, K) = u(Xi0,Yj,Zk,Tau,L);
                U1(X-1, J, K) = 0;
            }
        }
    } else {
        #pragma omp parallel for
        for (int J = 0; J < Y; J++) {
            double Yj = YJ(J);
            for (int K = 0; K < Z; K++) {
                double Zk = ZK(K);
                U1(0, J, K) = u(Xi0,Yj,Zk,Tau,L);
                U1(X-1, J, K) = u(Xi1,Yj,Zk,Tau,L);
            }
        }
    }

    // Border conditions for Y
    double Yj0 = YJ(0);
    double Yj1 = YJ(Y-1);
    if (DI.Coord.Y == 0) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++) {
            for (int K = 0; K < Z; K++) {
                double Xi = XI(I);
                double Zk = ZK(K);
                U1(I, 0, K) = 0;
                U1(I, Y-1, K) = u(Xi, Yj1, Zk, Tau, L);
            }
        }
    } else if (DI.Coord.Y == DI.CubeNum.Y-1) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++) {
            for (int K = 0; K < Z; K++) {
                double Xi = XI(I);
                double Zk = ZK(K);
                U1(I, 0, K) = u(Xi, Yj0, Zk, Tau, L);
                U1(I, Y-1, K) = 0;  
            }
        }
    } else {
        #pragma omp parallel for
        for (int I = 0; I < X; I++) {
            for (int K = 0; K < Z; K++) {
                double Xi = XI(I);
                double Zk = ZK(K);
                U1(I, 0, K) = u(Xi, Yj0, Zk, Tau, L);
                U1(I, Y-1, K) = u(Xi, Yj1, Zk, Tau, L);
            }
        }
    }

    // Border conditions for Z
    #pragma omp parallel
    {
        double Zk0 = ZK(0);
        double Zk1 = ZK(Z-1);
        #pragma omp for nowait
        for (int I = 0; I < X; I++) {
            double Xi = XI(I);
            for (int J = 0; J < Y; J++) {
                double Yj = YJ(J);
                U1(I, J, 0) = u(Xi, Yj, Zk0, Tau, L);
                U1(I, J, Z-1) = u(Xi, Yj, Zk1, Tau, L);
            }
        }

        #pragma omp for
        for (int I = 1; I < X-1; I++) {
            for (int J = 1; J < Y-1; J++) {
                for (int K = 1; K < Z-1; K++) {
                    double U0IJK = U0(I,J,K);
                    double Sx = (U0(I-1,J,K) - 2*U0IJK + U0(I+1, J, K)) / (H.x*H.x);
                    double Sy = (U0(I,J-1,K) - 2*U0IJK + U0(I, J+1, K)) / (H.y*H.y);
                    double Sz = (U0(I,J,K-1) - 2*U0IJK + U0(I, J, K+1)) / (H.z*H.z);
                    U1(I, J, K) = U0IJK + Tau*Tau*(Sx+Sy+Sz) / 2.0;
                }
            }
        }
    }
    return U1;
}

void calculateInnerPoints(double *U2, double *U1, double *U0,
                          const DivisionInfo &DI, const HValues &H, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;

    #pragma omp parallel for firstprivate(X,Y,Z)
    for (int I = 1; I < X-1; I++) {
        for (int J = 1; J < Y-1; J++) {
            for (int K = 1; K < Z-1; K++) {
                double U1IJK = U1(I,J,K);
                double S1 = (U1(I-1,J,K)-2*U1IJK+U1(I+1,J,K)) / (H.x*H.x);
                double S2 = (U1(I,J-1,K)-2*U1IJK+U1(I,J+1,K)) / (H.y*H.y);
                double S3 = (U1(I,J,K-1)-2*U1IJK+U1(I,J,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,K) = 2*U1IJK - U0(I,J,K) + Tau*Tau*Delta;
            }
        }
    }
}

enum Neighbour {
    LEFT = 0,
    RIGHT
};

enum Dimension {
    DIM_X = 0,
    DIM_Y,
    DIM_Z
};

int getNeighbourRank(const DivisionInfo &DI, Dimension Dim, Neighbour NB) {
    int NX = -1, NY = -1, NZ = -1;
    switch (Dim)
    {
    case DIM_X:
        if (NB == LEFT) {
            NX = (DI.Coord.X == 0) ? DI.CubeNum.X-1 : DI.Coord.X-1;
        } else {
            NX = (DI.Coord.X == DI.CubeNum.X-1) ? 0 : DI.Coord.X+1;
        }
        NY = DI.Coord.Y;
        NZ = DI.Coord.Z;
        break;
    case DIM_Y:
        NX = DI.Coord.X;
        if (NB == LEFT) {
            NY = (DI.Coord.Y == 0) ? DI.CubeNum.Y-1 : DI.Coord.Y-1;
        } else {
            NY = (DI.Coord.Y == DI.CubeNum.Y-1) ? 0 : DI.Coord.Y+1;
        }
        NZ = DI.Coord.Z;
        break;
    case DIM_Z:
        NX = DI.Coord.X;
        NY = DI.Coord.Y;
        if (NB == LEFT) {
            NZ = (DI.Coord.Z == 0) ? DI.CubeNum.Z-1 : DI.Coord.Z-1;
        } else {
            NZ = (DI.Coord.Z == DI.CubeNum.Z-1) ? 0 : DI.Coord.Z+1;
        }
        break;
    default:
        assert("unreachable");
    }
    int Square = DI.CubeNum.X * DI.CubeNum.Y;
    int Width = DI.CubeNum.X;
    return Square * NZ + Width * NY + NX;
}

void updateYZRight(double *U2, double *U1, double *U0,
                   const DivisionInfo &DI, const HValues &H,
                   const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;
    

    // Exchange borders for X
    int LeftRankX = getNeighbourRank(DI, DIM_X, LEFT);
    int RightRankX = getNeighbourRank(DI, DIM_X, RIGHT);
    double *OwnYZ = Wall.Own;
    double *OtherYZ = Wall.Other;

    // Fill left YZ dimension
    #pragma omp parallel for
    for (int J = 0; J < Y; J++)
        for (int K = 0; K < Z; K++)
            OwnYZ(J, K) = U1(0, J, K);
    // Send left YZ to the left neighbour. Receive left YZ from the right neighbour.
    // Recalculate right X border.
    int Count = DI.Volume.Y * DI.Volume.Z;
    MPI_Status Status;
    MPI_Sendrecv(OwnYZ, Count, MPI_DOUBLE, LeftRankX, 0,
                 OtherYZ, Count, MPI_DOUBLE, RightRankX, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate right border.
    if (DI.Coord.X == DI.CubeNum.X - 1) {
        #pragma omp parallel for
        for (int J = 0; J < Y; J++)
            for (int K = 0; K < Z; K++)
                U2(X-1, J, K) = 0;
    } else {
        /*--------------------Fill inner points of YZ-------------------------*/
        int I = X-1;
        #pragma omp parallel firstprivate(I)
        {
        for (int J = 1; J < Y-1; J++) {
            for (int K = 1; K < Z-1; K++) {
                double S1 = (U1(I-1,J,K)-2*U1(I,J,K)+OtherYZ(J, K)) / (H.x*H.x);
                double S2 = (U1(I,J-1,K)-2*U1(I,J,K)+U1(I,J+1,K)) / (H.y*H.y);
                double S3 = (U1(I,J,K-1)-2*U1(I,J,K)+U1(I,J,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
            }
        }
        /*----------Fill border points of YZ, exclude corner points-----------*/
        // Top line

            #pragma omp for nowait
            for (int J = 1; J < Y-1; J++) {
                double U1IJ0 = U1(I,J,0);
                double S1 = (U1(I-1,J,0)-2*U1IJ0+OtherYZ(J, 0)) / (H.x*H.x);
                double S2 = (U1(I,J-1,0)-2*U1IJ0+U1(I,J+1,0)) / (H.y*H.y);
                double S3 = (           -2*U1IJ0+U1(I,J,1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,0) = 2*U1IJ0 - U0(I,J,0) + Tau*Tau*Delta;
            }
            // Bottom line
            #pragma omp for
            for (int J = 1; J < Y-1; J++) {
                double U1IJZ1 = U1(I,J,Z-1);
                double S1 = (U1(I-1,J,Z-1)-2*U1IJZ1+OtherYZ(J, Z-1)) / (H.x*H.x);
                double S2 = (U1(I,J-1,Z-1)-2*U1IJZ1+U1(I,J+1,Z-1)) / (H.y*H.y);
                double S3 = (U1(I,J,Z-2)  -2*U1IJZ1              ) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,Z-1) = 2*U1IJZ1 - U0(I,J,Z-1) + Tau*Tau*Delta;
            }
        }
        // Left line
        #pragma omp parallel firstprivate(I)
        {
            #pragma omp for
            for (int K = 1; K < Z-1; K++) {
                double S1 = (U1(I-1,0,K)-2*U1(I,0,K)+OtherYZ(0, K)) / (H.x*H.x);
                double S2 = (           -2*U1(I,0,K)+U1(I,1,K)) / (H.y*H.y);
                double S3 = (U1(I,0,K-1)-2*U1(I,0,K)+U1(I,0,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,0,K) = 2*U1(I,0,K) - U0(I,0,K) + Tau*Tau*Delta;
            }
            // Right line
            #pragma omp for
            for (int K = 1; K < Z-1; K++) {
                double S1 = (U1(I-1,Y-1,K)-2*U1(I,Y-1,K)+OtherYZ(Y-1, K)) / (H.x*H.x);
                double S2 = (U1(I,Y-2,K)  -2*U1(I,Y-1,K)          ) / (H.y*H.y);
                double S3 = (U1(I,Y-1,K-1)-2*U1(I,Y-1,K)+U1(I,Y-1,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,Y-1,K) = 2*U1(I,Y-1,K) - U0(I,Y-1,K) + Tau*Tau*Delta;
            }
        }
        double S1, S2, S3, Delta;
        /*---------------------Fill corner points of YZ-----------------------*/
        // Top left
        S1 = (U1(I-1,0,0)-2*U1(I,0,0)+OtherYZ(0,0)) / (H.x*H.x);
        S2 = (U1(I,1,0)  -2*U1(I,0,0)) / (H.y*H.y);
        S3 = (U1(I,0,1)  -2*U1(I,0,0)) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,0,0) = 2*U1(I,0,0) - U0(I,0,0) + Tau*Tau*Delta;
        // Top right
        S1 = (OtherYZ(Y-1,0)-2*U1(I,Y-1,0)+U1(I-1,Y-1,0)) / (H.x*H.x);
        S2 = (U1(I,Y-2,0)   -2*U1(I,Y-1,0)       ) / (H.y*H.y);
        S3 = (              -2*U1(I,Y-1,0)+U1(I,Y-1,1)) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,Y-1,0) = 2*U1(I,Y-1,0) - U0(I,Y-1,0) + Tau*Tau*Delta;
        // Bottom left
        S1 = (OtherYZ(0,Z-1)-2*U1(I,0,Z-1)+U1(I-1,0,Z-1)) / (H.x*H.x);
        S2 = (             -2*U1(I,0,Z-1)+U1(I,1,Z-1)) / (H.y*H.y);
        S3 = (U1(I,0,Z-2)  -2*U1(I,0,Z-1)        ) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,0,Z-1) = 2*U1(I,0,Z-1) - U0(I,0,Z-1) + Tau*Tau*Delta;
        // Bottom right
        S1 = (OtherYZ(Y-1,Z-1)-2*U1(I,Y-1,Z-1)+U1(I-1,Y-1,Z-1)) / (H.x*H.x);
        S2 = (U1(I,Y-2,Z-1)  -2*U1(I,Y-1,Z-1)                  ) / (H.y*H.y);
        S3 = (U1(I,Y-1,Z-2)  -2*U1(I,Y-1,Z-1)        ) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,Y-1,Z-1) = 2*U1(I,Y-1,Z-1) - U0(I,Y-1,Z-1) + Tau*Tau*Delta;
    }
}

void updateYZLeft(double *U2, double *U1, double *U0,
                  const DivisionInfo &DI, const HValues &H,
                  const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;

    // Exchange borders for X
    int LeftRankX = getNeighbourRank(DI, DIM_X, LEFT);
    int RightRankX = getNeighbourRank(DI, DIM_X, RIGHT);
    double *OwnYZ = Wall.Own;
    double *OtherYZ = Wall.Other;

    // Fill right YZ dimension
    #pragma omp parallel for
    for (int J = 0; J < Y; J++)
        for (int K = 0; K < Z; K++)
            OwnYZ(J, K) = U1(X-1, J, K);
    // Send right YZ to the right neighbour. Receive right YZ from the left neighbour.
    // Recalculate left X border.
    int Count = DI.Volume.Y * DI.Volume.Z;
    MPI_Status Status;
    MPI_Sendrecv(OwnYZ, Count, MPI_DOUBLE, RightRankX, 0,
                 OtherYZ, Count, MPI_DOUBLE, LeftRankX, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate left border.
    if (DI.Coord.X == 0) {
        #pragma omp parallel for
        for (int J = 0; J < Y; J++)
            for (int K = 0; K < Z; K++)
                U2(0, J, K) = 0;
    } else {
        /*--------------------Fill inner points of YZ-------------------------*/
        int I = 0;
        #pragma omp parallel firstprivate(I)
        {
            #pragma omp for nowait
            for (int J = 1; J < Y-1; J++) {
                for (int K = 1; K < Z-1; K++) {
                    double U1IJK = U1(I,J,K);
                    double S1 = (OtherYZ(J,K)-2*U1IJK+U1(I+1,J,K)) / (H.x*H.x);
                    double S2 = (U1(I,J-1,K)-2*U1IJK+U1(I,J+1,K)) / (H.y*H.y);
                    double S3 = (U1(I,J,K-1)-2*U1IJK+U1(I,J,K+1)) / (H.z*H.z);
                    double Delta = S1 + S2 + S3;
                    U2(I,J,K) = 2*U1IJK - U0(I,J,K) + Tau*Tau*Delta;
                }
            }

            /*----------Fill border points of YZ, exclude corner points-----------*/
            // Top line
            #pragma omp for nowait
            for (int J = 1; J < Y-1; J++) {
                double U1IJ0 = U1(I,J,0);
                double S1 = (OtherYZ(J,0)-2*U1IJ0+U1(I+1,J,0)) / (H.x*H.x);
                double S2 = (U1(I,J-1,0)-2*U1IJ0+U1(I,J+1,0)) / (H.y*H.y);
                double S3 = (           -2*U1IJ0+U1(I,J,1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,0) = 2*U1IJ0 - U0(I,J,0) + Tau*Tau*Delta;
            }
            // Bottom line
            #pragma omp parallel
            for (int J = 1; J < Y-1; J++) {
                double U1IJZ1 = U1(I,J,Z-1);
                double S1 = (OtherYZ(J,Z-1)-2*U1IJZ1+U1(I+1,J,Z-1)) / (H.x*H.x);
                double S2 = (U1(I,J-1,Z-1)-2*U1IJZ1+U1(I,J+1,Z-1)) / (H.y*H.y);
                double S3 = (U1(I,J,Z-2)  -2*U1IJZ1              ) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,Z-1) = 2*U1IJZ1 - U0(I,J,Z-1) + Tau*Tau*Delta;
            }
        }
        #pragma omp parallel firstprivate(I)
        {
            // Left line
            #pragma omp for nowait
            for (int K = 1; K < Z-1; K++) {
                double U1I0K = U1(I,0,K);
                double S1 = (OtherYZ(0,K)  -2*U1I0K+U1(I+1,0,K)) / (H.x*H.x);
                double S2 = (           -2*U1I0K+U1(I,1,K)) / (H.y*H.y);
                double S3 = (U1(I,0,K-1)-2*U1I0K+U1(I,0,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,0,K) = 2*U1I0K - U0(I,0,K) + Tau*Tau*Delta;
            }
            // Right line
            #pragma omp for
            for (int K = 1; K < Z-1; K++) {
                double U1Y1K = U1(I,Y-1,K);
                double S1 = (OtherYZ(Y-1,K)   -2*U1Y1K+U1(I+1,Y-1,K)) / (H.x*H.x);
                double S2 = (U1(I,Y-2,K)  -2*U1Y1K          ) / (H.y*H.y);
                double S3 = (U1(I,Y-1,K-1)  -2*U1Y1K+U1(I,Y-1,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,Y-1,K) = 2*U1Y1K - U0(I,Y-1,K) + Tau*Tau*Delta;
            }
        }
        double S1, S2, S3, Delta;
        /*---------------------Fill corner points of YZ-----------------------*/
        // Top left
        double U1I00 = U1(I,0,0);
        S1 = (OtherYZ(0,0)-2*U1I00+U1(I+1,0,0)) / (H.x*H.x);
        S2 = (            -2*U1I00+U1(I,1,0)) / (H.y*H.y);
        S3 = (            -2*U1I00+U1(I,0,1)) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,0,0) = 2*U1I00 - U0(I,0,0) + Tau*Tau*Delta;
        // Top right
        double U1IY10 = U1(I,Y-1,0);
        S1 = (OtherYZ(Y-1,0)-2*U1IY10+U1(I+1,Y-1,0)) / (H.x*H.x);
        S2 = (U1(I,Y-2,0)   -2*U1IY10       ) / (H.y*H.y);
        S3 = (              -2*U1IY10+U1(I,Y-1,1)) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,Y-1,0) = 2*U1IY10 - U0(I,Y-1,0) + Tau*Tau*Delta;
        // Bottom left
        double U10Z1 = U1(I,0,Z-1);
        S1 = (OtherYZ(0,Z-1)-2*U10Z1+U1(I+1,0,Z-1)) / (H.x*H.x);
        S2 = (             -2*U10Z1+U1(I,1,Z-1)) / (H.y*H.y);
        S3 = (U1(I,0,Z-2)  -2*U10Z1        ) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,0,Z-1) = 2*U10Z1 - U0(I,0,Z-1) + Tau*Tau*Delta;
        // Bottom right
        double T1IY1Z1 = U1(I,Y-1,Z-1);
        S1 = (OtherYZ(Y-1,Z-1)-2*T1IY1Z1+U1(I+1,Y-1,Z-1)) / (H.x*H.x);
        S2 = (U1(I,Y-2,Z-1)  -2*T1IY1Z1                  ) / (H.y*H.y);
        S3 = (U1(I,Y-1,Z-2)  -2*T1IY1Z1        ) / (H.z*H.z);
        Delta = S1 + S2 + S3;
        U2(I,Y-1,Z-1) = 2*T1IY1Z1 - U0(I,Y-1,Z-1) + Tau*Tau*Delta;
    }
}

void updateXZRight(double *U2, double *U1, double *U0,
                   const DivisionInfo &DI, const HValues &H,
                   const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;
    

    // Exchange borders for Y
    int LeftRankY = getNeighbourRank(DI, DIM_Y, LEFT);
    int RightRankY = getNeighbourRank(DI, DIM_Y, RIGHT);
    double *OwnXZ = Wall.Own;
    double *OtherXZ = Wall.Other;

    // Fill left XZ dimension
    #pragma omp parallel for
    for (int I = 0; I < X; I++)
        for (int K = 0; K < Z; K++)
            OwnXZ(I, K) = U1(I, 0, K);
    // Send left XZ to the left neighbour. Receive left XZ from the right neighbour.
    // Recalculate right Y border.
    int Count = DI.Volume.X * DI.Volume.Z;
    MPI_Status Status;
    MPI_Sendrecv(OwnXZ, Count, MPI_DOUBLE, LeftRankY, 0,
                 OtherXZ, Count, MPI_DOUBLE, RightRankY, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate right border.
    if (DI.Coord.Y == DI.CubeNum.Y - 1) {
        for (int I = 0; I < X; I++)
            for (int K = 0; K < Z; K++)
                U2(I, Y-1, K) = 0;
    } else {
        /*--------------------Fill inner points of XZ-------------------------*/
        int J = Y-1;
        #pragma omp parallel firstprivate(J)
        {
            #pragma omp for nowait
            for (int I = 1; I < X-1; I++) {
                for (int K = 1; K < Z-1; K++) {
                    double S1 = (U1(I-1,J,K)-2*U1(I,J,K)+U1(I+1,J,K)) / (H.x*H.x);
                    double S2 = (U1(I,J-1,K)-2*U1(I,J,K)+OtherXZ(I,K)) / (H.y*H.y);
                    double S3 = (U1(I,J,K-1)-2*U1(I,J,K)+U1(I,J,K+1)) / (H.z*H.z);
                    double Delta = S1 + S2 + S3;
                    U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
                }
            }

            /*----------Fill border points of XZ, exclude corner points-----------*/
            // Top line
            #pragma omp for nowait
            for (int I = 1; I < X-1; I++) {
                double S1 = (U1(I-1,J,0)-2*U1(I,J,0)+U1(I+1,J,0)) / (H.x*H.x);
                double S2 = (U1(I,J-1,0)-2*U1(I,J,0)+OtherXZ(I,0)) / (H.y*H.y);
                double S3 = (           -2*U1(I,J,0)+U1(I,J,1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,0) = 2*U1(I,J,0) - U0(I,J,0) + Tau*Tau*Delta;
            }
            // Bottom line
            #pragma omp for
            for (int I = 1; I < X-1; I++) {
                double S1 = (U1(I-1,J,Z-1)-2*U1(I,J,Z-1)+U1(I+1,J,Z-1)) / (H.x*H.x);
                double S2 = (U1(I,J-1,Z-1)-2*U1(I,J,Z-1)+OtherXZ(I,Z-1)) / (H.y*H.y);
                double S3 = (U1(I,J,Z-2)  -2*U1(I,J,Z-1)              ) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,Z-1) = 2*U1(I,J,Z-1) - U0(I,J,Z-1) + Tau*Tau*Delta;
            }   
        }
        // Left line
        #pragma omp parallel firstprivate(J)
        {
            #pragma omp for nowait
            for (int K = 1; K < Z-1; K++) {
                double Add = OtherXZ(0, K);
                U2(0,J,K) += Tau*Tau*Add/(H.y*H.y);
            }
            // Right line
            #pragma omp for 
            for (int K = 1; K < Z-1; K++) {
                double Add = OtherXZ(X-1,K);
                U2(X-1,J,K) += Tau*Tau*Add/(H.y*H.y);
            }
        }

        /*---------------------Fill corner points of XZ-----------------------*/
        // Top left
        double Add = OtherXZ(X-1,0);
        U2(X-1,J,0) += Tau*Tau*Add/(H.y*H.y);
        // Top right
        Add = OtherXZ(0,0);
        U2(0,J,0) += Tau*Tau*Add/(H.y*H.y);
        // Bottom left
        Add = OtherXZ(X-1,Z-1);
        U2(X-1,J,Z-1) += Tau*Tau*Add/(H.y*H.y);
        // Bottom right
        Add = OtherXZ(0,Z-1);
        U2(0,J,Z-1) += Tau*Tau*Add/(H.y*H.y);
    }
}

void updateXZLeft(double *U2, double *U1, double *U0,
                   const DivisionInfo &DI, const HValues &H,
                   const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;
    

 // Exchange borders for X
    int LeftRankY = getNeighbourRank(DI, DIM_Y, LEFT);
    int RightRankY = getNeighbourRank(DI, DIM_Y, RIGHT);
    double *OwnXZ = Wall.Own;
    double *OtherXZ = Wall.Other;

    // Fill right XZ dimension
    #pragma omp parallel for
    for (int I = 0; I < X; I++)
        for (int K = 0; K < Z; K++)
            OwnXZ(I, K) = U1(I, Y-1, K);
    // Send right XZ to the right neighbour. Receive right YZ from the left neighbour.
    // Recalculate left Y border.
    int Count = DI.Volume.X * DI.Volume.Z;
    MPI_Status Status;
    MPI_Sendrecv(OwnXZ, Count, MPI_DOUBLE, RightRankY, 0,
                 OtherXZ, Count, MPI_DOUBLE, LeftRankY, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate left border.
    if (DI.Coord.Y == 0) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++)
            for (int K = 0; K < Z; K++)
                U2(I, 0, K) = 0;
    } else {
        /*--------------------Fill inner points of XZ-------------------------*/
        int J = 0;
        #pragma omp parallel firstprivate(J)
        {
            #pragma omp for nowait
            for (int I = 1; I < X-1; I++) {
                for (int K = 1; K < Z-1; K++) {
                    double S1 = (U1(I-1,J,K)-2*U1(I,J,K)+U1(I+1,J,K)) / (H.x*H.x);
                    double S2 = (OtherXZ(I,K)-2*U1(I,J,K)+U1(I,J+1,K)) / (H.y*H.y);
                    double S3 = (U1(I,J,K-1)-2*U1(I,J,K)+U1(I,J,K+1)) / (H.z*H.z);
                    double Delta = S1 + S2 + S3;
                    U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
                }
            }

            /*----------Fill border points of XZ, exclude corner points-----------*/
            // Top line
            #pragma omp for nowait
            for (int I = 1; I < X-1; I++) {
                double S1 = (U1(I-1,J,0)-2*U1(I,J,0)+U1(I+1,J,0)) / (H.x*H.x);
                double S2 = (OtherXZ(I,0)-2*U1(I,J,0)+U1(I,J+1,0)) / (H.y*H.y);
                double S3 = (           -2*U1(I,J,0)+U1(I,J,1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,0) = 2*U1(I,J,0) - U0(I,J,0) + Tau*Tau*Delta;
            }
            // Bottom line
            #pragma omp for
            for (int I = 1; I < X-1; I++) {
                double S1 = (U1(I-1,J,Z-1)-2*U1(I,J,Z-1)+U1(I+1,J,Z-1)) / (H.x*H.x);
                double S2 = (OtherXZ(I,Z-1)-2*U1(I,J,Z-1)+U1(I,J+1,Z-1)) / (H.y*H.y);
                double S3 = (U1(I,J,Z-2)  -2*U1(I,J,Z-1)              ) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,Z-1) = 2*U1(I,J,Z-1) - U0(I,J,Z-1) + Tau*Tau*Delta;
            }
        }
        #pragma omp parallel firstprivate(J)
        {
            // Left line
            #pragma omp for nowait
            for (int K = 1; K < Z-1; K++) {
                double Add = OtherXZ(0, K);
                U2(0,J,K) += Tau*Tau*Add/(H.y*H.y);
            }
            // Right line
            #pragma omp for
            for (int K = 1; K < Z-1; K++) {
                double Add = OtherXZ(X-1,K);
                U2(X-1,J,K) += Tau*Tau*Add/(H.y*H.y);
            }
        }

        /*---------------------Fill corner points of XZ-----------------------*/
        // Top left
        double Add = OtherXZ(X-1,0);
        U2(X-1,J,0) += Tau*Tau*Add/(H.y*H.y);
        // Top right
        Add = OtherXZ(0,0);
        U2(0,J,0) += Tau*Tau*Add/(H.y*H.y);
        // Bottom left
        Add = OtherXZ(X-1,Z-1);
        U2(X-1,J,Z-1) += Tau*Tau*Add/(H.y*H.y);
        // Bottom right
        Add = OtherXZ(0,Z-1);
        U2(0,J,Z-1) += Tau*Tau*Add/(H.y*H.y);
    }
}

void updateXYRight(double *U2, double *U1, double *U0,
                   const DivisionInfo &DI, const HValues &H,
                   const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;

    // Exchange borders for Z
    int LeftRankZ = getNeighbourRank(DI, DIM_Z, LEFT);
    int RightRankZ = getNeighbourRank(DI, DIM_Z, RIGHT);
    double *OwnXY = Wall.Own;
    double *OtherXY = Wall.Other;

    // Fill left XY dimension
    if (DI.Coord.Z == 0) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++)
            for (int J = 0; J < Y; J++)
                OwnXY(I, J) = U1(I, J, 1);
    } else {
        #pragma omp parallel for
        for (int I = 0; I < X; I++)
            for (int J = 0; J < Y; J++)
                OwnXY(I, J) = U1(I, J, 0);
    }
    // Send left XY to the left neighbour. Receive right XY from the right neighbour.
    // Recalculate right Z border.
    int Count = DI.Volume.X * DI.Volume.Y;
    MPI_Status Status;
    MPI_Sendrecv(OwnXY, Count, MPI_DOUBLE, LeftRankZ, 0,
                 OtherXY, Count, MPI_DOUBLE, RightRankZ, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate right border.
    int K = Z-1;
    #pragma omp parallel for firstprivate(K)
    for (int I = 1; I < X-1; I++) {
        for (int J = 1; J < Y-1; J++) {
            double Sx = (U1(I-1,J,K) - 2*U1(I,J,K) + U1(I+1, J, K)) / (H.x*H.x);
            double Sy = (U1(I,J-1,K) - 2*U1(I,J,K) + U1(I, J+1, K)) / (H.y*H.y);
            // OtherXY(I,J) == U1(I,J,1)
            double Sz = (U1(I,J,K-1) - 2*U1(I,J,K) + OtherXY(I,J)) / (H.z*H.z);
            double Delta = Sx + Sy + Sz;
            U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
        }
    }
    /*----------Fill border points of XY, exclude corner points-----------*/
    if (DI.Coord.Z < DI.CubeNum.Z-1) {
        // Top line
        if (DI.Coord.X == 0) {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++)
               U2(0,J,K) = 0;
        } else {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++) {
                double Add = OtherXY(0,J);
                U2(0,J,K) += Tau*Tau*Add/(H.z*H.z);
            }
        }
        // Bottom line
        if (DI.Coord.X < DI.CubeNum.X-1) {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++) {
                double Add = OtherXY(X-1, J);
                U2(X-1,J,K) += Tau*Tau*Add/(H.z*H.z);
            }
        } else {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++) {
                U2(X-1,J,K) = 0;
            }
        }
        // Left line
        if (DI.Coord.Y > 0) {
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++) {
                double Add = OtherXY(I, 0);
                U2(I,0,K) += Tau*Tau*Add/(H.z*H.z);
            }
        } else {
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++)
                U2(I,0,K) = 0;
        }
        // Right line
        if (DI.Coord.Y < DI.CubeNum.Y-1) {
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++) {
                double Add = OtherXY(I, Y-1);
                U2(I,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
            }
        } else {
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++)
                U2(I,Y-1,K) = 0;
        }
    } else {
        // Top line
        if (DI.Coord.X > 0) {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++) {
                U2(0,J,K) = 0;
            }
        }
        // Bottom line
        if (DI.Coord.X < DI.CubeNum.X-1) {
            #pragma omp parallel for firstprivate(K)
            for (int J = 1; J < Y-1; J++) {
                U2(X-1,J,K) = 0;
            }
        }
        // Left line
        if (DI.Coord.Y > 0) {
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++) {
                U2(I,0,K) = 0;
            }
        }
        if (DI.Coord.Y < DI.CubeNum.Y-1) {
            // Right line
            #pragma omp parallel for firstprivate(K)
            for (int I = 1; I < X-1; I++) {
                U2(I,Y-1,K) = 0;
            }
        }
    }
    /*---------------------Fill corner points of XY-----------------------*/
    // Top left
    double Add = OtherXY(0,0);
    U2(0,0,K) += Tau*Tau*Add/(H.z*H.z);
    // Top right
    Add = OtherXY(0,Y-1);
    U2(0,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
    // Bottom left
    Add = OtherXY(X-1,0);
    U2(X-1,0,K) += Tau*Tau*Add/(H.z*H.z);
    // Bottom right
    Add = OtherXY(X-1,Y-1);
    U2(X-1,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
}

void updateXYLeft(double *U2, double *U1, double *U0,
                  const DivisionInfo &DI, const HValues &H,
                  const Walls &Wall, double Tau) {
    const int &X = DI.Volume.X;
    const int &Y = DI.Volume.Y;
    const int &Z = DI.Volume.Z;
    

    // Exchange borders for Z
    int LeftRankZ = getNeighbourRank(DI, DIM_Z, LEFT);
    int RightRankZ = getNeighbourRank(DI, DIM_Z, RIGHT);
    double *OwnXY = Wall.Own;
    double *OtherXY = Wall.Other;

    // Fill right XY dimension
    if (DI.Coord.Z == DI.CubeNum.Z-1) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++)
            for (int J = 0; J < Y; J++)
                OwnXY(I, J) = U2(I, J, Z-1); // important!
    } else {
        #pragma omp parallel for
        for (int I = 0; I < X; I++)
            for (int J = 0; J < Y; J++)
                OwnXY(I, J) = U1(I, J, Z-1);
    }
    // Send right XY to the right neighbour. Receive left XY from the left neighbour.
    // Recalculate left Z border.
    int Count = DI.Volume.X * DI.Volume.Y;
    MPI_Status Status;
    MPI_Sendrecv(OwnXY, Count, MPI_DOUBLE, RightRankZ, 0,
                 OtherXY, Count, MPI_DOUBLE, LeftRankZ, 0,
                 MPI_COMM_WORLD, &Status);
    // Calculate left border.
    if (DI.Coord.Z == 0) {
        #pragma omp parallel for
        for (int I = 0; I < X; I++) {
            for (int J = 0; J < Y; J++) {
                U2(I,J,0) = OtherXY(I,J);
            }
        }
    } else {
        /*--------------------Fill inner points of XY-------------------------*/
        int K = 0;
        #pragma omp parallel for firstprivate(K)
        for (int I = 1; I < X-1; I++) {
            for (int J = 1; J < Y-1; J++) {
                double S1 = (U1(I-1,J,K)-2*U1(I,J,K)+U1(I+1,J,K)) / (H.x*H.x);
                double S2 = (U1(I,J-1,K)-2*U1(I,J,K)+U1(I,J+1,K)) / (H.y*H.y);
                double S3 = (OtherXY(I,J)-2*U1(I,J,K)+U1(I,J,K+1)) / (H.z*H.z);
                double Delta = S1 + S2 + S3;
                U2(I,J,K) = 2*U1(I,J,K) - U0(I,J,K) + Tau*Tau*Delta;
            }
        }

        /*----------Fill border points of XY, exclude corner points-----------*/
        // Top line
        #pragma omp parallel for firstprivate(K)
        for (int J = 1; J < Y-1; J++) {
            double Add = OtherXY(0, J);
            U2(0,J,K) += Tau*Tau*Add/(H.z*H.z);
        }
        // Bottom line
        #pragma omp parallel for firstprivate(K)
        for (int J = 1; J < Y-1; J++) {
            double Add = OtherXY(X-1, J);
            U2(X-1,J,K) += Tau*Tau*Add/(H.z*H.z);
        }
        // Left line
        #pragma omp parallel for firstprivate(K)
        for (int I = 1; I < X-1; I++) {
            double Add = OtherXY(I, 0);
            U2(I,0,K) += Tau*Tau*Add/(H.z*H.z);
        }
        // Right line
        #pragma omp parallel for firstprivate(K)
        for (int I = 1; I < X-1; I++) {
            double Add = OtherXY(I,Y-1);
            U2(I,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
        }

        /*---------------------Fill corner points of XY-----------------------*/
        // Top left
        double Add = OtherXY(0,0);
        U2(0,0,K) += Tau*Tau*Add/(H.z*H.z);
        // Top right
        Add = OtherXY(0,Y-1);
        U2(0,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
        // Bottom left
        Add = OtherXY(X-1,0);
        U2(X-1,0,K) += Tau*Tau*Add/(H.z*H.z);
        // Bottom right
        Add = OtherXY(X-1,Y-1);
        U2(X-1,Y-1,K) += Tau*Tau*Add/(H.z*H.z);
    }
}

void updateTopLayer(double *U2, double *U1, double *U0,
                    DivisionInfo &DI, const HValues &H,
                    const Walls &Wall, double Tau) {
    calculateInnerPoints(U2, U1, U0, DI, H, Tau);
    updateYZRight(U2, U1, U0, DI, H, Wall, Tau);
    updateYZLeft(U2, U1, U0, DI, H, Wall, Tau);
    updateXZRight(U2, U1, U0, DI, H, Wall, Tau);
    updateXZLeft(U2, U1, U0, DI, H, Wall, Tau);
    updateXYRight(U2, U1, U0, DI, H, Wall, Tau);
    updateXYLeft(U2, U1, U0, DI, H, Wall, Tau);
}

double getDelta(double *U2, const DivisionInfo &DI, double tn, LValues &L, HValues &H, Values &Vals) {
    double MaxDelta = 0;
    int X = DI.Volume.X;
    int Y = DI.Volume.Y;
    int Z = DI.Volume.Z;

    for (int I = 0; I < X; I++) {
        double Xi = XI(I);
        for (int J = 0; J < Y; J++) {
            double Yj = YJ(J);
            for (int K = 0; K < Z; K++) {
                double Zk = ZK(K);
                double Delta = fabs(u(Xi,Yj,Zk,tn,L) - U2(I,J,K));
                if (Delta > MaxDelta)
                    MaxDelta = Delta;
            }
        }
    }
    return MaxDelta;
}

#ifdef DEBUG
void printMatrix(double *A, DivisionInfo &DI, double tn, LValues &L, Values &Vals) {
    int X = DI.Volume.X;
    int Y = DI.Volume.Y;
    int Z = DI.Volume.Z;
    for (int I = 0; I < X; I++) {
        printf("[\n");
        for (int J = 0; J < Y; J++) {
            //printf("\t[ ");
            //for (int K = Z-1; K < Z; K++) 
            {
                int K = Z-1;
                double Xi = Vals.Xi[I];
                double Yj = Vals.Yj[J];
                double Zk = Vals.Zk[K];
                //printf("%6.3lf {%6.3lf, %6.3lf}", u(Xi,Yj,Zk,tn,L) - A(I,J,K),
                // u(Xi,Yj,Zk,tn,L), A(I,J,K));
                printf("%5.4lf ", u(Xi,Yj,Zk,tn,L) - A(I,J,K));
                //printf("%6.3lf ", u(Xi,Yj,Zk,tn,L));
            }
            //printf("]\n");
        }
        printf("]\n");
    }
}
#endif

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Not enough arguments. Usage: ./task3 <Lx> <Ly> <Lz> <N> <T>"
                  << std::endl;
        std::exit(1);
    }
    double Lx = std::atof(argv[1]);
    double Ly = std::atof(argv[2]);
    double Lz = std::atof(argv[3]);
    int N = std::atoi(argv[4]);
    int T = std::atoi(argv[5]);
    LValues L = {Lx, Ly, Lz};
    int ProcNum, ProcRank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (N != 128 && N != 256 && N != 512) {
        if (ProcRank == 0)
            std::cerr << "Invalid number of points: " << N << std::endl;
        exit(1);
    }
    if (ProcRank == 0) {
        printf("MPI proc number: %d\n", ProcNum);
    }
    #pragma omp parallel
    {
        #pragma omp master
        if (ProcRank == 0)
            printf("OpenMP threads: %d\n", omp_get_num_threads());
    }
    double Time1 = MPI_Wtime();
    DivisionInfo DI(N+1, ProcRank, ProcNum);
    PRINT_DEBUG(
        std::stringstream SS;
        SS << "[" << ProcRank << "] DivisionInfo:\n";
        SS << DI;
        dbgs() << SS.str();
    );

    double Hx = L.x / double(N);
    double Hy = L.y / double(N);
    double Hz = L.z / double(N);

    HValues H = {Hx, Hy, Hz};

    double stabCond = 1 / (getAt(L) * sqrt(1/(H.x*H.x) + 1/(H.y*H.y) + 1/(H.z*H.z)));
    int K = T / stabCond;
    double Tau = T / (double) K;

    double *Xi = new double[DI.Volume.X];
    double *Yi = new double[DI.Volume.Y];
    double *Zi = new double[DI.Volume.Z];
    
    for (int I = 0; I < DI.Volume.X; I++)
        Xi[I] = (I + DI.Volume.RangeX.Min) * H.x;
    for (int I = 0; I < DI.Volume.Y; I++)
        Yi[I] = (I + DI.Volume.RangeY.Min) * H.y;
    for (int I = 0; I < DI.Volume.Z; I++)
        Zi[I] = (I + DI.Volume.RangeZ.Min) * H.z;

    Values Vals(Xi, Yi, Zi);

    double *ZeroLayer = initZeroLayer(DI, L, H, Vals);
    double *FirstLayer = initFirstLayer(ZeroLayer, DI, L, H, Vals, Tau);
    double *TopLayer = new double[DI.Volume.X * DI.Volume.Y * DI.Volume.Z];
    int MaxSize = DI.Volume.X;
    if (DI.Volume.Y > MaxSize)
        MaxSize = DI.Volume.Y;
    if (DI.Volume.Z > MaxSize)
        MaxSize = DI.Volume.Z;
    double *Own = new double[MaxSize * MaxSize];
    double *Other = new double[MaxSize * MaxSize];
    Walls Wall = {Own, Other};

    double Delta = getDelta(ZeroLayer, DI, Tau*0, L, H, Vals);
    double Max = 0;
    MPI_Reduce(&Delta, &Max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("Max delta [%d]: %.15lf\n", 0, Max);
    }

    Delta = getDelta(FirstLayer, DI, Tau*1, L, H, Vals);
    Max = 0;
    MPI_Reduce(&Delta, &Max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("Max delta [%d]: %.15lf\n", 1, Max);
    }

    for (int n = 2; n <= 20; n++) {
        updateTopLayer(TopLayer, FirstLayer, ZeroLayer, DI, H, Wall, Tau);
        Delta = getDelta(TopLayer, DI, Tau*n, L, H, Vals);
        double Max = 0;
        MPI_Reduce(&Delta, &Max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (ProcRank == 0) {
            printf("Max delta [%d]: %.15lf\n", n, Max);
        }
        double *Top = TopLayer;
        TopLayer = ZeroLayer; // will be rewritten on the next step
        ZeroLayer = FirstLayer;
        FirstLayer = Top;
    }
    double Time2 = MPI_Wtime();
    double Time = Time2-Time1;
    double MaxTime = 0;
    MPI_Reduce(&Time, &MaxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("Total time: %.4lf\n", MaxTime);
    }
    delete[] Own;
    delete[] Other;
    delete[] TopLayer;
    delete[] FirstLayer;
    delete[] ZeroLayer;
    delete[] Xi;
    delete[] Yi;
    delete[] Zi;
    MPI_Finalize();
    return 0;
}