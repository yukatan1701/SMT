#include <mpich/mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>

const uint64_t Delta = 1000;

const double IExp = 1.0 / 364.0;

double F(double x, double y, double z) {
  // if point is in G
  if (0.0 <= x && x <= 1.0 && 0.0 <= y && y <= x && 0.0 <= z && z <= x*y) {
    return x * y*y * z*z*z;
  }
  return 0.0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Not enough arguments." << std::endl;
    return -1;
  }
  double Eps = std::atof(argv[1]);
  int ProcNum, ProcRank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double Time1 = MPI_Wtime();
  uint64_t CurrentPointNumber = Delta;
  uint64_t TotalPointNumber = 0;
  double OverallSum = 0.0;
  std::srand(std::time(NULL));
  double ICalc = 0.0f;
  while (true) {
    TotalPointNumber += CurrentPointNumber;
    double Sum = 0.0;
    for (uint64_t I = 0; I < CurrentPointNumber; I++) {
      double RX = std::rand() / double(RAND_MAX);
      double RY = std::rand() / double(RAND_MAX);
      double RZ = std::rand() / double(RAND_MAX);
      Sum += F(RX, RY, RZ);
    }
    double TotalSum = 0.0;
    MPI_Reduce(&Sum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    OverallSum += TotalSum;
    ICalc = OverallSum / double(TotalPointNumber * ProcNum);
    int Continue = 1;
    if (ProcRank == 0) {
      double Diff = std::abs(IExp - ICalc);
      if (Diff < Eps)
        Continue = 0;
    }
    MPI_Bcast(&Continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (Continue == 0) {
      break;
    }
  }
  double Time2 = MPI_Wtime();
  std::stringstream Results;
  if (ProcRank == 0) {
    Results << "Calculated value: " << ICalc << "\n";
    Results << "Expected value: " << IExp << "\n";
    Results << "Delta: " << std::abs(IExp - ICalc) << "\n";
    Results << "Points: " << TotalPointNumber * ProcNum << "\n";
  }
  double Time = Time2 - Time1;
  double *Times = new double[ProcNum];
  MPI_Gather(&Time, 1, MPI_DOUBLE, Times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (ProcRank == 0) {
    double MaxTime = 0.0;
    for (int I = 0; I < ProcNum; I++) {
      if (Times[I] > MaxTime) {
        MaxTime = Times[I];
      }
    }
    Results << "MaxTime: " << MaxTime << "\n";
    std::cout << Results.str();
  }
  MPI_Finalize();
  return 0;
}
