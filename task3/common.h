#ifndef COMMON_H
#define COMMON_H

#include <sstream>
#include <assert.h>

#define dbgs() std::cerr

#ifdef DEBUG
#define PRINT_DEBUG(x) do { x; } while (0);
#else
#define PRINT_DEBUG(x) do {} while (0);
#endif

struct LValues {
    double x;
    double y;
    double z;
};

struct HValues {
    double x;
    double y;
    double z;
};

// count of cubes in the division
struct CubeNumber {
    int X;
    int Y;
    int Z;
    CubeNumber() : X(0), Y(0), Z(0) {}
    CubeNumber(int X, int Y, int Z) : X(X), Y(Y), Z(Z) {}
    friend std::ostream& operator<<(std::ostream& OS, const CubeNumber& CN) {
        OS << "[" << CN.X << ", " << CN.Y << ", " << CN.Z << "]";
        return OS;
    }
};

// a coordinate for a cube for the current process rank
struct CubeCoordinate {
    int X;
    int Y;
    int Z;
    CubeCoordinate() : X(0), Y(0), Z(0) {}
    CubeCoordinate(int X, int Y, int Z) : X(X), Y(Y), Z(Z) {}
    friend std::ostream& operator<<(std::ostream& OS, const CubeCoordinate& CC) {
        OS << "[" << CC.X << ", " << CC.Y << ", " << CC.Z << "]";
        return OS;
    }
};

struct IndexRange {
    int Min;
    int Max;
    IndexRange() : Min(0), Max(0) {}
    IndexRange(int Min, int Max) : Min(Min), Max(Max) {}
    friend std::ostream& operator<<(std::ostream& OS, const IndexRange& IR) {
        OS << "(" << IR.Min << ", " << IR.Max << ")";
        return OS;
    }
};

typedef std::pair<int, IndexRange> VolumeRangePair;

// number of points for every dimension
struct CubeVolume {
    int X;
    int Y;
    int Z;
    IndexRange RangeX;
    IndexRange RangeY;
    IndexRange RangeZ;
    CubeVolume() : X(0), Y(0), Z(0) {}
    CubeVolume(const CubeCoordinate &CC, const CubeNumber &CN, int N);
    friend std::ostream& operator<<(std::ostream& OS, const CubeVolume& CV) {
        OS << "[" << CV.X << ", " << CV.Y << ", " << CV.Z << "], "
           << CV.RangeX << ", " << CV.RangeY << ", " << CV.RangeZ;
        return OS;
    }
private:
    struct PointNumber {
        int Left;
        int Center;
        int Right;
    };
    PointNumber getPointNumberInfo(int N, int BlockNum) const;
    VolumeRangePair getCoordinateVolume(int Coord, int CubeNum, int N) const;
};

class DivisionInfo {
public:
    CubeVolume Volume;
    CubeCoordinate Coord;
    CubeNumber CubeNum;
    int Rank;
    DivisionInfo(int N, int ProcRank, int ProcNum);
    friend std::ostream& operator<<(std::ostream& OS, const DivisionInfo& DI) {
        OS << "\tVolume:\t" << DI.Volume
           << "\n\tCoordinate:\t" << DI.Coord
           << "\n\tCube number:\t" << DI.CubeNum << "\n";
        return OS;
    }
};

struct Values {
    double *Xi;
    double *Yj;
    double *Zk;
    Values() : Xi(NULL), Yj(NULL), Zk(NULL) {}
    Values(double *Xi, double *Yj, double *Zk) : Xi(Xi), Yj(Yj), Zk(Zk) {}
};

struct Walls {
    double *Own;
    double *Other;
};

#endif