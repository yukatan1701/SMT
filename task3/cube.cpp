#include "common.h"
#include <iostream>
#include <cstdlib>

CubeVolume::PointNumber CubeVolume::getPointNumberInfo(int N, int BlockNum) const {
    int Inner = N / BlockNum;
    int Outer = N - Inner * BlockNum;
    PointNumber PN;
    PN.Left = Outer / 2;
    PN.Right = Outer - PN.Left;
    PN.Center = Inner;
    return PN;
}

VolumeRangePair CubeVolume::getCoordinateVolume(int Coord, int CubeNum, int N) const {
    PointNumber PN = getPointNumberInfo(N, CubeNum);
    IndexRange Range;
    int VolCoord = PN.Center;
    if (Coord == 0) {
        VolCoord += PN.Left;
        Range.Min = 0;
        Range.Max = VolCoord - 1;
    } else if (Coord == CubeNum - 1) {
        VolCoord += PN.Right;
        assert(CubeNum >= 2);
        Range.Min = (PN.Center+PN.Left) + PN.Center * (CubeNum - 2);
        Range.Max = N-1;
    } else {
        Range.Min = (PN.Center+PN.Left) + PN.Center * (Coord - 1);
        Range.Max = (PN.Center+PN.Left) + PN.Center * Coord - 1;
    }
    return std::make_pair(VolCoord, Range);
}

CubeVolume::CubeVolume(const CubeCoordinate &CC, const CubeNumber &CN, int N) {
    VolumeRangePair XInfo = getCoordinateVolume(CC.X, CN.X, N);
    VolumeRangePair YInfo = getCoordinateVolume(CC.Y, CN.Y, N);
    VolumeRangePair ZInfo = getCoordinateVolume(CC.Z, CN.Z, N);
    X = XInfo.first;
    RangeX = XInfo.second;
    Y = YInfo.first;
    RangeY = YInfo.second;
    Z = ZInfo.first;
    RangeZ = ZInfo.second;
}

DivisionInfo::DivisionInfo(int N, int ProcRank, int ProcNum) : Rank(ProcRank) {
    switch (ProcNum)
    {
    case 10:
        CubeNum = CubeNumber(5, 2, 1);
        break;
    case 20:
        CubeNum = CubeNumber(5, 2, 2);
        break;
    case 40:
        CubeNum = CubeNumber(5, 2, 4);
        break;
    case 1:
        CubeNum = CubeNumber(1, 1, 1);
        break;
    case 2:
        CubeNum = CubeNumber(2, 1, 1);
        break;
    case 4:
        CubeNum = CubeNumber(2, 2, 1);
        break;
    case 8:
        CubeNum = CubeNumber(2, 2, 2);
        break;
    case 16:
        CubeNum = CubeNumber(4, 2, 2);
        break;
    case 32:
        CubeNum = CubeNumber(4, 4, 2);
        break;
    case 64:
        CubeNum = CubeNumber(4, 4, 4);
        break;
    case 128:
        CubeNum = CubeNumber(4, 4, 8);
        break;
    case 256:
        CubeNum = CubeNumber(8, 4, 8);
        break;
    default:
        if (ProcRank == 0)
            std::cerr << "Invalid number of processors: " << ProcNum << std::endl;
        exit(2);
    }
    Coord = CubeCoordinate(ProcRank % CubeNum.X,
                           (ProcRank % (CubeNum.X * CubeNum.Y)) / CubeNum.X,
                           ProcRank / (CubeNum.X * CubeNum.Y));
    Volume = CubeVolume(Coord, CubeNum, N);
}
