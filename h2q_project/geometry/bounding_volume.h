#ifndef H2Q_PROJECT_GEOMETRY_BOUNDING_VOLUME_H
#define H2Q_PROJECT_GEOMETRY_BOUNDING_VOLUME_H

#include "h2q_project/geometry/vector3.h"

namespace H2Q
{

class BoundingVolume
{
public:
    BoundingVolume() = default;
    BoundingVolume(const Vector3& min, const Vector3& max) : min_(min), max_(max) {}

    const Vector3& getMin() const { return min_; }
    const Vector3& getMax() const { return max_; }

private:
    Vector3 min_;
    Vector3 max_;
};

}

#endif // H2Q_PROJECT_GEOMETRY_BOUNDING_VOLUME_H
