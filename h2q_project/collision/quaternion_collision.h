#ifndef H2Q_PROJECT_COLLISION_QUATERNION_COLLISION_H
#define H2Q_PROJECT_COLLISION_QUATERNION_COLLISION_H

#include <vector>
#include "h2q_project/geometry/quaternion.h"
#include "h2q_project/geometry/bounding_volume.h"

namespace H2Q
{

class QuaternionCollision
{
public:
    /**
     * @brief Checks for collision between two bounding volumes using quaternion rotations.
     *
     * This function performs a fast collision check using quaternion-based rotations
     * to determine potential overlaps between the bounding volumes.
     *
     * @param bv1 The first bounding volume.
     * @param bv2 The second bounding volume.
     * @return True if a collision is detected, false otherwise.
     */
    static bool checkCollision(const BoundingVolume& bv1, const BoundingVolume& bv2);
};

}

#endif // H2Q_PROJECT_COLLISION_QUATERNION_COLLISION_H
