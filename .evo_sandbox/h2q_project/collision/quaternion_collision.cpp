#include "h2q_project/collision/quaternion_collision.h"
#include "h2q_project/geometry/bounding_volume.h"
#include "h2q_project/geometry/quaternion.h"
#include <iostream>

namespace H2Q
{

bool QuaternionCollision::checkCollision(const BoundingVolume& bv1, const BoundingVolume& bv2)
{
    // Simple AABB collision check (can be replaced with more sophisticated quaternion-based methods)
    // This is a placeholder implementation.

    //Get the min and max points for both bounding volumes
    const auto& min1 = bv1.getMin();
    const auto& max1 = bv1.getMax();
    const auto& min2 = bv2.getMin();
    const auto& max2 = bv2.getMax();

    // Check for overlap in each dimension
    if (max1.x < min2.x || min1.x > max2.x)
        return false;
    if (max1.y < min2.y || min1.y > max2.y)
        return false;
    if (max1.z < min2.z || min1.z > max2.z)
        return false;

    return true; // AABB Collision detected
}

}
