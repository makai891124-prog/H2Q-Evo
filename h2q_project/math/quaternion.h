#ifndef H2Q_PROJECT_MATH_QUATERNION_H
#define H2Q_PROJECT_MATH_QUATERNION_H

#include <cmath>

namespace h2q_project {

// Using a smaller quaternion implementation to reduce memory footprint.
// This assumes a 'tiny_quaternion' library is available.  For example, Eigen's Quaternion.
// This is a placeholder; replace with your chosen library's include.
#include "Eigen/Core"
#include "Eigen/Geometry"


class Quaternion {
public:
    // Default constructor
    Quaternion() : q(Eigen::Quaternionf::Identity()) {}

    // Constructor with scalar and vector parts
    Quaternion(float w, float x, float y, float z) : q(w, x, y, z) {}

    // Constructor from axis-angle representation (example using Eigen)
    Quaternion(const Eigen::Vector3f& axis, float angle) : q(Eigen::AngleAxisf(angle, axis)) {}

    // Access to the underlying Eigen quaternion (for advanced usage/interoperability)
    Eigen::Quaternionf& eigen() { return q; }
    const Eigen::Quaternionf& eigen() const { return q; }

    // Identity quaternion
    static Quaternion identity() { return Quaternion(); }

    // Getters
    float w() const { return q.w(); }
    float x() const { return q.x(); }
    float y() const { return q.y(); }
    float z() const { return q.z(); }

    // Method to normalize the quaternion
    void normalize() { q.normalize(); }

    // Method to conjugate the quaternion
    Quaternion conjugate() const { return Quaternion(q.w(), -q.x(), -q.y(), -q.z()); }

    // Quaternion multiplication
    Quaternion operator*(const Quaternion& other) const {
        return Quaternion((q * other.q).w(), (q * other.q).x(), (q * other.q).y(), (q * other.q).z());
    }

    // Rotate a vector by the quaternion
    Eigen::Vector3f rotate(const Eigen::Vector3f& v) const {
        return q._transformVector(v);
    }

private:
    Eigen::Quaternionf q; // The underlying quaternion representation
};

} // namespace h2q_project

#endif // H2Q_PROJECT_MATH_QUATERNION_H
