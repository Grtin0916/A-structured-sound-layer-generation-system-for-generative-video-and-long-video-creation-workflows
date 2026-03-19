#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix3f A;
    A << 1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f;

    Eigen::Vector3f x;
    x << 1.0f, 0.5f, -1.0f;

    Eigen::Vector3f y = A * x;

    std::cout << "A =\n" << A << "\n\n";
    std::cout << "x =\n" << x << "\n\n";
    std::cout << "y = A * x =\n" << y << "\n";

    return 0;
}