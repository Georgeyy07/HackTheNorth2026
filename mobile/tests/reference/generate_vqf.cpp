// Generate with unmodified upstream v2.1.2 basicvqf.cpp/.hpp:
// clang++ -std=c++11 -I/path/to/upstream generate_vqf.cpp /path/to/upstream/basicvqf.cpp -o /tmp/vqf-ref
// /tmp/vqf-ref > tests/reference/vqf.json
#include "basicvqf.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
int main() {
    BasicVQF filter(0.01);
    std::cout << std::setprecision(17) << "[\n";
    bool first = true;
    for (int i = 0; i < 4000; ++i) {
        if (i == 2000) filter.resetState();
        double t = i * 0.01;
        double gyr[] = {0.1*std::sin(t), 0.07*std::cos(0.3*t), 0.2};
        double acc[] = {0.4*std::sin(2*t), 9.80665 + 0.3*std::cos(t), 0.6*std::sin(t)};
        filter.update(gyr, acc);
        if (i % 37 == 0 || i == 299 || i == 300 || i == 2000 || i == 2299 || i == 2300 || i == 3999) {
            double q[4]; filter.getQuat6D(q);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "  {\"i\":" << i << ",\"q\":[" << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "]}";
        }
    }
    std::cout << "\n]\n";
}
