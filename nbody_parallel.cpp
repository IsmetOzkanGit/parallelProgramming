#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <thread>

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

void compute_forces(std::vector<Particle>& particles) {
    const double G = 6.67430e-11; // Gravitational constant
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].vx = particles[i].vy = particles[i].vz = 0.0; // Reset velocities
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i != j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > 1e-9) { // Avoid division by zero
                    double F = (G * particles[i].mass * particles[j].mass) / (dist * dist);
                    particles[i].vx += F * dx / dist;
                    particles[i].vy += F * dy / dist;
                    particles[i].vz += F * dz / dist;
                }
            }
        }
    }
}

void update_positions(std::vector<Particle>& particles, double dt) {
    #pragma omp parallel for
    for (auto& p : particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

int main() {
    const int num_particles = 1000;
    const double dt = 0.01;
    std::vector<Particle> particles(num_particles);

    // Initialize particles with positions closer together
    double pos_range = 0.01; // Reduced range for closer initial positions
    for (auto& p : particles) {
        p.x = (rand() / (double)RAND_MAX) * pos_range;
        p.y = (rand() / (double)RAND_MAX) * pos_range;
        p.z = (rand() / (double)RAND_MAX) * pos_range;
        p.mass = rand() / (double)RAND_MAX + 1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Simulation loop
    const int num_steps = 100;
    for (int step = 0; step < num_steps; ++step) {
        compute_forces(particles);
        update_positions(particles, dt);

        // Print positions of the first 10 particles (or adjust as needed)
        std::cout << "Step " << step << ":\n";
        for (int i = 0; i < 10 && i < num_particles; ++i) {
            std::cout << "Particle " << i << ": (" 
                      << particles[i].x << ", "
                      << particles[i].y << ", "
                      << particles[i].z << ")\n";
        }
        std::cout << "\n";

        // Sleep to simulate real-time visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjust delay as needed
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Parallel execution time: " << elapsed.count() << " seconds\n";

    return 0;
}
