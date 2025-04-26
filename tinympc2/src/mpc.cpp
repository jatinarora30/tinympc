
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib> 
#include <numeric>
#include "/home/jatinarora/osqp/include/osqp.h"    
#include "matplotlibcpp.h"
#include <sys/resource.h>

namespace plt = matplotlibcpp;


// Parameters
struct Parameters {
    int N_Horizon = 30;               // from paper
    int Input_State_Dimension = 12;   // full state: x, y, z, vx, vy, vz, roll, pitch, yaw, and rates
    int Output_State_Dimension = 4;   // motor/thrust commands
    float g = 9.81f;                   // gravity [m/s^2]
    float mass = 0.027f;               // Crazyflie mass [kg]
    float I_x = 1.4e-5f;               // Inertia x [kg m^2]
    float I_y = 1.4e-5f;               // Inertia y [kg m^2]
    float I_z = 2.17e-5f;              // Inertia z [kg m^2]
    float simulation_time = 12.0f;     // total simulation time [s]
    float dt = 0.002f;                 // control timestep [s], paper used 500Hz (dt = 0.002)
    float rho = 0.1f;                  // ADMM penalty parameter
    int max_iterations = 2;           // ADMM iterations per control step, paper uses up to 7
};

inline Eigen::VectorXd figureEightRef(double t, double T_period) {
    double omega = 2.0 * M_PI / T_period;
    double xref = 0.05 * std::sin(omega * t);      // 5 cm amplitude
    double yref = 0.05 * std::sin(2.0 * omega * t);
    Eigen::VectorXd ref = Eigen::VectorXd::Zero(12);
    ref(0) = xref;
    ref(1) = yref;
    ref(2) = 0.4;  // 40 cm constant height
    return ref;
}


double mean(const std::vector<double>& vec) {
    if (vec.empty()) return 0.0;
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

class TinyMPC {
private:
    Eigen::VectorXd u_min, u_max;
    Eigen::VectorXd q_vec, r_vec;
    Parameters params;
    Eigen::MatrixXd A, B;
    std::vector<Eigen::MatrixXd> P, K,M_inv;
    Eigen::MatrixXd Qf, Q, R, Qf_tilde, Q_tilde, R_tilde;
    Eigen::VectorXd r, q,r_tilde,q_tilde;
    Eigen::VectorXd x_current, u_current;
    std::vector<Eigen::VectorXd> x, u, z, w, lambda, mu;

public:
    TinyMPC(const Parameters& _params, const Eigen::MatrixXd& _A, const Eigen::MatrixXd& _B)
        : params(_params), A(_A), B(_B) {
        int n_x = params.Input_State_Dimension;
        int n_u = params.Output_State_Dimension;

        Q = Eigen::MatrixXd::Identity(n_x, n_x);
        Q.diagonal().segment(0, 3).array() = 20;  // Position x, y, z
        Q.diagonal().segment(3, 3).array() = 0.1;   // Velocity vx, vy, vz
        Q.diagonal().segment(6, 3).array() = 0.1;    // Orientation roll, pitch, yaw
        Q.diagonal().segment(9, 3).array() = 0.1;    // Angular velocity
        Qf = Eigen::MatrixXd::Identity(n_x, n_x);
        Qf.diagonal().segment(0, 3).array() = 20;
        Qf.diagonal().segment(3, 3).array() = 0.1;
        Qf.diagonal().segment(6, 3).array() = 0.1;
        Qf.diagonal().segment(9, 3).array() = 0.1;
        R = 0.1 * Eigen::MatrixXd::Identity(n_u, n_u);

        Q_tilde = Q + params.rho * Eigen::MatrixXd::Identity(n_x, n_x);
        Qf_tilde = Qf + params.rho * Eigen::MatrixXd::Identity(n_x, n_x);
        R_tilde = R + params.rho * Eigen::MatrixXd::Identity(n_u, n_u);

        K = std::vector<Eigen::MatrixXd>(params.N_Horizon, Eigen::MatrixXd::Zero(n_u, n_x));
        P = std::vector<Eigen::MatrixXd>(params.N_Horizon + 1, Eigen::MatrixXd::Zero(n_x, n_x));
        M_inv = std::vector<Eigen::MatrixXd>(params.N_Horizon, Eigen::MatrixXd::Zero(n_x, n_x));
        x_current = Eigen::VectorXd::Zero(n_x);
        u_current = Eigen::VectorXd::Zero(n_u);
        r = 0.01*Eigen::VectorXd::Ones(n_u);
        q = 0.01*Eigen::VectorXd::Ones(n_x);

        r_tilde = Eigen::VectorXd::Zero(n_u);
        q_tilde = Eigen::VectorXd::Zero(n_x);


        x = std::vector<Eigen::VectorXd>(params.N_Horizon + 1, Eigen::VectorXd::Zero(n_x));
        z = std::vector<Eigen::VectorXd>(params.N_Horizon + 1, Eigen::VectorXd::Zero(n_x));
        lambda = std::vector<Eigen::VectorXd>(params.N_Horizon + 1, Eigen::VectorXd::Zero(n_x));
        u = std::vector<Eigen::VectorXd>(params.N_Horizon, Eigen::VectorXd::Zero(n_u));
        w = std::vector<Eigen::VectorXd>(params.N_Horizon, Eigen::VectorXd::Zero(n_u));
        mu = std::vector<Eigen::VectorXd>(params.N_Horizon, Eigen::VectorXd::Zero(n_u));

        u_min = Eigen::VectorXd::Constant(n_u, -2);
        u_max = Eigen::VectorXd::Constant(n_u, 2);  // realistic thrust limits in N per motor


        q_vec = Eigen::VectorXd::Zero(n_x);  // linear term in state cost
        r_vec = Eigen::VectorXd::Zero(n_u);  // linear term in input cost

        PreComputation();
    }

    void PreComputation() {
        int N = params.N_Horizon;
        P[N] = Qf_tilde;

        for (int i = N - 1; i >= 0; --i) {
            Eigen::MatrixXd S = R_tilde + B.transpose() * P[i + 1] * B;
            M_inv[i] = S.inverse();
            K[i] = M_inv[i] * B.transpose() * P[i + 1] * A;
            P[i] = Q_tilde + A.transpose() * P[i + 1] * (A - B * K[i]);
        }
    }

    void Solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& ref) {
        int N = params.N_Horizon;
        int n_x = params.Input_State_Dimension;
        int n_u = params.Output_State_Dimension;
    
        x[0] = x0;
    
        std::vector<Eigen::VectorXd> p(N + 1, Eigen::VectorXd::Zero(n_x));
        std::vector<Eigen::VectorXd> d(N, Eigen::VectorXd::Zero(n_u));
    
        for (int iter = 0; iter < params.max_iterations; ++iter) {
            // Backward pass to compute p[k] and d[k] using reference
            p[N] = -Qf_tilde * ref;
            for (int k = N - 1; k >= 0; --k) {
                Eigen::VectorXd qk = -Q_tilde * ref;
                Eigen::VectorXd rk = Eigen::VectorXd::Zero(n_u);  // Assuming no linear term for input
    
                d[k] = M_inv[k] * (B.transpose() * p[k + 1] + rk);
                p[k] = qk + (A - B * K[k]).transpose() * (p[k + 1] - P[k + 1] * B * d[k]) + K[k].transpose() * (R_tilde * d[k] - rk);
            }
    
            // Forward pass using optimal control
            x[0] = x0;
            for (int k = 0; k < N; ++k) {
                u[k] = (-K[k] * x[k] - d[k]).cwiseMax(u_min).cwiseMin(u_max);
                x[k + 1] = A * x[k] + B * u[k];
            }
    
            // Slack and dual updates
            for (int k = 0; k <= N; ++k) {
                z[k] = x[k];
                lambda[k] += params.rho * (x[k] - z[k]);
                q_tilde = lambda[k] - params.rho * z[k];
            }
    
            for (int k = 0; k < N; ++k) {
                w[k] = u[k].cwiseMax(u_min).cwiseMin(u_max);
                mu[k] += params.rho * (u[k] - w[k]);
                r_tilde = mu[k] - params.rho * w[k];
            }
        }
        u_current = u[0];
    }

    Eigen::VectorXd getControl() const { return u_current; }
};

class OSQP_Solver {
    private:
        Parameters params;
        Eigen::MatrixXd A, B, Q, Qf, R;
        Eigen::VectorXd u_current;
    
    public:
        OSQP_Solver(const Parameters& _params, const Eigen::MatrixXd& _A, const Eigen::MatrixXd& _B)
            : params(_params), A(_A), B(_B) {
            int n_x = params.Input_State_Dimension;
            int n_u = params.Output_State_Dimension;
    
            Q = Eigen::MatrixXd::Identity(n_x, n_x);
            Q.diagonal().segment(0, 3).array() = 20;
            Q.diagonal().segment(3, 3).array() = 0.1;
            Q.diagonal().segment(6, 3).array() = 0.1;
            Q.diagonal().segment(9, 3).array() = 0.1;
            Qf = Q;
            R = 0.1 * Eigen::MatrixXd::Identity(n_u, n_u);
    
            u_current = Eigen::VectorXd::Zero(n_u);
        }
    
        void Solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& ref) {
            int N = params.N_Horizon;
            int n_x = params.Input_State_Dimension;
            int n_u = params.Output_State_Dimension;
        
            int nx_total = (N + 1) * n_x;
            int nu_total = N * n_u;
            int n_vars = nx_total + nu_total;
            int n_constraints = N * n_x;
        
            // --- Use SparseMatrix with long long indices (to match c_int = long long) ---
            typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long long> SparseMatrixLL;
            SparseMatrixLL P(n_vars, n_vars);
            Eigen::VectorXd q = Eigen::VectorXd::Zero(n_vars);
        
            std::vector<Eigen::Triplet<double>> triplets;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < n_x; ++j)
                    triplets.emplace_back(i * n_x + j, i * n_x + j, Q(j, j));
                for (int j = 0; j < n_u; ++j)
                    triplets.emplace_back(nx_total + i * n_u + j, nx_total + i * n_u + j, R(j, j));
            }
            for (int j = 0; j < n_x; ++j)
                triplets.emplace_back(N * n_x + j, N * n_x + j, Qf(j, j));
            P.setFromTriplets(triplets.begin(), triplets.end());
        
            for (int i = 0; i < N; ++i)
                q.segment(i * n_x, n_x) = -Q * ref;
            q.segment(N * n_x, n_x) = -Qf * ref;
        
            SparseMatrixLL Aeq(n_constraints, n_vars);
            Eigen::VectorXd beq = Eigen::VectorXd::Zero(n_constraints);
            std::vector<Eigen::Triplet<double>> Aeq_triplets;
        
            for (int i = 0; i < N; ++i) {
                for (int row = 0; row < n_x; ++row) {
                    for (int col = 0; col < n_x; ++col)
                        Aeq_triplets.emplace_back(i * n_x + row, (i + 1) * n_x + col, A(row, col));
                    for (int col = 0; col < n_u; ++col)
                        Aeq_triplets.emplace_back(i * n_x + row, nx_total + i * n_u + col, B(row, col));
                    Aeq_triplets.emplace_back(i * n_x + row, i * n_x + row, -1.0);
                }
            }
            Aeq.setFromTriplets(Aeq_triplets.begin(), Aeq_triplets.end());
            beq.segment(0, n_x) = -A * x0;
        
            Eigen::VectorXd z_min = Eigen::VectorXd::Constant(n_vars, -OSQP_INFTY);
            Eigen::VectorXd z_max = Eigen::VectorXd::Constant(n_vars, OSQP_INFTY);
            Eigen::VectorXd u_min = Eigen::VectorXd::Constant(n_u, -2);
            Eigen::VectorXd u_max = Eigen::VectorXd::Constant(n_u, 2);
            for (int i = 0; i < N; ++i) {
                z_min.segment(nx_total + i * n_u, n_u) = u_min;
                z_max.segment(nx_total + i * n_u, n_u) = u_max;
            }
     
        
            OSQPSettings* settings = (OSQPSettings*)c_malloc(sizeof(OSQPSettings));
            osqp_set_default_settings(settings);
            settings->alpha = 1.0;
            settings->verbose = false;
        
            OSQPData* data = (OSQPData*)c_malloc(sizeof(OSQPData));
            data->n = n_vars;
            data->m = n_constraints;
        
            data->P = csc_matrix(n_vars, n_vars, P.nonZeros(),
                                 reinterpret_cast<c_float*>(const_cast<double*>(P.valuePtr())),
                                 reinterpret_cast<c_int*>(const_cast<long long*>(P.innerIndexPtr())),
                                 reinterpret_cast<c_int*>(const_cast<long long*>(P.outerIndexPtr())));
            data->q = reinterpret_cast<c_float*>(const_cast<double*>(q.data()));
        
            data->A = csc_matrix(n_constraints, n_vars, Aeq.nonZeros(),
                                 reinterpret_cast<c_float*>(const_cast<double*>(Aeq.valuePtr())),
                                 reinterpret_cast<c_int*>(const_cast<long long*>(Aeq.innerIndexPtr())),
                                 reinterpret_cast<c_int*>(const_cast<long long*>(Aeq.outerIndexPtr())));
            data->l = reinterpret_cast<c_float*>(const_cast<double*>(z_min.data()));
            data->u = reinterpret_cast<c_float*>(const_cast<double*>(z_max.data()));
        
            OSQPWorkspace* work = nullptr;
            osqp_setup(&work, data, settings);
            osqp_solve(work);
        
            if (!work || !work->solution || !work->solution->x) {
                std::cerr << "[ERROR] OSQP solve failed — null solution!" << std::endl;
                osqp_cleanup(work);
                c_free(data->A);
                c_free(data->P);
                c_free(data);
                c_free(settings);
                return;
            }
        
            if (work->info->status_val != OSQP_SOLVED) {
                std::cerr << "[WARNING] OSQP did not solve optimally: "
                          << work->info->status << std::endl;
            }
        
            Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd>(work->solution->x, n_vars);
            u_current = z.segment(nx_total, n_u);
        
            osqp_cleanup(work);
            c_free(data->A);
            c_free(data->P);
            c_free(data);
            c_free(settings);
        }
        
        
        
        
    
        Eigen::VectorXd getControl() const { return u_current; }
    };
    
    
    
    int main() {
        Parameters params;
        int steps = params.simulation_time / params.dt;
    
        Eigen::MatrixXd A(12, 12);
        A << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, -params.g, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, params.g, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    
        Eigen::MatrixXd B(12, 4);
        B << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             1 / params.mass, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 1 / params.I_x, 0, 0,
             0, 0, 1 / params.I_y, 0,
             0, 0, 0, 1 / params.I_z;
    
        TinyMPC tinympc(params, A, B);
        OSQP_Solver osqp_solver(params, A, B);
    
        // State and reference logging
        std::vector<double> x_log, y_log, z_log;
        std::vector<double> x_ref_log, y_ref_log, z_ref_log;
        std::vector<double> x_osqp_log, y_osqp_log, z_osqp_log;
    
        // Timing and memory logging
        std::vector<double> time_tiny_log, time_osqp_log;
        std::vector<size_t> mem_tiny_log, mem_osqp_log;
    
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);
        Eigen::VectorXd x = x0;
        Eigen::VectorXd x_osqp = x0;
    
        for (int k = 0; k < steps; ++k) {
            double t = k * params.dt;
            Eigen::VectorXd ref = figureEightRef(t, 5.0);
    
            // --- TinyMPC Solver ---
            auto start_tiny = std::chrono::high_resolution_clock::now();
            tinympc.Solve(x, ref);

            auto end_tiny = std::chrono::high_resolution_clock::now();
            time_tiny_log.push_back(std::chrono::duration<double>(end_tiny - start_tiny).count() * 1000.0);
    
            Eigen::VectorXd u_tiny = tinympc.getControl();
            x = A * x + B * u_tiny;
    
            // --- OSQP Solver ---
            auto start_osqp = std::chrono::high_resolution_clock::now();
   
            osqp_solver.Solve(x_osqp, ref);
      
            auto end_osqp = std::chrono::high_resolution_clock::now();
            time_osqp_log.push_back(std::chrono::duration<double>(end_osqp - start_osqp).count() * 1000.0);
    
            Eigen::VectorXd u_osqp = osqp_solver.getControl();
            x_osqp = A * x_osqp + B * u_osqp;
    
            // Log positions
            x_log.push_back(x(0));
            y_log.push_back(x(1));
            z_log.push_back(x(2));
            x_osqp_log.push_back(x_osqp(0));
            y_osqp_log.push_back(x_osqp(1));
            z_osqp_log.push_back(x_osqp(2));
            x_ref_log.push_back(ref(0));
            y_ref_log.push_back(ref(1));
            z_ref_log.push_back(ref(2));
        }
    
        // --- Plot Computation Time ---
        plt::figure();
        plt::named_plot("TinyMPC", time_tiny_log, "-r");
        plt::named_plot("OSQP", time_osqp_log, "-b");
        plt::xlabel("Iteration");
        plt::ylabel("Time per iteration [ms]");
        plt::title("Control Computation Time per Iteration");
        plt::ylim(0, 10);  // Limit max Y-axis to 10 ms
        plt::legend();
        plt::show();
    
    
        // --- Plot Trajectories ---
        plt::figure();
        plt::named_plot("TinyMPC", x_log, y_log, "-r");
        plt::named_plot("OSQP", x_osqp_log, y_osqp_log, "-b");
        plt::named_plot("Reference", x_ref_log, y_ref_log, "--k");
        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::title("Trajectory Comparison: TinyMPC vs OSQP");
        plt::legend();
        plt::axis("equal");
        plt::show();
    
        return 0;
    }
    
