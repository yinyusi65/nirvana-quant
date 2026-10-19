#ifndef NIRVANA_MANIFOLD_HPP
#define NIRVANA_MANIFOLD_HPP

#include "types.hpp"
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>

namespace nirvana {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct DualClockCoordinates {
    double macro_sin_1{0.0};
    double macro_cos_1{0.0};
    double macro_sin_2{0.0};
    double macro_cos_2{0.0};
    double micro_sin_1{0.0};
    double micro_cos_1{0.0};
    double micro_sin_2{0.0};
    double micro_cos_2{0.0};
    double resonance_gamma{0.0};

    std::vector<double> to_vector() const {
        return {
            macro_sin_1, macro_cos_1, macro_sin_2, macro_cos_2,
            micro_sin_1, micro_cos_1, micro_sin_2, micro_cos_2,
            resonance_gamma
        };
    }
};

class Base12Manifold {
public:
    double volume_quantum_{1000.0};
    std::pair<double, double> macro_periods_{63.0, 252.0};

    explicit Base12Manifold(double volume_quantum = 1000.0,
                            std::pair<double, double> macro_periods = {63.0, 252.0})
        : volume_quantum_(std::max(volume_quantum, 1e-4)),
          macro_periods_(macro_periods) {}

    /**
     * @brief Transforms discrete calendar index & volume tick onto 2-Torus manifold (S1 x S1).
     */
    DualClockCoordinates transform(int calendar_index, double cumulative_volume = 0.0) const {
        double p1 = macro_periods_.first;
        double p2 = macro_periods_.second;

        double theta_m1 = (2.0 * M_PI * std::fmod(static_cast<double>(calendar_index), p1)) / p1;
        double m_sin_1 = std::sin(theta_m1);
        double m_cos_1 = std::cos(theta_m1);

        double theta_m2 = (2.0 * M_PI * std::fmod(static_cast<double>(calendar_index), p2)) / p2;
        double m_sin_2 = std::sin(theta_m2);
        double m_cos_2 = std::cos(theta_m2);

        // Volume / Information Time (Micro Base-12)
        long long volume_bars = static_cast<long long>(std::max(0.0, cumulative_volume / volume_quantum_));
        double theta_v1 = (2.0 * M_PI * (volume_bars % 12)) / 12.0;
        double v_sin_1 = std::sin(theta_v1);
        double v_cos_1 = std::cos(theta_v1);

        double theta_v2 = (2.0 * M_PI * (volume_bars % 144)) / 144.0;
        double v_sin_2 = std::sin(theta_v2);
        double v_cos_2 = std::cos(theta_v2);

        // Phase Resonance Index: gamma(t) = cos(theta_macro - theta_micro)
        double resonance_gamma = std::cos(theta_m1 - theta_v1);

        return DualClockCoordinates{
            m_sin_1, m_cos_1, m_sin_2, m_cos_2,
            v_sin_1, v_cos_1, v_sin_2, v_cos_2,
            resonance_gamma
        };
    }

    /**
     * @brief Batch transformation of time series vectors.
     * Returns Matrix of shape (N, 9).
     */
    Matrix batch_transform(const std::vector<int>& calendar_indices,
                           const std::vector<double>& cumulative_volumes = {}) const {
        size_t n = calendar_indices.size();
        Matrix res(n, 9);
        for (size_t i = 0; i < n; ++i) {
            double vol = (i < cumulative_volumes.size()) ? cumulative_volumes[i] : 0.0;
            DualClockCoordinates coords = transform(calendar_indices[i], vol);
            res(i, 0) = coords.macro_sin_1;
            res(i, 1) = coords.macro_cos_1;
            res(i, 2) = coords.macro_sin_2;
            res(i, 3) = coords.macro_cos_2;
            res(i, 4) = coords.micro_sin_1;
            res(i, 5) = coords.micro_cos_1;
            res(i, 6) = coords.micro_sin_2;
            res(i, 7) = coords.micro_cos_2;
            res(i, 8) = coords.resonance_gamma;
        }
        return res;
    }

    /**
     * @brief Estimates dominant cyclical periods from a 1D signal via Discrete Fourier Transform.
     */
    static std::pair<double, double> estimate_dominant_periods(const std::vector<double>& signal) {
        size_t n = signal.size();
        if (n < 16) return {63.0, 252.0};

        // Linear detrend
        std::vector<double> detrended(n);
        double start = signal.front();
        double end = signal.back();
        for (size_t i = 0; i < n; ++i) {
            double trend = start + (end - start) * (static_cast<double>(i) / (n - 1));
            detrended[i] = signal[i] - trend;
        }

        // Discrete Fourier Transform Power Spectrum
        size_t n_freqs = n / 2;
        std::vector<std::pair<double, double>> power_period; // (power, period)

        for (size_t k = 1; k < n_freqs; ++k) {
            double re = 0.0, im = 0.0;
            for (size_t t = 0; t < n; ++t) {
                double angle = 2.0 * M_PI * k * t / n;
                re += detrended[t] * std::cos(angle);
                im -= detrended[t] * std::sin(angle);
            }
            double power = re * re + im * im;
            double period = static_cast<double>(n) / k;
            if (period >= 3.0 && period <= static_cast<double>(n)) {
                power_period.push_back({power, period});
            }
        }

        if (power_period.empty()) return {63.0, 252.0};

        std::sort(power_period.begin(), power_period.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        double p1 = power_period[0].second;
        double p2 = (power_period.size() > 1) ? power_period[1].second : p1 * 4.0;
        if (p1 > p2) std::swap(p1, p2);

        return {std::round(p1 * 10.0) / 10.0, std::round(p2 * 10.0) / 10.0};
    }
};


/**
 * @brief Fractional Base-12 Continuous Torus Manifold (S1 x S1 x R).
 * Generalizes integer discrete cycles into fractional-order powers:
 *   P_alpha = 12^(alpha_quarter) and P_beta = 12^(alpha_annual)
 */
class FractionalBase12Manifold : public Base12Manifold {
public:
    double alpha_quarter_{1.6667}; // 12^1.6667 ~= 63.0 sessions
    double alpha_annual_{2.2255};  // 12^2.2255 ~= 252.0 sessions
    double fractional_shift_{0.0};

    explicit FractionalBase12Manifold(double alpha_quarter = 1.6667,
                                     double alpha_annual = 2.2255,
                                     double volume_quantum = 1000.0,
                                     double fractional_shift = 0.0)
        : Base12Manifold(volume_quantum, {std::pow(12.0, alpha_quarter), std::pow(12.0, alpha_annual)}),
          alpha_quarter_(alpha_quarter),
          alpha_annual_(alpha_annual),
          fractional_shift_(fractional_shift) {}

    /**
     * @brief Constructor accepting single alpha scale factor
     */
    static FractionalBase12Manifold from_alpha(double alpha, double volume_quantum = 1000.0, double fractional_shift = 0.0) {
        return FractionalBase12Manifold(alpha, alpha + 0.5588, volume_quantum, fractional_shift);
    }

    /**
     * @brief Transforms continuous real-valued time coordinates t in R into fractional 2-Torus.
     */
    DualClockCoordinates continuous_transform(double t_continuous, double cumulative_volume = 0.0) const {
        double t = t_continuous + fractional_shift_;
        double p1 = macro_periods_.first;
        double p2 = macro_periods_.second;

        double theta_m1 = (2.0 * M_PI * std::fmod(std::fmod(t, p1) + p1, p1)) / p1;
        double theta_m2 = (2.0 * M_PI * std::fmod(std::fmod(t, p2) + p2, p2)) / p2;

        double v_bars = std::max(0.0, cumulative_volume / volume_quantum_);
        double theta_v1 = (2.0 * M_PI * std::fmod(v_bars, 12.0)) / 12.0;
        double theta_v2 = (2.0 * M_PI * std::fmod(v_bars, 144.0)) / 144.0;

        double resonance_gamma = std::cos(theta_m1 - theta_v1);

        return DualClockCoordinates{
            std::sin(theta_m1), std::cos(theta_m1),
            std::sin(theta_m2), std::cos(theta_m2),
            std::sin(theta_v1), std::cos(theta_v1),
            std::sin(theta_v2), std::cos(theta_v2),
            resonance_gamma
        };
    }

    /**
     * @brief Batch transformation of continuous real-valued time coordinates.
     * Returns Matrix of shape (N, 9).
     */
    Matrix batch_continuous_transform(const std::vector<double>& t_continuous_vec,
                                      const std::vector<double>& cumulative_volumes = {}) const {
        size_t n = t_continuous_vec.size();
        Matrix res(n, 9);
        for (size_t i = 0; i < n; ++i) {
            double vol = (i < cumulative_volumes.size()) ? cumulative_volumes[i] : 0.0;
            DualClockCoordinates coords = continuous_transform(t_continuous_vec[i], vol);
            res(i, 0) = coords.macro_sin_1;
            res(i, 1) = coords.macro_cos_1;
            res(i, 2) = coords.macro_sin_2;
            res(i, 3) = coords.macro_cos_2;
            res(i, 4) = coords.micro_sin_1;
            res(i, 5) = coords.micro_cos_1;
            res(i, 6) = coords.micro_sin_2;
            res(i, 7) = coords.micro_cos_2;
            res(i, 8) = coords.resonance_gamma;
        }
        return res;
    }

    /**
     * @brief Auto-learns optimal fractional powers directly from empirical signal Fourier spectrum.
     */
    static FractionalBase12Manifold fit_optimal_alpha(const std::vector<double>& signal,
                                                     double volume_quantum = 1000.0) {
        auto [p1, p2] = Base12Manifold::estimate_dominant_periods(signal);
        if (p1 > p2) std::swap(p1, p2);

        double ln12 = std::log(12.0);
        double alpha_q = std::max(std::log(std::max(p1, 2.0)) / ln12, 0.5);
        double alpha_a = std::max(std::log(std::max(p2, p1 + 1.0)) / ln12, alpha_q + 0.1);

        return FractionalBase12Manifold(alpha_q, alpha_a, volume_quantum, 0.0);
    }
};

} // namespace nirvana

#endif // NIRVANA_MANIFOLD_HPP
