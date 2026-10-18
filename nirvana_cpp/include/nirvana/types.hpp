#ifndef NIRVANA_TYPES_HPP
#define NIRVANA_TYPES_HPP

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <functional>

namespace nirvana {

/**
 * @brief High-performance contiguous dense Matrix for low-latency quantitative finance.
 * Memory layout is row-major: data[r * cols + c].
 */
class Matrix {
public:
    size_t rows_{0};
    size_t cols_{0};
    std::vector<double> data_{};

    Matrix() = default;
    Matrix(size_t rows, size_t cols, double init_val = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    Matrix(size_t rows, size_t cols, const std::vector<double>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Matrix dimension mismatch with data size.");
        }
    }

    inline double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    inline const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    inline size_t rows() const { return rows_; }
    inline size_t cols() const { return cols_; }
    inline size_t size() const { return data_.size(); }
    inline double* data() { return data_.data(); }
    inline const double* data() const { return data_.data(); }

    static Matrix zeros(size_t r, size_t c) {
        return Matrix(r, c, 0.0);
    }

    static Matrix ones(size_t r, size_t c) {
        return Matrix(r, c, 1.0);
    }

    static Matrix eye(size_t n, double diag_val = 1.0) {
        Matrix m(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            m(i, i) = diag_val;
        }
        return m;
    }

    static Matrix diag(const std::vector<double>& vec) {
        size_t n = vec.size();
        Matrix m(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            m(i, i) = vec[i];
        }
        return m;
    }

    Matrix transpose() const {
        Matrix res(cols_, rows_);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                res(c, r) = (*this)(r, c);
            }
        }
        return res;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix addition dimension mismatch.");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = data_[i] + other.data_[i];
        }
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix subtraction dimension mismatch.");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = data_[i] - other.data_[i];
        }
        return res;
    }

    Matrix operator*(double scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = data_[i] * scalar;
        }
        return res;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix multiplication inner dimension mismatch.");
        }
        Matrix res(rows_, other.cols_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                double r_ik = (*this)(i, k);
                if (std::abs(r_ik) < 1e-15) continue;
                for (size_t j = 0; j < other.cols_; ++j) {
                    res(i, j) += r_ik * other(k, j);
                }
            }
        }
        return res;
    }

    std::vector<double> operator*(const std::vector<double>& vec) const {
        if (cols_ != vec.size()) {
            throw std::invalid_argument("Matrix-vector multiplication dimension mismatch.");
        }
        std::vector<double> res(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * vec[j];
            }
            res[i] = sum;
        }
        return res;
    }

    std::vector<double> row_sums() const {
        std::vector<double> sums(rows_, 0.0);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                sums[r] += (*this)(r, c);
            }
        }
        return sums;
    }

    std::vector<double> col_sums() const {
        std::vector<double> sums(cols_, 0.0);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                sums[c] += (*this)(r, c);
            }
        }
        return sums;
    }

    std::vector<double> col_means() const {
        std::vector<double> means(cols_, 0.0);
        if (rows_ == 0) return means;
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                means[c] += (*this)(r, c);
            }
        }
        for (size_t c = 0; c < cols_; ++c) {
            means[c] /= static_cast<double>(rows_);
        }
        return means;
    }

    std::vector<double> col(size_t c) const {
        std::vector<double> v(rows_);
        for (size_t r = 0; r < rows_; ++r) {
            v[r] = (*this)(r, c);
        }
        return v;
    }

    std::vector<double> row(size_t r) const {
        std::vector<double> v(cols_);
        for (size_t c = 0; c < cols_; ++c) {
            v[c] = (*this)(r, c);
        }
        return v;
    }

    /**
     * @brief Inversion of square matrix via Gauss-Jordan elimination with partial pivoting.
     */
    Matrix inverse(double ridge_epsilon = 1e-8) const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square to invert.");
        }
        size_t n = rows_;
        Matrix A = *this;
        // Apply slight ridge regularization for ill-conditioned covariance matrices
        if (ridge_epsilon > 0.0) {
            for (size_t i = 0; i < n; ++i) {
                A(i, i) += ridge_epsilon;
            }
        }
        Matrix inv = Matrix::eye(n);

        for (size_t i = 0; i < n; ++i) {
            // Find pivot
            size_t max_row = i;
            double max_val = std::abs(A(i, i));
            for (size_t k = i + 1; k < n; ++k) {
                double val = std::abs(A(k, i));
                if (val > max_val) {
                    max_val = val;
                    max_row = k;
                }
            }
            if (max_val < 1e-15) {
                // Singular matrix: fallback to diagonal pseudo-inverse
                Matrix pseudo = Matrix::zeros(n, n);
                for (size_t j = 0; j < n; ++j) {
                    double d = (*this)(j, j);
                    pseudo(j, j) = (std::abs(d) > 1e-12) ? (1.0 / d) : 0.0;
                }
                return pseudo;
            }

            // Swap rows
            if (max_row != i) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(max_row, j));
                    std::swap(inv(i, j), inv(max_row, j));
                }
            }

            double pivot = A(i, i);
            for (size_t j = 0; j < n; ++j) {
                A(i, j) /= pivot;
                inv(i, j) /= pivot;
            }

            for (size_t k = 0; k < n; ++k) {
                if (k == i) continue;
                double factor = A(k, i);
                for (size_t j = 0; j < n; ++j) {
                    A(k, j) -= factor * A(i, j);
                    inv(k, j) -= factor * inv(i, j);
                }
            }
        }
        return inv;
    }

    /**
     * @brief Ridge Regression pseudo-inverse: (X^T * X + lambda * I)^(-1) * X^T
     */
    static Matrix ridge_pinv(const Matrix& X, double lambda = 0.5) {
        Matrix Xt = X.transpose();
        Matrix XtX = Xt * X;
        for (size_t i = 0; i < XtX.rows(); ++i) {
            XtX(i, i) += lambda;
        }
        Matrix inv = XtX.inverse(1e-8);
        return inv * Xt;
    }

    /**
     * @brief Truncated SVD via Deflated Power Iteration.
     * Computes top k principal components: U, S (singular values), V.
     */
    void svd_truncated(size_t k, Matrix& U, std::vector<double>& S, Matrix& V, size_t max_iter = 100) const {
        size_t m = rows_;
        size_t n = cols_;
        k = std::min(k, std::min(m, n));

        U = Matrix::zeros(m, k);
        V = Matrix::zeros(n, k);
        S.assign(k, 0.0);

        Matrix A_deflated = *this;

        for (size_t comp = 0; comp < k; ++comp) {
            // Random non-zero initial guess
            std::vector<double> v(n);
            for (size_t j = 0; j < n; ++j) {
                v[j] = std::sin(static_cast<double>(comp + 1) * (j + 1));
            }
            double v_norm = 0.0;
            for (double x : v) v_norm += x * x;
            v_norm = std::sqrt(v_norm);
            if (v_norm < 1e-12) v[0] = 1.0, v_norm = 1.0;
            for (double& x : v) x /= v_norm;

            std::vector<double> u(m, 0.0);
            double s_val = 0.0;

            for (size_t iter = 0; iter < max_iter; ++iter) {
                // u = A * v
                u = A_deflated * v;
                double u_norm = 0.0;
                for (double x : u) u_norm += x * x;
                u_norm = std::sqrt(u_norm);
                if (u_norm < 1e-12) break;
                for (double& x : u) x /= u_norm;

                // v = A^T * u
                Matrix At = A_deflated.transpose();
                v = At * u;
                v_norm = 0.0;
                for (double x : v) v_norm += x * x;
                v_norm = std::sqrt(v_norm);
                if (v_norm < 1e-12) break;
                for (double& x : v) x /= v_norm;

                s_val = v_norm;
            }

            S[comp] = s_val;
            for (size_t i = 0; i < m; ++i) U(i, comp) = u[i];
            for (size_t j = 0; j < n; ++j) V(j, comp) = v[j];

            // Deflation: A_{new} = A - s_comp * (u * v^T)
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    A_deflated(i, j) -= s_val * u[i] * v[j];
                }
            }
        }
    }

    void print(const std::string& label = "", size_t max_r = 10, size_t max_c = 10) const {
        if (!label.empty()) {
            std::cout << label << " (" << rows_ << "x" << cols_ << "):\n";
        }
        for (size_t r = 0; r < std::min(rows_, max_r); ++r) {
            std::cout << "  [ ";
            for (size_t c = 0; c < std::min(cols_, max_c); ++c) {
                std::cout << std::fixed << std::setprecision(3) << (*this)(r, c) << " ";
            }
            if (cols_ > max_c) std::cout << "... ";
            std::cout << "]\n";
        }
        if (rows_ > max_r) std::cout << "  ...\n";
    }
};

} // namespace nirvana

#endif // NIRVANA_TYPES_HPP
