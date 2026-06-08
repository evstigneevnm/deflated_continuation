#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/external_libraries/lapack_wrap.h>

namespace
{

int checks = 0;
int failures = 0;

template<class T>
std::string type_name()
{
    return std::is_same<T, float>::value ? "float" : "double";
}

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(2e-4) : T(2e-10);
}

std::size_t idx(std::size_t row, std::size_t col, std::size_t rows)
{
    return row + rows * col;
}

void record_failure(const std::string& message)
{
    ++failures;
    std::cerr << "FAIL " << message << std::endl;
}

template<class T>
void check_close(T value, T expected, T tol, const std::string& label)
{
    ++checks;
    const T err = std::abs(value - expected);
    if(!(err <= tol))
    {
        record_failure(label + " value=" + std::to_string(static_cast<double>(value)) +
                       " expected=" + std::to_string(static_cast<double>(expected)) +
                       " err=" + std::to_string(static_cast<double>(err)) +
                       " tol=" + std::to_string(static_cast<double>(tol)));
    }
}

template<class T>
void check_complex_close(
    const std::complex<T>& value,
    const std::complex<T>& expected,
    T tol,
    const std::string& label
)
{
    check_close<T>(value.real(), expected.real(), tol, label + " real");
    check_close<T>(value.imag(), expected.imag(), tol, label + " imag");
}

template<class T>
std::vector<T> matmul(
    const std::vector<T>& A,
    const std::vector<T>& B,
    std::size_t n,
    bool transpose_A = false,
    bool transpose_B = false
)
{
    std::vector<T> C(n * n, T(0));
    for(std::size_t col = 0; col < n; ++col)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            T sum = T(0);
            for(std::size_t k = 0; k < n; ++k)
            {
                const T a = transpose_A ? A[idx(k, row, n)] : A[idx(row, k, n)];
                const T b = transpose_B ? B[idx(col, k, n)] : B[idx(k, col, n)];
                sum += a * b;
            }
            C[idx(row, col, n)] = sum;
        }
    }
    return C;
}

template<class T>
std::vector<std::complex<T>> matmul_complex(
    const std::vector<std::complex<T>>& A,
    const std::vector<std::complex<T>>& B,
    std::size_t n
)
{
    std::vector<std::complex<T>> C(n * n, std::complex<T>(0));
    for(std::size_t col = 0; col < n; ++col)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            std::complex<T> sum(0);
            for(std::size_t k = 0; k < n; ++k)
            {
                sum += A[idx(row, k, n)] * B[idx(k, col, n)];
            }
            C[idx(row, col, n)] = sum;
        }
    }
    return C;
}

template<class T>
std::vector<T> transpose(const std::vector<T>& A, std::size_t n)
{
    std::vector<T> AT(n * n, T(0));
    for(std::size_t col = 0; col < n; ++col)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            AT[idx(row, col, n)] = A[idx(col, row, n)];
        }
    }
    return AT;
}

template<class T>
void check_matrix_close(
    const std::vector<T>& value,
    const std::vector<T>& expected,
    T tol,
    const std::string& label
)
{
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        check_close<T>(value[i], expected[i], tol * (T(1) + std::abs(expected[i])), label + " i=" + std::to_string(i));
    }
}

template<class T>
void check_complex_matrix_close(
    const std::vector<std::complex<T>>& value,
    const std::vector<std::complex<T>>& expected,
    T tol,
    const std::string& label
)
{
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        check_complex_close<T>(
            value[i],
            expected[i],
            tol * (T(1) + std::abs(expected[i])),
            label + " i=" + std::to_string(i)
        );
    }
}

template<class T>
std::vector<T> test_matrix_3()
{
    return {
        T(2.0), T(1.0), T(0.5),
        T(-1.0), T(3.0), T(1.5),
        T(0.25), T(-2.0), T(4.0)
    };
}

template<class T>
std::vector<T> triangular_eig_matrix_3()
{
    return {
        T(1.0), T(0.0), T(0.0),
        T(0.5), T(2.0), T(0.0),
        T(-0.25), T(1.0), T(3.0)
    };
}

template<class T>
bool has_eigenvalue(const std::vector<std::complex<T>>& eigs, std::complex<T> expected, T tol)
{
    for(const auto& eig : eigs)
    {
        if(std::abs(eig - expected) <= tol)
        {
            return true;
        }
    }
    return false;
}

template<class T>
void test_helpers(scfd::lapack_wrap<T>& lapack)
{
    const std::size_t n = 3;
    std::vector<T> I(n * n, T(-7));
    lapack.eye(I.data(), n);
    for(std::size_t col = 0; col < n; ++col)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            check_close<T>(I[idx(row, col, n)], row == col ? T(1) : T(0), tolerance<T>(), type_name<T>() + " eye");
        }
    }

    auto A = test_matrix_3<T>();
    lapack.add_to_diagonal(T(2), n, A.data());
    check_close<T>(A[idx(0, 0, n)], T(4), tolerance<T>(), type_name<T>() + " add_to_diagonal 0");
    check_close<T>(A[idx(1, 1, n)], T(5), tolerance<T>(), type_name<T>() + " add_to_diagonal 1");
    check_close<T>(A[idx(2, 2, n)], T(6), tolerance<T>(), type_name<T>() + " add_to_diagonal 2");

    std::vector<T> row(n, T(0));
    lapack.return_row(1, A.data(), n, row.data());
    check_close<T>(row[0], A[idx(1, 0, n)], tolerance<T>(), type_name<T>() + " return_row 0");
    check_close<T>(row[1], A[idx(1, 1, n)], tolerance<T>(), type_name<T>() + " return_row 1");
    check_close<T>(row[2], A[idx(1, 2, n)], tolerance<T>(), type_name<T>() + " return_row 2");

    std::vector<T> col(n, T(0));
    lapack.return_col(2, A.data(), n, col.data());
    check_close<T>(col[0], A[idx(0, 2, n)], tolerance<T>(), type_name<T>() + " return_col 0");
    check_close<T>(col[1], A[idx(1, 2, n)], tolerance<T>(), type_name<T>() + " return_col 1");
    check_close<T>(col[2], A[idx(2, 2, n)], tolerance<T>(), type_name<T>() + " return_col 2");

    std::vector<T> sub(4, T(0));
    lapack.return_submatrix(A.data(), {n, n}, {1, 1}, sub.data(), {2, 2});
    check_close<T>(sub[idx(0, 0, 2)], A[idx(1, 1, n)], tolerance<T>(), type_name<T>() + " return_submatrix 00");
    check_close<T>(sub[idx(1, 0, 2)], A[idx(2, 1, n)], tolerance<T>(), type_name<T>() + " return_submatrix 10");
    check_close<T>(sub[idx(0, 1, 2)], A[idx(1, 2, n)], tolerance<T>(), type_name<T>() + " return_submatrix 01");
    check_close<T>(sub[idx(1, 1, 2)], A[idx(2, 2, n)], tolerance<T>(), type_name<T>() + " return_submatrix 11");

    std::vector<T> B(n * n, T(0));
    lapack.set_submatrix({1, 1}, sub.data(), {2, 2}, B.data(), {n, n});
    check_close<T>(B[idx(1, 1, n)], sub[idx(0, 0, 2)], tolerance<T>(), type_name<T>() + " set_submatrix 00");
    check_close<T>(B[idx(2, 2, n)], sub[idx(1, 1, 2)], tolerance<T>(), type_name<T>() + " set_submatrix 11");

    std::vector<std::complex<T>> C(n * n);
    std::vector<T> D(n * n, T(0));
    lapack.double2complex(A.data(), n, n, C.data());
    lapack.complex2double(C.data(), n, n, D.data());
    check_matrix_close<T>(D, A, tolerance<T>(), type_name<T>() + " double-complex conversion");
}

template<class T>
void test_gemm(scfd::lapack_wrap<T>& lapack)
{
    const std::size_t n = 3;
    const auto A = test_matrix_3<T>();
    const auto B = triangular_eig_matrix_3<T>();
    std::vector<T> C(n * n, T(0));
    lapack.gemm(A.data(), 'N', B.data(), 'N', n, C.data());
    check_matrix_close<T>(C, matmul(A, B, n), tolerance<T>() * T(20), type_name<T>() + " real gemm");

    std::vector<T> C2(n * n, T(0));
    lapack.mat_sq(A.data(), n, C2.data());
    check_matrix_close<T>(C2, matmul(A, A, n), tolerance<T>() * T(20), type_name<T>() + " mat_sq");

    std::vector<std::complex<T>> AC(n * n);
    std::vector<std::complex<T>> BC(n * n);
    for(std::size_t i = 0; i < n * n; ++i)
    {
        AC[i] = std::complex<T>(A[i], T(0.25) * A[i]);
        BC[i] = std::complex<T>(B[i], T(-0.125) * B[i]);
    }
    std::vector<std::complex<T>> CC(n * n, std::complex<T>(0));
    lapack.gemm(AC.data(), 'N', BC.data(), 'N', n, CC.data());
    check_complex_matrix_close<T>(
        CC,
        matmul_complex(AC, BC, n),
        tolerance<T>() * T(50),
        type_name<T>() + " complex gemm"
    );
}

template<class T>
void test_qr(scfd::lapack_wrap<T>& lapack)
{
    const std::size_t n = 3;
    const auto A = test_matrix_3<T>();
    std::vector<T> Q(n * n, T(0));
    std::vector<T> R(n * n, T(0));
    lapack.qr(A.data(), n, Q.data(), R.data());

    check_matrix_close<T>(matmul(Q, R, n), A, tolerance<T>() * T(100), type_name<T>() + " QR reconstruction");
    std::vector<T> I(n * n, T(0));
    lapack.eye(I.data(), n);
    check_matrix_close<T>(matmul(Q, Q, n, true, false), I, tolerance<T>() * T(100), type_name<T>() + " QR orthogonality");

    std::vector<T> Q_only(n * n, T(0));
    lapack.qr(A.data(), n, Q_only.data());
    check_matrix_close<T>(Q_only, Q, tolerance<T>() * T(100), type_name<T>() + " QR no-R Q match");
}

template<class T>
void test_eigs(scfd::lapack_wrap<T>& lapack)
{
    const std::size_t n = 3;
    const auto A = triangular_eig_matrix_3<T>();
    std::vector<std::complex<T>> eigs(n);
    lapack.eigs(A.data(), n, eigs.data());
    const T tol = tolerance<T>() * T(100);
    for(std::size_t i = 0; i < n; ++i)
    {
        ++checks;
        if(!has_eigenvalue<T>(eigs, std::complex<T>(T(i + 1), T(0)), tol))
        {
            record_failure(type_name<T>() + " eigs missing eigenvalue " + std::to_string(i + 1));
        }
    }

    std::vector<std::complex<T>> eigs_from_h(n);
    lapack.hessinberg_eigs(A.data(), n, eigs_from_h.data());
    for(std::size_t i = 0; i < n; ++i)
    {
        ++checks;
        if(!has_eigenvalue<T>(eigs_from_h, std::complex<T>(T(i + 1), T(0)), tol))
        {
            record_failure(type_name<T>() + " hessinberg_eigs missing eigenvalue " + std::to_string(i + 1));
        }
    }

    std::vector<std::complex<T>> eigv(n);
    std::vector<std::complex<T>> V(n * n);
    lapack.eigsv(A.data(), n, eigv.data(), V.data());
    for(std::size_t k = 0; k < n; ++k)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            std::complex<T> av(0);
            for(std::size_t col = 0; col < n; ++col)
            {
                av += std::complex<T>(A[idx(row, col, n)], T(0)) * V[idx(col, k, n)];
            }
            const auto lv = eigv[k] * V[idx(row, k, n)];
            check_complex_close<T>(
                av,
                lv,
                tolerance<T>() * T(200) * (T(1) + std::abs(lv)),
                type_name<T>() + " eigsv residual k=" + std::to_string(k) + " row=" + std::to_string(row)
            );
        }
    }
}

template<class T>
void test_schur(scfd::lapack_wrap<T>& lapack)
{
    const std::size_t n = 3;
    const auto A = triangular_eig_matrix_3<T>();
    std::vector<T> Q(n * n, T(0));
    std::vector<T> R(n * n, T(0));
    std::vector<std::complex<T>> eigs(n);
    lapack.eigs_schur(A.data(), n, eigs.data(), Q.data(), R.data());
    const auto QR = matmul(Q, R, n);
    const auto QRQT = matmul(QR, Q, n, false, true);
    check_matrix_close<T>(QRQT, A, tolerance<T>() * T(200), type_name<T>() + " eigs_schur reconstruction");

    std::vector<T> Qh(n * n, T(0));
    std::vector<T> Rh(n * n, T(0));
    std::vector<std::complex<T>> eigs_h(n);
    lapack.hessinberg_schur(A.data(), n, Qh.data(), Rh.data(), eigs_h.data());
    const auto QhRh = matmul(Qh, Rh, n);
    const auto QhRhQht = matmul(QhRh, Qh, n, false, true);
    check_matrix_close<T>(QhRhQht, A, tolerance<T>() * T(200), type_name<T>() + " hessinberg_schur reconstruction");
}

template<class T>
void run_common_type_tests()
{
    std::cout << "Testing LAPACK wrapper " << type_name<T>() << std::endl;
    scfd::lapack_wrap<T> lapack(3);
    test_helpers<T>(lapack);
    test_gemm<T>(lapack);
    test_qr<T>(lapack);
    test_schur<T>(lapack);
}

void run_double_only_tests()
{
    scfd::lapack_wrap<double> lapack(3);
    test_eigs<double>(lapack);
}

} // namespace

int main()
{
    try
    {
        run_common_type_tests<double>();
        run_common_type_tests<float>();
        run_double_only_tests();
    }
    catch(const std::exception& exc)
    {
        record_failure(std::string("unexpected exception: ") + exc.what());
    }

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
