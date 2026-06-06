/**
 * @file tests.h
 * @brief Статистические тесты для проверки качества ПСП.
 *
 * Реализует:
 * - Критерий Хи-квадрат на равномерность распределения;
 * - Пять адаптированных тестов из набора NIST SP 800-22 / Diehard:
 *   1) Monobit Test (частотный побитовый);
 *   2) Frequency Test within a Block;
 *   3) Runs Test (последовательности одинаковых битов);
 *   4) Longest Run of Ones in a Block;
 *   5) Serial Test (последовательные пары/тройки битов).
 *
 * @author Лабораторная работа по ПСП
 * @date 2026
 */

#ifndef TESTS_H
#define TESTS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace prng {

/** @brief Уровень значимости для статистических тестов. */
constexpr double kAlpha = 0.05;

/** @brief Минимальный порог p-value для принятия гипотезы. */
constexpr double kPValueThreshold = kAlpha;

// ---------------------------------------------------------------------------
// Вспомогательная математика
// ---------------------------------------------------------------------------

/**
 * @brief Дополнительная функция ошибок erfc(x) (аппроксимация Абрамовица-Стегана).
 * @param x Аргумент.
 * @return Значение erfc(x).
 */
inline double erfcApprox(double x) {
    const double ax = std::abs(x);
    const double t = 1.0 / (1.0 + 0.5 * ax);
    const double tau = t * std::exp(-ax * ax - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
        t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
        t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? tau : 2.0 - tau;
}

/**
 * @brief Нижняя неполная гамма-функция P(a, x) = gamma(a, x) / Gamma(a).
 *
 * Реализация по формулам Numerical Recipes: ряд при x < a+1,
 * непрерывная дробь Лежандра в противном случае.
 * @param a Параметр формы (df/2).
 * @param x Аргумент (chi2/2).
 * @return Вероятность P(a, x) в [0, 1].
 */
inline double regularizedGammaP(double a, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (a <= 0.0) {
        return 1.0;
    }

    const double ln_gamma_a = std::lgamma(a);

    if (x < a + 1.0) {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 1; n <= 200; ++n) {
            ++ap;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-12) {
                break;
            }
        }
        return sum * std::exp(-x + a * std::log(x) - ln_gamma_a);
    }

    double b = x + 1.0 - a;
    double c = 1.0 / 1e-30;
    double d = 1.0 / b;
    double h = d;
    for (int n = 1; n <= 200; ++n) {
        const double an = -static_cast<double>(n) * (static_cast<double>(n) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < 1e-30) {
            d = 1e-30;
        }
        c = b + an / c;
        if (std::abs(c) < 1e-30) {
            c = 1e-30;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-12) {
            break;
        }
    }
    const double result = std::exp(-x + a * std::log(x) - ln_gamma_a) * h;
    return std::min(1.0, std::max(0.0, 1.0 - result));
}

/**
 * @brief Вычисляет p-value для статистики Хи-квадрат.
 * @param chi2 Наблюдаемое значение статистики.
 * @param df Число степеней свободы.
 * @return p-value (вероятность получить значение >= chi2 при H0).
 */
inline double chi2PValue(double chi2, int df) {
    if (df <= 0) {
        return 1.0;
    }
    const double p = 1.0 - regularizedGammaP(static_cast<double>(df) / 2.0, chi2 / 2.0);
    return std::min(1.0, std::max(0.0, p));
}

/**
 * @brief Критическое значение Хи-квадрат для заданных df и alpha (табличные значения).
 *
 * Для df, отсутствующих в таблице, используется аппроксимация Вилсона-Хилферти.
 * @param df Число степеней свободы.
 * @param alpha Уровень значимости.
 * @return Критическое значение chi2_{1-alpha, df}.
 */
inline double chi2Critical(int df, double alpha = kAlpha) {
    static constexpr std::array<double, 101> kTable05 = {
        0.0,    3.841,  5.991,  7.815,  9.488,  11.070, 12.592, 14.067, 15.507, 16.919,
        18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296, 27.587, 28.869, 30.144,
        31.410, 32.671, 33.924, 35.172, 36.415, 37.652, 38.885, 40.113, 41.337, 42.557,
        43.773, 44.985, 46.194, 47.400, 48.602, 49.802, 50.998, 52.192, 53.384, 54.572,
        55.758, 56.942, 58.124, 59.304, 60.481, 61.656, 62.830, 64.001, 65.171, 66.339,
        67.505, 68.669, 69.832, 70.993, 72.153, 73.311, 74.468, 75.624, 76.778, 77.931,
        79.082, 80.232, 81.381, 82.529, 83.675, 84.821, 85.965, 87.108, 88.250, 89.391,
        90.531, 91.670, 92.808, 93.945, 95.081, 96.217, 97.351, 98.484, 99.616, 100.75,
        101.88, 103.01, 104.14, 105.27, 106.40, 107.52, 108.65, 109.77, 110.89, 112.02,
        113.14, 114.26, 115.38, 116.50, 117.62, 118.74, 119.85, 120.97, 122.08, 123.20,
        124.31};

    (void)alpha;

    if (df >= 1 && df < static_cast<int>(kTable05.size())) {
        return kTable05[static_cast<std::size_t>(df)];
    }

    // Аппроксимация Вилсона-Хилферти для больших df.
    const double z05 = 1.6448536269514722;  // z_{0.05} для alpha=0.05 (односторонний)
    const double df_d = static_cast<double>(df);
    const double term = 1.0 - 2.0 / (9.0 * df_d) + z05 * std::sqrt(2.0 / (9.0 * df_d));
    return df_d * term * term * term;
}

// ---------------------------------------------------------------------------
// Результаты тестов
// ---------------------------------------------------------------------------

/**
 * @brief Результат одного статистического теста.
 */
struct TestResult {
    std::string test_name;   ///< Название теста.
    double statistic = 0.0;  ///< Наблюдаемая статистика.
    double p_value = 1.0;    ///< Вычисленный p-value.
    bool passed = true;      ///< true, если гипотеза принимается (p >= alpha).

    /**
     * @brief Возвращает текстовый вердикт.
     * @return "ПРИНИМАЕТСЯ" или "ОТКЛОНЯЕТСЯ".
     */
    std::string verdict() const { return passed ? "ПРИНИМАЕТСЯ" : "ОТКЛОНЯЕТСЯ"; }
};

/**
 * @brief Результат критерия Хи-квадрат на равномерность.
 */
struct Chi2Result {
    double chi2_statistic = 0.0;  ///< Значение статистики chi^2.
    int degrees_of_freedom = 0;    ///< Число степеней свободы (K-1).
    int num_bins = 0;              ///< Число карманов K.
    double p_value = 1.0;        ///< p-value.
    double critical_value = 0.0;   ///< Критическое значение при alpha.
    bool passed = true;            ///< true, если H0 принимается.

    std::string verdict() const { return passed ? "ПРИНИМАЕТСЯ" : "ОТКЛОНЯЕТСЯ"; }
};

// ---------------------------------------------------------------------------
// Описательная статистика
// ---------------------------------------------------------------------------

/**
 * @brief Описательные статистики выборки.
 */
struct SampleStats {
    double mean = 0.0;              ///< Математическое ожидание (выборочное среднее).
    double variance = 0.0;          ///< Выборочная дисперсия (несмещённая, n-1).
    double std_dev = 0.0;           ///< Среднеквадратичное отклонение.
    double coeff_of_variation = 0.0; ///< Коэффициент вариации, %.
};

/**
 * @brief Вычисляет описательные статистики для выборки вещественных значений.
 * @param data Вектор значений.
 * @return Структура SampleStats.
 */
inline SampleStats computeStats(const std::vector<double>& data) {
    SampleStats stats;
    const std::size_t n = data.size();
    if (n == 0) {
        return stats;
    }

    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);

    if (n > 1) {
        double sum_sq = 0.0;
        for (double v : data) {
            const double d = v - stats.mean;
            sum_sq += d * d;
        }
        stats.variance = sum_sq / static_cast<double>(n - 1);
    }

    stats.std_dev = std::sqrt(stats.variance);
    stats.coeff_of_variation = (stats.mean != 0.0) ? (stats.std_dev / stats.mean) * 100.0 : 0.0;
    return stats;
}

// ---------------------------------------------------------------------------
// Критерий Хи-квадрат
// ---------------------------------------------------------------------------

/**
 * @brief Проверяет выборку целых чисел на равномерность по критерию Хи-квадрат.
 *
 * Диапазон [min_val, max_val] разбивается на num_bins равных карманов.
 * Сравниваются наблюдаемые и ожидаемые частоты.
 *
 * @param data Выборка целых значений.
 * @param min_val Минимум диапазона.
 * @param max_val Максимум диапазона.
 * @param num_bins Число карманов K.
 * @return Результат Chi2Result с вердиктом.
 */
inline Chi2Result chiSquareUniformityTest(const std::vector<std::uint32_t>& data,
                                          std::uint32_t min_val,
                                          std::uint32_t max_val,
                                          int num_bins = 10) {
    Chi2Result result;
    result.num_bins = num_bins;
    result.degrees_of_freedom = num_bins - 1;
    result.critical_value = chi2Critical(result.degrees_of_freedom, kAlpha);

    if (data.empty() || num_bins <= 0) {
        return result;
    }

    const double range = static_cast<double>(max_val) - min_val + 1.0;
    const double bin_width = range / static_cast<double>(num_bins);
    const double expected = static_cast<double>(data.size()) / static_cast<double>(num_bins);

    std::vector<int> observed(static_cast<std::size_t>(num_bins), 0);

    for (std::uint32_t v : data) {
        int bin = static_cast<int>((static_cast<double>(v) - min_val) / bin_width);
        if (bin >= num_bins) {
            bin = num_bins - 1;
        }
        if (bin < 0) {
            bin = 0;
        }
        ++observed[static_cast<std::size_t>(bin)];
    }

    double chi2 = 0.0;
    for (int count : observed) {
        const double diff = static_cast<double>(count) - expected;
        chi2 += (diff * diff) / expected;
    }

    result.chi2_statistic = chi2;
    result.p_value = chi2PValue(chi2, result.degrees_of_freedom);
    // Вердикт: chi2 <= критическое значение (alpha=0.05) И p-value >= alpha.
    result.passed = (chi2 <= result.critical_value) && (result.p_value >= kPValueThreshold);
    return result;
}

// ---------------------------------------------------------------------------
// Преобразование целых чисел в битовую последовательность
// ---------------------------------------------------------------------------

/**
 * @brief Разворачивает вектор 32-битных чисел в последовательность битов (MSB first).
 * @param values Входные числа.
 * @param bits_per_value Число младших битов каждого числа (1..32).
 * @return Вектор битов {0, 1}.
 */
inline std::vector<int> expandToBitStream(const std::vector<std::uint32_t>& values,
                                          int bits_per_value = 32) {
    std::vector<int> bits;
    bits.reserve(values.size() * static_cast<std::size_t>(bits_per_value));

    for (std::uint32_t v : values) {
        for (int b = bits_per_value - 1; b >= 0; --b) {
            bits.push_back((v >> b) & 1);
        }
    }
    return bits;
}

// ---------------------------------------------------------------------------
// NIST SP 800-22 адаптированные тесты
// ---------------------------------------------------------------------------

/**
 * @brief 1) Monobit (Frequency) Test — проверка баланса нулей и единиц.
 * @param bits Битовая последовательность.
 * @return TestResult.
 */
inline TestResult monobitTest(const std::vector<int>& bits) {
    TestResult r;
    r.test_name = "Monobit Test";

    const int n = static_cast<int>(bits.size());
    if (n == 0) {
        r.passed = false;
        r.p_value = 0.0;
        return r;
    }

    int sum = 0;
    for (int b : bits) {
        sum += (b == 1) ? 1 : -1;
    }

    const double s_obs = std::abs(static_cast<double>(sum)) / std::sqrt(static_cast<double>(n));
    r.statistic = s_obs;
    r.p_value = erfcApprox(s_obs / std::sqrt(2.0));
    r.passed = r.p_value >= kPValueThreshold;
    return r;
}

/**
 * @brief 2) Frequency Test within a Block — частота единиц внутри блоков.
 * @param bits Битовая последовательность.
 * @param block_size Размер блока M (по умолчанию 128).
 * @return TestResult.
 */
inline TestResult frequencyWithinBlockTest(const std::vector<int>& bits, int block_size = 128) {
    TestResult r;
    r.test_name = "Frequency within Block";

    const int n = static_cast<int>(bits.size());
    if (n < block_size || block_size <= 0) {
        r.passed = false;
        r.p_value = 0.0;
        return r;
    }

    const int num_blocks = n / block_size;
    double chi2 = 0.0;
    const double pi_expected = 0.5;

    for (int i = 0; i < num_blocks; ++i) {
        int ones = 0;
        for (int j = 0; j < block_size; ++j) {
            ones += bits[static_cast<std::size_t>(i * block_size + j)];
        }
        const double pi = static_cast<double>(ones) / static_cast<double>(block_size);
        chi2 += (pi - pi_expected) * (pi - pi_expected);
    }

    chi2 *= 4.0 * static_cast<double>(block_size);
    r.statistic = chi2;
    r.p_value = chi2PValue(chi2, num_blocks);
    r.passed = r.p_value >= kPValueThreshold;
    return r;
}

/**
 * @brief 3) Runs Test — тест на последовательности одинаковых битов.
 * @param bits Битовая последовательность.
 * @return TestResult.
 */
inline TestResult runsTest(const std::vector<int>& bits) {
    TestResult r;
    r.test_name = "Runs Test";

    const int n = static_cast<int>(bits.size());
    if (n < 2) {
        r.passed = false;
        r.p_value = 0.0;
        return r;
    }

    int ones = 0;
    for (int b : bits) {
        ones += b;
    }
    const double pi = static_cast<double>(ones) / static_cast<double>(n);

    if (std::abs(pi - 0.5) >= 2.0 / std::sqrt(static_cast<double>(n))) {
        r.p_value = 0.0;
        r.passed = false;
        return r;
    }

    int runs = 1;
    for (int i = 1; i < n; ++i) {
        if (bits[static_cast<std::size_t>(i)] != bits[static_cast<std::size_t>(i - 1)]) {
            ++runs;
        }
    }

    const double num = std::abs(static_cast<double>(runs) - 2.0 * static_cast<double>(n) * pi * (1.0 - pi));
    const double den = 2.0 * std::sqrt(2.0 * static_cast<double>(n)) * pi * (1.0 - pi);
    r.statistic = num / den;
    r.p_value = erfcApprox(r.statistic);
    r.passed = r.p_value >= kPValueThreshold;
    return r;
}

/**
 * @brief 4) Longest Run of Ones in a Block — самая длинная серия единиц в блоке.
 * @param bits Битовая последовательность.
 * @return TestResult.
 */
inline TestResult longestRunOfOnesTest(const std::vector<int>& bits) {
    TestResult r;
    r.test_name = "Longest Run of Ones in a Block";

    const int n = static_cast<int>(bits.size());

    // Выбор размера блока M по NIST SP 800-22.
    int M = 8;
    std::array<double, 5> pi = {0.2148, 0.3672, 0.2305, 0.1172, 0.0703};  // M=8

    if (n >= 6272) {
        M = 128;
        pi = {0.1174, 0.2430, 0.2493, 0.1752, 0.1027};
    }
    if (n >= 750000) {
        M = 10000;
        pi = {0.0882, 0.2092, 0.2483, 0.1933, 0.1208};
    }

    if (n < M) {
        r.passed = false;
        r.p_value = 0.0;
        return r;
    }

    const int N = n / M;
    std::array<int, 5> nu = {0, 0, 0, 0, 0};

    for (int i = 0; i < N; ++i) {
        int max_run = 0;
        int current = 0;
        for (int j = 0; j < M; ++j) {
            if (bits[static_cast<std::size_t>(i * M + j)] == 1) {
                ++current;
                max_run = std::max(max_run, current);
            } else {
                current = 0;
            }
        }

        // Категоризация по NIST: v — длина самой длинной серии единиц в блоке.
        int category = 0;
        if (M == 8) {
            if (max_run <= 1) {
                category = 0;
            } else if (max_run == 2) {
                category = 1;
            } else if (max_run == 3) {
                category = 2;
            } else if (max_run == 4) {
                category = 3;
            } else {
                category = 4;
            }
        } else if (M == 128) {
            if (max_run <= 4) {
                category = 0;
            } else if (max_run == 5) {
                category = 1;
            } else if (max_run == 6) {
                category = 2;
            } else if (max_run == 7) {
                category = 3;
            } else {
                category = 4;
            }
        } else {  // M = 10000
            if (max_run <= 10) {
                category = 0;
            } else if (max_run == 11) {
                category = 1;
            } else if (max_run == 12) {
                category = 2;
            } else if (max_run == 13) {
                category = 3;
            } else {
                category = 4;
            }
        }
        ++nu[static_cast<std::size_t>(category)];
    }

    double chi2 = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double expected = static_cast<double>(N) * pi[static_cast<std::size_t>(i)];
        const double diff = static_cast<double>(nu[static_cast<std::size_t>(i)]) - expected;
        chi2 += (diff * diff) / expected;
    }

    r.statistic = chi2;
    r.p_value = chi2PValue(chi2, 4);
    r.passed = r.p_value >= kPValueThreshold;
    return r;
}

/**
 * @brief 5) Serial Test — частоты перекрывающихся пар и троек битов.
 * @param bits Битовая последовательность.
 * @return TestResult.
 */
inline TestResult serialTest(const std::vector<int>& bits) {
    TestResult r;
    r.test_name = "Serial Test";

    const int n = static_cast<int>(bits.size());
    if (n < 16) {
        r.passed = false;
        r.p_value = 0.0;
        return r;
    }

    const int m = 2;
    const int pattern_count = 1 << m;  // 2^m
    std::vector<int> psi2_m(pattern_count, 0);

    for (int i = 0; i <= n - m; ++i) {
        int pattern = 0;
        for (int j = 0; j < m; ++j) {
            pattern = (pattern << 1) | bits[static_cast<std::size_t>(i + j)];
        }
        ++psi2_m[static_cast<std::size_t>(pattern)];
    }

    double sum_sq = 0.0;
    for (int count : psi2_m) {
        sum_sq += static_cast<double>(count) * static_cast<double>(count);
    }

    const double factor = static_cast<double>(1 << m) / static_cast<double>(n - m + 1);
    const double psi2 = factor * sum_sq - static_cast<double>(n - m + 1);

    // Для m=2: delta^2 psi^2 = psi^2(m) - 2*psi^2(m-1) + psi^2(m-2), упрощённо psi^2.
    r.statistic = psi2;
    r.p_value = chi2PValue(psi2, static_cast<int>(std::pow(2, m - 1)));
    r.passed = r.p_value >= kPValueThreshold;
    return r;
}

/**
 * @brief Запускает все пять NIST-тестов на битовой последовательности.
 * @param bits Битовая последовательность.
 * @return Вектор из пяти TestResult.
 */
inline std::vector<TestResult> runAllNistTests(const std::vector<int>& bits) {
    return {
        monobitTest(bits),
        frequencyWithinBlockTest(bits),
        runsTest(bits),
        longestRunOfOnesTest(bits),
        serialTest(bits),
    };
}

/**
 * @brief Подсчитывает число пройденных NIST-тестов.
 * @param results Результаты тестов.
 * @return Количество тестов с passed == true.
 */
inline int countPassedNistTests(const std::vector<TestResult>& results) {
    return static_cast<int>(std::count_if(results.begin(), results.end(),
                                          [](const TestResult& t) { return t.passed; }));
}

}  // namespace prng

#endif  // TESTS_H
