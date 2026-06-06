/**
 * @file main.cpp
 * @brief Точка входа лабораторной работы по генерации и тестированию ПСП.
 *
 * Выполняет:
 * - Генерацию 20 выборок по 1000 элементов для каждого из 3 методов;
 * - Расчёт описательной статистики (среднее, дисперсия, СКО, коэфф. вариации);
 * - Проверку равномерности критерием Хи-квадрат;
 * - Пять NIST-адаптированных тестов и сравнение с Хи-квадрат;
 * - Бенчмарк производительности и выгрузку CSV для графиков.
 *
 * @author Лабораторная работа по ПСП
 * @date 2026
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "generator.h"
#include "tests.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

/** @brief Минимум диапазона генерируемых чисел. */
constexpr std::uint32_t kRangeMin = 0;

/** @brief Максимум диапазона генерируемых чисел. */
constexpr std::uint32_t kRangeMax = 5000;

/** @brief Объём каждой выборки для статистики. */
constexpr std::size_t kSampleSize = 1000;

/** @brief Число выборок на каждый генератор. */
constexpr int kNumSamples = 20;

/** @brief Число карманов для критерия Хи-квадрат. */
constexpr int kChi2Bins = 10;

/** @brief Размеры выборок для бенчмарка. */
const std::vector<std::size_t> kBenchmarkSizes = {
    1'000, 5'000, 10'000, 50'000, 100'000, 500'000, 1'000'000};

/** @brief Число прогревочных итераций бенчмарка. */
constexpr int kWarmupIterations = 3;

/** @brief Путь к CSV-файлу с результатами скорости. */
const std::string kSpeedCsvPath = "output/speed_results.csv";

/** @brief Путь к CSV-файлу с описательной статистикой выборок. */
const std::string kStatsCsvPath = "output/stats_summary.csv";

/** @brief Путь к CSV-файлу с результатами NIST-тестов. */
const std::string kNistCsvPath = "output/nist_results.csv";

/** @brief Путь к CSV-файлу с гистограммами (первая выборка каждого генератора). */
const std::string kHistogramCsvPath = "output/histogram.csv";

/**
 * @brief Volatile sink — предотвращает оптимизацию цикла компилятором.
 * @param value Значение для «поглощения».
 */
volatile std::uint32_t g_volatile_sink = 0;

/**
 * @brief Создаёт генератор по имени с уникальным зерном.
 * @param type Тип генератора: "LCG", "VonNeumann", "Xorshift".
 * @param seed Зерно.
 * @return Умный указатель на IGenerator.
 */
std::unique_ptr<IGenerator> makeGenerator(const std::string& type, std::uint64_t seed) {
    if (type == "LCG") {
        return std::make_unique<LCGGenerator>(seed);
    }
    if (type == "VonNeumann") {
        return std::make_unique<VonNeumannGenerator>(seed);
    }
    return std::make_unique<XorshiftGenerator>(seed);
}

/**
 * @brief Генерирует выборку целых чисел в диапазоне [min, max].
 * @param gen Генератор.
 * @param count Размер выборки.
 * @param min Нижняя граница.
 * @param max Верхняя граница.
 * @return Вектор значений.
 */
std::vector<std::uint32_t> generateSample(IGenerator& gen, std::size_t count,
                                          std::uint32_t min, std::uint32_t max) {
    std::vector<std::uint32_t> data;
    data.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        data.push_back(gen.next(min, max));
    }
    return data;
}

/**
 * @brief Генерирует сырую выборку 32-битных значений для NIST-тестов.
 * @param gen Генератор.
 * @param count Размер выборки.
 * @return Вектор сырых значений.
 */
std::vector<std::uint32_t> generateRawSample(IGenerator& gen, std::size_t count) {
    std::vector<std::uint32_t> data;
    data.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        data.push_back(gen.nextRaw());
    }
    return data;
}

/**
 * @brief Печатает разделитель секции.
 * @param title Заголовок секции.
 */
void printSection(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n"
              << title << "\n"
              << std::string(72, '=') << "\n";
}

/**
 * @brief Выполняет статистический анализ всех генераторов.
 */
void runStatisticalAnalysis() {
    const std::vector<std::pair<std::string, std::string>> generators = {
        {"LCG", "LCG"},
        {"VonNeumann", "VonNeumann+Weyl"},
        {"Xorshift", "Xorshift32"},
    };

    printSection("1. ГЕНЕРАЦИЯ ВЫБОРОК И ОПИСАТЕЛЬНАЯ СТАТИСТИКА");

    std::filesystem::create_directories("output");
    std::ofstream stats_csv(kStatsCsvPath);
    if (stats_csv) {
        stats_csv << "generator,sample,seed,mean,variance,std_dev,cv_percent,chi2,p_value,chi2_passed\n";
    }

    std::ofstream hist_csv(kHistogramCsvPath);
    if (hist_csv) {
        hist_csv << "generator,bin,bin_start,bin_end,count\n";
    }

    for (const auto& [type, display_name] : generators) {
        std::cout << "\n--- Генератор: " << display_name << " ---\n";
        std::cout << std::fixed << std::setprecision(4);

        int chi2_passed = 0;

        for (int s = 0; s < kNumSamples; ++s) {
            const std::uint64_t seed = static_cast<std::uint64_t>(s + 1) * 1'000'003ULL;
            auto gen = makeGenerator(type, seed);
            const auto sample = generateSample(*gen, kSampleSize, kRangeMin, kRangeMax);

            std::vector<double> as_double(sample.begin(), sample.end());
            const auto stats = prng::computeStats(as_double);
            const auto chi2 = prng::chiSquareUniformityTest(sample, kRangeMin, kRangeMax, kChi2Bins);

            if (chi2.passed) {
                ++chi2_passed;
            }

            if (stats_csv) {
                stats_csv << display_name << "," << (s + 1) << "," << seed << ","
                          << stats.mean << "," << stats.variance << "," << stats.std_dev
                          << "," << stats.coeff_of_variation << "," << chi2.chi2_statistic
                          << "," << chi2.p_value << "," << (chi2.passed ? 1 : 0) << "\n";
            }

            if (hist_csv && s == 0) {
                const double range = static_cast<double>(kRangeMax) - kRangeMin + 1.0;
                const double bin_width = range / static_cast<double>(kChi2Bins);
                std::vector<int> counts(kChi2Bins, 0);
                for (std::uint32_t v : sample) {
                    int bin = static_cast<int>((static_cast<double>(v) - kRangeMin) / bin_width);
                    if (bin >= kChi2Bins) {
                        bin = kChi2Bins - 1;
                    }
                    ++counts[static_cast<std::size_t>(bin)];
                }
                for (int b = 0; b < kChi2Bins; ++b) {
                    const auto bin_start = kRangeMin + static_cast<std::uint32_t>(b * bin_width);
                    const auto bin_end = kRangeMin + static_cast<std::uint32_t>((b + 1) * bin_width) - 1;
                    hist_csv << display_name << "," << b << "," << bin_start << "," << bin_end
                             << "," << counts[static_cast<std::size_t>(b)] << "\n";
                }
            }

            std::cout << "  Выборка " << std::setw(2) << (s + 1)
                      << " | seed=" << seed
                      << " | mean=" << stats.mean
                      << " | var=" << stats.variance
                      << " | std=" << stats.std_dev
                      << " | CV=" << stats.coeff_of_variation << "%"
                      << " | chi2=" << chi2.chi2_statistic
                      << " (df=" << chi2.degrees_of_freedom
                      << ", crit=" << chi2.critical_value
                      << ", p=" << chi2.p_value << ")"
                      << " => " << chi2.verdict() << "\n";
        }

        std::cout << "  Итого Хи-квадрат: " << chi2_passed << "/" << kNumSamples
                  << " выборок — гипотеза о равномерности ПРИНИМАЕТСЯ\n";
    }

    if (stats_csv) {
        std::cout << "Статистика выборок сохранена в " << kStatsCsvPath << "\n";
    }
    if (hist_csv) {
        std::cout << "Гистограммы сохранены в " << kHistogramCsvPath << "\n";
    }
}

/**
 * @brief Запускает NIST-тесты и сравнивает с Хи-квадрат.
 */
void runNistTests() {
    const std::vector<std::pair<std::string, std::string>> generators = {
        {"LCG", "LCG"},
        {"VonNeumann", "VonNeumann+Weyl"},
        {"Xorshift", "Xorshift32"},
    };

    printSection("2. NIST/DIEHARD ТЕСТЫ И СРАВНЕНИЕ С ХИ-КВАДРАТ");

    // Для NIST-тестов нужна длинная битовая последовательность.
    constexpr std::size_t kNistSampleSize = 10'000;

    std::ofstream nist_csv(kNistCsvPath);
    if (nist_csv) {
        nist_csv << "generator,test,statistic,p_value,passed\n";
    }

    for (const auto& [type, display_name] : generators) {
        std::cout << "\n--- Генератор: " << display_name << " ---\n";

        auto gen = makeGenerator(type, 42ULL);
        const auto raw = generateRawSample(*gen, kNistSampleSize);
        const auto bits = prng::expandToBitStream(raw, 32);

        const auto nist_results = prng::runAllNistTests(bits);
        const int nist_passed = prng::countPassedNistTests(nist_results);

        for (const auto& t : nist_results) {
            if (nist_csv) {
                nist_csv << display_name << "," << t.test_name << ","
                         << t.statistic << "," << t.p_value << ","
                         << (t.passed ? 1 : 0) << "\n";
            }

            std::cout << "  " << std::left << std::setw(40) << t.test_name
                      << " | stat=" << std::setw(10) << std::fixed << std::setprecision(4) << t.statistic
                      << " | p=" << std::setw(8) << t.p_value
                      << " => " << t.verdict() << "\n";
        }

        // Сравнение с Хи-квадрат на нормализованной выборке.
        gen->seed(42ULL);
        const auto normalized = generateSample(*gen, kNistSampleSize, kRangeMin, kRangeMax);
        const auto chi2 = prng::chiSquareUniformityTest(normalized, kRangeMin, kRangeMax, kChi2Bins);

        std::cout << "\n  Сводка:\n"
                  << "    NIST-тесты пройдены: " << nist_passed << "/5\n"
                  << "    Хи-квадрат (10 карманов, n=" << kNistSampleSize
                  << "): chi2=" << chi2.chi2_statistic
                  << ", p=" << chi2.p_value
                  << " => гипотеза о равномерности " << chi2.verdict() << "\n";

        if ((nist_passed >= 4 && chi2.passed) || (nist_passed < 3 && !chi2.passed)) {
            std::cout << "    Вывод: результаты NIST и Хи-квадрат СОГЛАСУЮТСЯ.\n";
        } else {
            std::cout << "    Вывод: результаты NIST и Хи-квадрат РАСХОДЯТСЯ"
                         " (разные аспекты случайности).\n";
        }
    }

    if (nist_csv) {
        std::cout << "Результаты NIST-тестов сохранены в " << kNistCsvPath << "\n";
    }
}

/**
 * @brief Замеряет время генерации count чисел.
 * @param gen Генератор.
 * @param count Количество чисел.
 * @return Время в миллисекундах.
 */
double benchmarkGenerator(IGenerator& gen, std::size_t count) {
    for (int w = 0; w < kWarmupIterations; ++w) {
        gen.seed(static_cast<std::uint64_t>(w + 1));
        for (std::size_t i = 0; i < count; ++i) {
            g_volatile_sink ^= gen.next(kRangeMin, kRangeMax);
        }
    }

    gen.seed(12345ULL);
    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < count; ++i) {
        g_volatile_sink ^= gen.next(kRangeMin, kRangeMax);
    }
    const auto t1 = Clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/**
 * @brief Выполняет бенчмарк и сохраняет результаты в CSV.
 */
void runBenchmark() {
    printSection("3. БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ");

    std::filesystem::create_directories("output");

    std::ofstream csv(kSpeedCsvPath);
    if (!csv) {
        std::cerr << "Ошибка: не удалось открыть " << kSpeedCsvPath << "\n";
        return;
    }

    csv << "size,LCG,VonNeumann,Xorshift,std_mt19937\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(10) << "size"
              << std::setw(14) << "LCG"
              << std::setw(18) << "VonNeumann"
              << std::setw(14) << "Xorshift"
              << std::setw(16) << "std::mt19937" << "\n";

    for (std::size_t n : kBenchmarkSizes) {
        LCGGenerator lcg(1);
        VonNeumannGenerator vn(2);
        XorshiftGenerator xs(3);
        MT19937Generator mt(4);

        const double t_lcg = benchmarkGenerator(lcg, n);
        const double t_vn = benchmarkGenerator(vn, n);
        const double t_xs = benchmarkGenerator(xs, n);
        const double t_mt = benchmarkGenerator(mt, n);

        csv << n << "," << t_lcg << "," << t_vn << "," << t_xs << "," << t_mt << "\n";

        std::cout << std::setw(10) << n
                  << std::setw(12) << t_lcg << " ms"
                  << std::setw(16) << t_vn << " ms"
                  << std::setw(14) << t_xs << " ms"
                  << std::setw(16) << t_mt << " ms\n";
    }

    std::cout << "\nРезультаты сохранены в " << kSpeedCsvPath << "\n";
}

}  // namespace

/**
 * @brief Главная функция программы.
 * @return Код завершения 0 при успехе.
 */
int main() {
    std::cout << "Лабораторная работа: Генерация и тестирование ПСП\n";
    std::cout << "Диапазон: [" << kRangeMin << ", " << kRangeMax << "], "
              << "выборки: " << kNumSamples << " x " << kSampleSize << " элементов\n";

    runStatisticalAnalysis();
    runNistTests();
    runBenchmark();

    std::cout << "\nПрограмма завершена.\n";
    return 0;
}
