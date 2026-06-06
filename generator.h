/**
 * @file generator.h
 * @brief Интерфейс и реализации генераторов псевдослучайных последовательностей (ПСП).
 *
 * Содержит единый абстрактный интерфейс IGenerator и три кастомных генератора:
 * - LCG (линейный конгруэнтный метод);
 * - Von Neumann с модификацией Вейля (метод середины квадрата);
 * - Xorshift (сдвиговый регистр).
 *
 * Также предоставляет обёртку над std::mt19937 для сравнительного бенчмарка.
 *
 * @author Лабораторная работа по ПСП
 * @date 2026
 */

#ifndef GENERATOR_H
#define GENERATOR_H

#include <cstdint>
#include <random>
#include <string>

/**
 * @brief Абстрактный интерфейс генератора псевдослучайных чисел.
 *
 * Все реализации обязаны предоставлять метод next() для получения
 * целого числа в заданном диапазоне [min, max] включительно.
 */
class IGenerator {
public:
    virtual ~IGenerator() = default;

    /**
     * @brief Генерирует следующее псевдослучайное целое число.
     * @param min Нижняя граница диапазона (включительно).
     * @param max Верхняя граница диапазона (включительно).
     * @return Число в диапазоне [min, max].
     */
    virtual std::uint32_t next(std::uint32_t min, std::uint32_t max) = 0;

    /**
     * @brief Возвращает сырое 32-битное значение без нормализации в диапазон.
     *
     * Используется для побитовых NIST-тестов.
     * @return Следующее 32-битное псевдослучайное значение.
     */
    virtual std::uint32_t nextRaw() = 0;

    /**
     * @brief Устанавливает начальное состояние (зерно) генератора.
     * @param seed Значение зерна.
     */
    virtual void seed(std::uint64_t seed) = 0;

    /**
     * @brief Возвращает человекочитаемое имя генератора.
     * @return Строковое имя метода.
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Линейный конгруэнтный генератор (LCG).
 *
 * Рекуррентная формула: X_{n+1} = (a * X_n + c) mod m.
 *
 * Используются кастомные коэффициенты (не стандартные glibc):
 * - a = 1'103'515'245
 * - c = 12'345
 * - m = 2^31
 */
class LCGGenerator final : public IGenerator {
public:
    static constexpr std::uint64_t kA = 1'103'515'245ULL;
    static constexpr std::uint64_t kC = 12'345ULL;
    static constexpr std::uint64_t kM = 1ULL << 31;

    /**
     * @brief Конструктор LCG.
     * @param seed Начальное значение X_0 (должно быть в [0, m)).
     */
    explicit LCGGenerator(std::uint64_t seed = 12'345'678ULL) { this->seed(seed); }

    std::uint32_t next(std::uint32_t min, std::uint32_t max) override {
        state_ = (kA * state_ + kC) % kM;
        const std::uint64_t span = static_cast<std::uint64_t>(max) - min + 1ULL;
        return min + static_cast<std::uint32_t>(state_ % span);
    }

    std::uint32_t nextRaw() override {
        state_ = (kA * state_ + kC) % kM;
        return static_cast<std::uint32_t>(state_);
    }

    void seed(std::uint64_t seed) override {
        state_ = seed % kM;
        if (state_ == 0) {
            state_ = 1;
        }
    }

    std::string name() const override { return "LCG"; }

private:
    std::uint64_t state_ = 0;
};

/**
 * @brief Генератор фон Неймана (метод середины квадрата) с модификацией Вейля.
 *
 * Базовый алгоритм: X_{n+1} = middle_digits(X_n^2).
 * Модификация Вейля предотвращает быстрое вырождение в ноль:
 * X_{n+1} = middle_digits(X_n^2 + w_n),  w_{n+1} = w_n + delta.
 *
 * delta — константа Вейля (золотое сечение в 64-битном представлении).
 */
class VonNeumannGenerator final : public IGenerator {
public:
    static constexpr std::uint64_t kWeylDelta = 0x9E37'9B97'F4A7'C15ULL;

    /**
     * @brief Конструктор генератора фон Неймана.
     * @param seed Начальное 32-битное состояние (не ноль).
     */
    explicit VonNeumannGenerator(std::uint64_t seed = 8'765'432'109ULL) { this->seed(seed); }

    std::uint32_t next(std::uint32_t min, std::uint32_t max) override {
        advance();
        const std::uint64_t span = static_cast<std::uint64_t>(max) - min + 1ULL;
        return min + static_cast<std::uint32_t>(state_ % span);
    }

    std::uint32_t nextRaw() override {
        advance();
        return state_;
    }

    void seed(std::uint64_t seed) override {
        state_ = static_cast<std::uint32_t>(seed);
        if (state_ == 0) {
            state_ = 1;
        }
        weyl_ = kWeylDelta;
    }

    std::string name() const override { return "VonNeumann+Weyl"; }

private:
    void advance() {
        const std::uint64_t squared =
            static_cast<std::uint64_t>(state_) * static_cast<std::uint64_t>(state_);
        const std::uint64_t mixed = squared + weyl_;
        weyl_ += kWeylDelta;
        // Извлекаем средние 32 бита из 64-битного квадрата.
        state_ = static_cast<std::uint32_t>((mixed >> 16) & 0xFFFF'FFFFULL);
        if (state_ == 0) {
            state_ = static_cast<std::uint32_t>(weyl_ >> 32) | 1U;
        }
    }

    std::uint32_t state_ = 0;
    std::uint64_t weyl_ = kWeylDelta;
};

/**
 * @brief Сдвиговый генератор Xorshift32.
 *
 * Реализует классический алгоритм Марсаглии:
 * x ^= x << 13; x ^= x >> 17; x ^= x << 5.
 */
class XorshiftGenerator final : public IGenerator {
public:
    /**
     * @brief Конструктор Xorshift.
     * @param seed Начальное состояние (не ноль).
     */
    explicit XorshiftGenerator(std::uint64_t seed = 2'462'353'482ULL) { this->seed(seed); }

    std::uint32_t next(std::uint32_t min, std::uint32_t max) override {
        const std::uint32_t raw = nextRaw();
        const std::uint64_t span = static_cast<std::uint64_t>(max) - min + 1ULL;
        return min + static_cast<std::uint32_t>(raw % span);
    }

    std::uint32_t nextRaw() override {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return state_;
    }

    void seed(std::uint64_t seed) override {
        state_ = static_cast<std::uint32_t>(seed);
        if (state_ == 0) {
            state_ = 0xDEAD'BEEFU;
        }
    }

    std::string name() const override { return "Xorshift32"; }

private:
    std::uint32_t state_ = 0;
};

/**
 * @brief Обёртка над стандартным генератором std::mt19937 (Вихрь Мерсенна).
 *
 * Используется только для сравнительного бенчмарка производительности.
 */
class MT19937Generator final : public IGenerator {
public:
    explicit MT19937Generator(std::uint64_t seed = 42ULL) { this->seed(seed); }

    std::uint32_t next(std::uint32_t min, std::uint32_t max) override {
        std::uniform_int_distribution<std::uint32_t> dist(min, max);
        return dist(rng_);
    }

    std::uint32_t nextRaw() override {
        return static_cast<std::uint32_t>(rng_());
    }

    void seed(std::uint64_t seed) override { rng_.seed(static_cast<std::uint32_t>(seed)); }

    std::string name() const override { return "std::mt19937"; }

private:
    std::mt19937 rng_;
};

/**
 * @brief Нормализует сырое значение в целочисленный диапазон [min, max].
 * @param raw Сырое 32-битное значение.
 * @param min Нижняя граница.
 * @param max Верхняя граница.
 * @return Число в [min, max].
 */
inline std::uint32_t normalizeToRange(std::uint32_t raw, std::uint32_t min, std::uint32_t max) {
    const std::uint64_t span = static_cast<std::uint64_t>(max) - min + 1ULL;
    return min + static_cast<std::uint32_t>(raw % span);
}

#endif  // GENERATOR_H
