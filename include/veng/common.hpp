/**
 * @file
 * @author chris
 * @brief Shared error-handling vocabulary: a stringifiable error concept, an error
 *        variant, and machinery to merge several `std::expected` error types into one.
 *
 * A call can fail in many ways, each layer returning its own `std::expected<T, E>`. These
 * helpers flatten the distinct error types into a single `std::expected` whose error is a
 * @ref ResultVariant over every constituent error — so a function composing several fallible
 * calls can propagate any of their failures through one uniform result type.
 *
 * @ingroup util
 */

#ifndef VENG_COMMON_HPP
#define VENG_COMMON_HPP
#include <expected>
#include <variant>
#include <vulkan/vulkan.hpp>

/**
 * @brief Overload-set builder for `std::visit`: inherits `operator()` from each lambda.
 * @ingroup util
 * @tparam Fs The callable types whose call operators are merged into one overload set.
 */
template <class... Fs>
struct overloaded : Fs...
{
	using Fs::operator()...;
};

/**
 * @brief Constrains a type to those that can be turned into a `std::string` via an
 *        unqualified `to_string(t)` — the requirement for a type to serve as an error.
 * @ingroup util
 */
template <class T>
concept ResultType = requires(const T& t) { to_string(t); };

/**
 * @brief A variant over error types, itself a @ref ResultType so variants can nest.
 * @ingroup util
 * @tparam Ts The candidate error types this variant may hold.
 * @see merge_expected
 */
template <ResultType... Ts>
struct ResultVariant : std::variant<Ts...>
{
	using Base = std::variant<Ts...>;
	using Base::Base;
};

/**
 * @brief Stringify whichever alternative a `ResultVariant` currently holds.
 * @param var The error variant to render.
 * @return The held alternative's `to_string` result.
 */
template <ResultType... Ts>
std::string to_string(const ResultVariant<Ts...>& var)
{
	return std::visit([](const auto& inner) { return to_string(inner); }, var);
}

/**
 * @brief Flattens several `std::expected` error types into one `std::expected`.
 *
 * Given an output type and a list of `std::expected<_, E>` (optionally preceded by an extra
 * own-error type), `::type` is `std::expected<Out, ResultVariant<...all the Es...>>` — the
 * single result type a composing function returns so that any constituent failure propagates.
 *
 * @ingroup util
 * @see ResultVariant
 * @see merge_expected_t
 */
template <class...>
struct merge_expected;

/// @brief Merge specialization over a pack of `std::expected` error types.
template <class Out, class... OtherOuts, ResultType... Errors>
struct merge_expected<Out, std::expected<OtherOuts, Errors>...>
{
	using type = std::expected<Out, ResultVariant<Errors...>>;
};

/// @brief Merge specialization that also folds in the caller's own error type.
template <class Out, class OwnErrors, class... OtherOuts, ResultType... Errors>
struct merge_expected<Out, OwnErrors, std::expected<OtherOuts, Errors>...>
{
	using type = std::expected<Out, ResultVariant<OwnErrors, Errors...>>;
};

/// @brief Convenience alias for `merge_expected<Ts...>::type`.
template <class... Ts>
using merge_expected_t = merge_expected<Ts...>::type;

/**
 * @brief Make any `vk::Result`-convertible type satisfy @ref ResultType.
 * @param t A value convertible to `vk::Result`.
 * @return `vk::to_string` of the corresponding result code.
 */
template <class T>
	requires std::convertible_to<T, vk::Result>
std::string to_string(const T& t)
{
	return vk::to_string(t);
}

#endif // VENG_COMMON_HPP
