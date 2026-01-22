//
// Created by chris on 1/22/26.
//

#ifndef VENG_COMMON_HPP
#define VENG_COMMON_HPP
#include <variant>
#include <expected>

template<class... Fs>
struct overloaded : Fs...
{
	using Fs::operator()...;
};

template<class T>
concept ResultType = requires(const T& t)
{
	to_string(t);
};

template<ResultType... Ts>
struct ResultVariant : std::variant<Ts...>
{
	using Base = std::variant<Ts...>;
	using Base::Base;
};

template<ResultType... Ts>
std::string to_string(const ResultVariant<Ts...>& var)
{
	return std::visit([](const auto& inner)
	{
		return to_string(inner);
	}, var);
}

/**
 * A function might fail in various ways, this let's me flatten those into one std::expected type
 */
template<class...>
struct merge_expected;

template<class Out, class... OtherOuts, ResultType... Errors>
struct merge_expected<Out, std::expected<OtherOuts, Errors>...>
{
	using type = std::expected<Out, ResultVariant<Errors...>>;
};

template<class Out, class OwnErrors, class... OtherOuts, ResultType... Errors>
struct merge_expected<Out, OwnErrors, std::expected<OtherOuts, Errors>...>
{
	using type = std::expected<Out, ResultVariant<OwnErrors, Errors...>>;
};

template<class...Ts>
using merge_expected_t = merge_expected<Ts...>::type;


template<class T> requires std::convertible_to<T, vk::Result>
std::string to_string(const T& t)
{
	return vk::to_string(t);
}




#endif // VENG_COMMON_HPP
