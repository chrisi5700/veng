//
// Created by chris on 1/22/26.
//

#ifndef VENG_COMMON_HPP
#define VENG_COMMON_HPP

template<class... Fs>
struct overloaded : Fs...
{
	using Fs::operator()...;
};

#endif // VENG_COMMON_HPP
