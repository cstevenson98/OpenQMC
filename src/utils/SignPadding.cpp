//
// Created by conor on 09/04/2022.
//


#include <iostream>
#include <cmath>
#include "SignPadding.h"

SignPadding::iter_type SignPadding::do_put(SignPadding::iter_type s,
                                           std::ios_base& f,
                                           SignPadding::char_type fill,
                                           double v) const
{
    if (!std::signbit(v)) *s++ = ' ';
    return std::num_put<char>::do_put(s, f, fill, v);
}
