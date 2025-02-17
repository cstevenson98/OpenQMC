//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 09/04/2022.
//


#include <iostream>
#include <cmath>
#include "utils/SignPadding.cuh"

SignPadding::iter_type SignPadding::do_put(SignPadding::iter_type s,
                                           std::ios_base& f,
                                           SignPadding::char_type fill,
                                           double v) const
{
    if (!std::signbit(v)) *s++ = ' ';
    return std::num_put<char>::do_put(s, f, fill, v);
}
