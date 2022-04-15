//
// Created by conor on 09/04/2022.
//

#ifndef MAIN_SIGNPADDING_CUH
#define MAIN_SIGNPADDING_CUH


#include <locale>

class SignPadding : public std::num_put<char> {
public:
    iter_type do_put(iter_type s,
                     std::ios_base& f,
                     char_type fill,
                     double v) const override;
};

#endif //MAIN_SIGNPADDING_CUH
