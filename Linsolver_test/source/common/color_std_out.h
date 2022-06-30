#ifndef __COLOR_STD_OUT_H__
#define __COLOR_STD_OUT_H__

//https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal


#include <ostream>
namespace color 
{
    enum code 
    {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class modifier 
    {
        code code_;
    public:
        Modifier(code pcode) : code_(pcode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const modifier& mod) 
        {
            return os << "\033[" << mod.code_ << "m";
        }
    };
}


// #include <common/color_std_out.h>
// #include <iostream>
// int main() 
// {
//     color::modifier red(color::FG_RED);
//     color::modifier def(color::FG_DEFAULT);
//     cout << "This ->" << red << "word" << def << "<- is red." << endl;
//     return(0);
// }

#endif