// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __LOG_H__
#define __LOG_H__

#include <string>
#include <exception>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>

namespace utils
{

class log
{
    char    buf[200];
public:
    //INFO_ALL refers to multi-process applications (like MPI) and means message, that must be said distintly by each process (like, 'i'm 1st; i'm second etc')
    enum t_msg_type { INFO, INFO_ALL, WARNING, ERROR };

    //lesser log level corresponds to more important messages
    virtual void msg(const std::string &s, t_msg_type mt = INFO, int _log_lev = 1) = 0;
    void info(const std::string &s, int _log_lev = 1)
    {
        msg(s, INFO, _log_lev);
    }
    void info_all(const std::string &s, int _log_lev = 1)
    {
        msg(s, INFO_ALL, _log_lev);
    }
    void warning(const std::string &s, int _log_lev = 1)
    {
        msg(s, WARNING, _log_lev);
    }
    void error(const std::string &s, int _log_lev = 1)
    {
        msg(s, ERROR, _log_lev);
    }   
    

    #define LOG__FORMATTED_OUT_V__(METHOD_NAME,LOG_LEV)   \
        vsprintf(buf, s.c_str(), arguments);              \
        METHOD_NAME(std::string(buf), LOG_LEV);
    void v_info_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info, _log_lev)
    }
    void v_info_all_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info_all, _log_lev)
    }
    void v_warning_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(warning, _log_lev)
    }
    void v_error_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(error, _log_lev)
    }
    void v_info_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info, 1)
    }
    void v_info_all_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info_all, 1)
    }
    void v_warning_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(warning, 1)
    }
    void v_error_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(error, 1)
    }
    #undef LOG__FORMATTED_OUT_V__ 


    #define LOG__FORMATTED_OUT__(METHOD_NAME,LOG_LEV)   \
        va_list arguments;                              \
        va_start ( arguments, s );                      \
        vsprintf(buf, s.c_str(), arguments);            \
        METHOD_NAME(std::string(buf), LOG_LEV);         \
        va_end ( arguments );   
    void info_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info, _log_lev)
    }
    void info_all_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info_all, _log_lev)
    }
    void warning_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(warning, _log_lev)
    }
    void error_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(error, _log_lev)
    }
    void info_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info, 1)
    }
    void info_all_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info_all, 1)
    }
    void warning_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(warning, 1)
    }
    void error_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(error, 1)
    }
    #undef LOG__FORMATTED_OUT__ 

    //set_verbosity sets maximum level of messages to log
    //NOTE log_lev doesnot affects errors
    virtual void set_verbosity(int _log_lev = 1) = 0;
};

class log_std : public log
{
    int     log_lev;
public:
    log_std() : log_lev(1) {}

    virtual void msg(const std::string &s, t_msg_type mt = INFO, int _log_lev = 1)
    {
        if ((mt != ERROR)&&(_log_lev > log_lev)) return;
        //TODO
        if ((mt == INFO)||(mt == INFO_ALL))
            printf("INFO:    %s\n", s.c_str());
        else if (mt == WARNING)
            printf("WARNING: %s\n", s.c_str());
        else if (mt == ERROR)
            printf("ERROR:   %s\n", s.c_str());
        else 
            throw std::logic_error("log_std::log: wrong t_msg_type argument");
    }
    virtual void set_verbosity(int _log_lev = 1) { log_lev = _log_lev; }
};

}

#endif
