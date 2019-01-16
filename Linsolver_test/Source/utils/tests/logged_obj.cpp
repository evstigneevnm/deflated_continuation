
#include <cstdarg>
#include <utils/logged_obj_base.h>
#include <utils/log.h>

using namespace utils;

class logged_obj : public logged_obj_base<log_std>
{
public:
    logged_obj(log_std *log_ = NULL) : logged_obj_base<log_std>(log_) {}

    void test()
    {
        info("logged_obj: test to log");
        error("logged_obj: test error to log");
    }
};

template<class Log>
class logged_obj_template : public logged_obj_base<Log>
{
    typedef logged_obj_base<Log> logged_obj_t;
public:
    logged_obj_template(log_std *log_ = NULL) : logged_obj_base<log_std>(log_) {}

    void test()
    {
        logged_obj_t::info("logged_obj_template: test to log");
        logged_obj_t::error("logged_obj_template: test error to log");
        logged_obj_t::set_log_msg_prefix("logged_obj_template: ");
        logged_obj_t::info_f("test prefixed log");
        logged_obj_t::set_log_msg_prefix("");
        logged_obj_t::info("logged_obj_template: test to log normal againg");
    }
};

int main()
{
    log_std                         l;
    logged_obj                      o1(&l);
    logged_obj_template<log_std>    o2(&l);

    o1.test();
    o2.test();

    return 0;
}