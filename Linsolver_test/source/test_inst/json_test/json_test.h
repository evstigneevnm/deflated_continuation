#include <vector>
#include <string>
#include <fstream>

//includes json library by nlohmann
#include <contrib/json/nlohmann/json.hpp>


template<class T>
struct parameters
{
    struct internal1_s
    {
        struct internal2_s
        {
            T a_2;
            T b_2;
            std::vector<T> keys;

            void set_default()
            {
                a_2 = 0.0;
                b_2 = 0.0;
                keys = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
            }
        };
        T a_i;
        T b_i;
        internal2_s internal;

        void set_default()
        {
            a_i = 0.0;
            b_i = 0.0;
            internal.set_default();
        }
    };

    T a_e;
    T b_e;
    internal1_s internal;
    void set_default()
    {
        a_e = 0.0;
        b_e = 0.0;
        internal.set_default();
    }
};

typedef parameters<double> parameters_d;
typedef parameters<float> parameters_f;


void from_json(const nlohmann::json &j, parameters_d::internal1_s::internal2_s &internal2_)
{
    internal2_ = parameters_d::internal1_s::internal2_s
    {
        j.at("a_i").get<double>(),
        j.at("b_i").get<double>(),
        j.at("keys").get< std::vector<double> >()
    };
}
void from_json(const nlohmann::json &j, parameters_f::internal1_s::internal2_s &internal2_)
{
    internal2_ = parameters_f::internal1_s::internal2_s
    {
        j.at("a_i").get<float>(),
        j.at("b_i").get<float>(),
        j.at("keys").get< std::vector<float> >()
    };
}

void from_json(const nlohmann::json &j, parameters_d::internal1_s &internal1_)
{
    internal1_ = parameters_d::internal1_s
    {
        j.at("a_i").get<double>(),
        j.at("b_i").get<double>(),
        j.at("inlined_internal").get<parameters_d::internal1_s::internal2_s>()
    };
}
void from_json(const nlohmann::json &j, parameters_f::internal1_s &internal1_)
{
    internal1_ = parameters_f::internal1_s
    {
        j.at("a_i").get<float>(),
        j.at("b_i").get<float>(),
        j.at("inlined_internal").get<parameters_f::internal1_s::internal2_s>()
    };
}


void from_json(const nlohmann::json &j, parameters_d &parameters_)
{
    parameters_ = parameters_d
    {
        j.at("a_i").get<double>(),
        j.at("b_i").get<double>(),
        j.at("internal").get<parameters_d::internal1_s>()
    };
}
void from_json(const nlohmann::json &j, parameters_f &parameters_)
{
    parameters_ = parameters_f
    {
        j.at("a_i").get<float>(),
        j.at("b_i").get<float>(),
        j.at("internal").get<parameters_f::internal1_s>()
    };
}

nlohmann::json read_json_file(const std::string &file_path)
{
    try
    {
        std::ifstream f(file_path);
        if (f)
        {
            std::cout << "file opened..." << std::endl;
            nlohmann::json j;
            f >> j;
            return j;
        }
        else
        {
            throw std::runtime_error(std::string("Failed to open file ") + file_path + " for reading");
        }
    }
    catch (const nlohmann::json::exception &exception)
    {
        std::throw_with_nested(std::runtime_error{"json path: " + file_path});
    }
}


template<class T>
parameters<T> read_json(const std::string &file_name_)
{
    
    return read_json_file(file_name_).get< parameters<T> >();

}
