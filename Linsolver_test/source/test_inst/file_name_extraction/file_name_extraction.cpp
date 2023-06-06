#include <iostream>
#include <string>
#include <sstream>

std::string remove_extension(const std::string& filename)
{
    size_t lastslash = filename.find_last_of("/");

    size_t lastdot = filename.find_last_of(".");
    
    std::cout << "/:" << lastslash << " .:" << lastdot << std::endl;
    
    if ((lastdot == std::string::npos)&&(lastslash == std::string::npos))
    {
        return filename;
    }
    else
    {
        if (lastslash != std::string::npos)
        {
            if(lastslash<lastdot)
            {
                return filename.substr(lastslash+1, lastdot-lastslash-1);
            }
            else
            {
                return filename.substr(lastslash+1, filename.length() );
            }
            
        }
        else
        {
            return filename.substr(0, lastdot);
        }
    }
    
}

int main()
{
    std::string file_name = "../ob/0/1.dat";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl;
    file_name = "../ob/0/1";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl;
    file_name = "../1";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl; 
    file_name = "1";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl;    
    file_name = "1.txt";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl;    
    file_name = "../../dwr.re/dfnwrf/Document/../dew/_324dede1.txt";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl; 
    file_name = "~/Documents/problems/mouse.txt";
    std::cout << file_name << " -> " << remove_extension(file_name) << std::endl;

    return 0;
}