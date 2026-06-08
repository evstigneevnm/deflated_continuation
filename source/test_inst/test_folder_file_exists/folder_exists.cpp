//#include <sys/stat.h>
//#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>

inline bool exists_test0 (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// inline bool exists_test1 (const std::string& name) {
//     if (FILE *file = fopen(name.c_str(), "r")) {
//         fclose(file);
//         return true;
//     } else {
//         return false;
//     }   
// }

// inline bool exists_test2 (const std::string& name) {
//     return ( access( name.c_str(), F_OK ) != -1 );
// }

// inline bool exists_test3 (const std::string& name) {
//   struct stat buffer;   
//   return (stat (name.c_str(), &buffer) == 0); 
// }


int main(int argc, char const *argv[])
{
    std::string f_string = "test_folder";
    std::string aaa = f_string + std::string("/") + "test_file";
    if(exists_test0 (aaa))
    {
        std::cout << f_string << " exists!" << std::endl;
    }
    else
    {
        std::cout << f_string << " doesn't exist!" << std::endl;
    }


    return 0;
}