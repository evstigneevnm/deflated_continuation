#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>
#include <vector>



std::vector<std::string> match_file_names(const std::string& path, const std::string& regex_mask)
{
    std::regex rx(regex_mask);

    const std::filesystem::path current_folder{path};

    std::vector<std::string> matched_file_names;

    for(auto const& dir_entry: std::filesystem::directory_iterator{current_folder})
    {
        std::string path_and_file_name( dir_entry.path() );

        std::ptrdiff_t number_of_matches = std::distance( std::sregex_iterator(path_and_file_name.begin(), path_and_file_name.end(), rx ), std::sregex_iterator() );
        if(number_of_matches > 0)
        {
            // std::cout << path_and_file_name << " number_of_matches = " << number_of_matches << std::endl;
            matched_file_names.push_back(path_and_file_name);
        }

    }
    std::sort( matched_file_names.begin(), matched_file_names.end() );
    return matched_file_names;
}



int main(int argc, char const *argv[]) 
{
    
    if(argc != 3)
    {
        std::cout << argv[0] << " path_to_root_folder \"regex_file_mask\" " << std::endl;
        std::cout << "argc = " << argc << std::endl;
        for(int j = 2; j<argc;j++)
            std::cout << argv[j] << std::endl;
        return 0;
    }
    std::cout <<  std::endl;
    std::string path(argv[1]);
    std::string mask(argv[2]);
    auto found_files = match_file_names(path, mask);
    for(auto &v: found_files)
    {
        std::cout << v << std::endl;
    }


    return 0;
}