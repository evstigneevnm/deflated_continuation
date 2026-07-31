#ifndef __BIFURCATION_DIAGRAM_ARCHIVE_H__
#define __BIFURCATION_DIAGRAM_ARCHIVE_H__

#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace container
{

enum class diagram_archive_status
{
    success,
    missing,
    open_failed,
    archive_failed,
    commit_failed
};

struct diagram_archive_result
{
    diagram_archive_status status = diagram_archive_status::open_failed;
    std::string message;

    bool succeeded() const
    {
        return status == diagram_archive_status::success;
    }
};

template<class Diagram>
diagram_archive_result load_diagram_archive(
    const std::string& file_name,
    Diagram& diagram)
{
    std::ifstream input(file_name);
    if(!input)
    {
        std::error_code error;
        const bool exists = std::filesystem::exists(file_name, error);
        return {
            exists
                ? diagram_archive_status::open_failed
                : diagram_archive_status::missing,
            "unable to open archive for reading"};
    }

    try
    {
        boost::archive::text_iarchive archive(input);
        archive >> diagram;
    }
    catch(const boost::archive::archive_exception& error)
    {
        return {diagram_archive_status::archive_failed, error.what()};
    }
    catch(const std::exception& error)
    {
        return {diagram_archive_status::archive_failed, error.what()};
    }
    return {diagram_archive_status::success, {}};
}

template<class Diagram>
diagram_archive_result save_diagram_archive(
    const std::string& file_name,
    const Diagram& diagram)
{
    const std::filesystem::path destination(file_name);
    const std::filesystem::path temporary(
        destination.string() + ".tmp");
    const auto remove_temporary = [&temporary]()
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
    };

    remove_temporary();
    std::ofstream output(temporary);
    if(!output)
    {
        return {
            diagram_archive_status::open_failed,
            "unable to open temporary archive for writing"};
    }

    try
    {
        {
            boost::archive::text_oarchive archive(output);
            archive << diagram;
        }
        output.flush();
        output.close();
        if(!output)
        {
            remove_temporary();
            return {
                diagram_archive_status::open_failed,
                "archive write did not complete"};
        }
    }
    catch(const boost::archive::archive_exception& error)
    {
        output.close();
        remove_temporary();
        return {diagram_archive_status::archive_failed, error.what()};
    }
    catch(const std::exception& error)
    {
        output.close();
        remove_temporary();
        return {diagram_archive_status::archive_failed, error.what()};
    }

    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if(error)
    {
        remove_temporary();
        return {
            diagram_archive_status::commit_failed,
            "unable to replace archive: " + error.message()};
    }

    return {diagram_archive_status::success, {}};
}

} // namespace container

#endif // __BIFURCATION_DIAGRAM_ARCHIVE_H__
