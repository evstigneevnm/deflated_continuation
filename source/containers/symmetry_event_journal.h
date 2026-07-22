#ifndef __CONTAINERS_SYMMETRY_EVENT_JOURNAL_H__
#define __CONTAINERS_SYMMETRY_EVENT_JOURNAL_H__

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <containers/symmetry_event_record.h>

namespace container
{

template<class T>
class symmetry_event_journal
{
public:
    using record_type = symmetry_event_record<T>;

    symmetry_event_journal() = default;

    explicit symmetry_event_journal(std::filesystem::path file_name_):
        file_name(std::move(file_name_))
    {
    }

    void set_file_name(std::filesystem::path file_name_)
    {
        file_name = std::move(file_name_);
    }

    void stage(const record_type& record)
    {
        records.push_back(record);
        dirty = true;
    }

    const std::vector<record_type>& all() const
    {
        return records;
    }

    std::vector<record_type>& all_mutable()
    {
        return records;
    }

    bool has_staged_changes() const
    {
        return dirty;
    }

    bool commit()
    {
        if(file_name.empty())
        {
            return false;
        }

        const std::filesystem::path temporary = file_name.string() + ".tmp";
        std::ofstream stream(temporary, std::ofstream::out | std::ofstream::trunc);
        if(!stream)
        {
            return false;
        }
        stream << "# symmetry_events_version 2\n";
        stream
            << "# point_index lambda vector_available vector_file_id segment_id semicurve_id "
            << "previous_order candidate_order previous_transverse_ratio candidate_transverse_ratio "
            << "refinements\n";
        for(const auto& record: records)
        {
            stream
                << record.point_index << " "
                << std::setprecision(16) << record.lambda << " "
                << (record.vector_available ? 1 : 0) << " "
                << record.vector_file_id << " "
                << record.segment_id << " "
                << record.semicurve_id << " "
                << record.previous_order << " "
                << record.candidate_order << " "
                << std::setprecision(16) << record.previous_transverse_ratio << " "
                << record.candidate_transverse_ratio << " "
                << record.refinements << "\n";
        }
        stream.close();
        if(!stream)
        {
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            return false;
        }

        std::error_code rename_error;
        std::filesystem::rename(temporary, file_name, rename_error);
        if(rename_error)
        {
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            return false;
        }
        dirty = false;
        return true;
    }

    bool load()
    {
        records.clear();
        dirty = false;
        std::ifstream stream(file_name);
        if(!stream)
        {
            return false;
        }

        unsigned int version = 1;
        std::string line;
        while(std::getline(stream, line))
        {
            if(line.empty())
            {
                continue;
            }
            if(line[0] == '#')
            {
                std::istringstream header(line.substr(1));
                std::string label;
                header >> label;
                if(label == "symmetry_events_version")
                {
                    header >> version;
                }
                continue;
            }

            record_type record;
            std::istringstream values(line);
            if(version >= 2)
            {
                unsigned int vector_available = 0;
                values
                    >> record.point_index
                    >> record.lambda
                    >> vector_available
                    >> record.vector_file_id
                    >> record.segment_id
                    >> record.semicurve_id
                    >> record.previous_order
                    >> record.candidate_order
                    >> record.previous_transverse_ratio
                    >> record.candidate_transverse_ratio
                    >> record.refinements;
                record.vector_available = vector_available != 0;
            }
            else
            {
                values
                    >> record.point_index
                    >> record.lambda
                    >> record.segment_id
                    >> record.semicurve_id
                    >> record.previous_order
                    >> record.candidate_order
                    >> record.previous_transverse_ratio
                    >> record.candidate_transverse_ratio
                    >> record.refinements;
            }
            if(!values)
            {
                records.clear();
                return false;
            }
            record.previous_orbit_type =
                symmetry::translation::orbit_type::cyclic_1d(record.previous_order);
            record.candidate_orbit_type =
                symmetry::translation::orbit_type::cyclic_1d(record.candidate_order);
            records.push_back(record);
        }
        return true;
    }

private:
    std::filesystem::path file_name;
    std::vector<record_type> records;
    bool dirty = false;
};

} // namespace container

#endif
