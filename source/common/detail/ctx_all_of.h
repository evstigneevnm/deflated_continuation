#ifndef __SCFD_CTX_ALL_OF__
#define __SCFD_CTX_ALL_OF__

namespace scfd
{
namespace linspace
{
namespace detail
{

template< bool ... b> struct bool_array{};
template< bool ... b> struct ctx_all_of: std::is_same< bool_array<b...>, bool_array<(b||true)...> >{};

} // namespace detail
} // namespace linspace
} // namespace scfd



#endif
