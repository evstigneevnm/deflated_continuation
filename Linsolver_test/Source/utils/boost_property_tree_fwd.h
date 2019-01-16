#ifndef __BOOST_PROPERTY_TREE_FWD_H__
#define __BOOST_PROPERTY_TREE_FWD_H__

#include <string>

namespace boost {
namespace property_tree {
template<typename Key, typename Data, typename KeyCompare> 
class basic_ptree;
typedef basic_ptree<std::string, std::string, std::less<std::string> > ptree;
}
}

#endif
