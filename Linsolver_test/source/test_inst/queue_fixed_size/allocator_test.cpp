#include <cassert>
#include <memory>
#include <vector>
#include <deque>

class Arena {
public:
    Arena() {}
    ~Arena() {
        assert(m_allocations == 0);
    }

    void* allocate(std::size_t n) {
        if (n > m_available) {
            m_chunks.emplace_back(100500);
            m_available = m_chunks.back().size();
            m_memory = &m_chunks.back().front();
        }

        auto mem = m_memory;
        m_available -= n;
        m_memory += n;
        ++m_allocations;
        return mem;
    }
    void deallocate(void* p, std::size_t n) {
        --m_allocations;
        auto mem = (unsigned char*)p;
        if (mem + n == m_memory) {
            m_memory = mem;
            m_available += n;
        }
    }

private:
    std::deque<std::vector<unsigned char>> m_chunks;
    std::size_t m_available = 0;
    unsigned char* m_memory;
    int m_allocations = 0;
};

template <class T>
struct ArenaAllocator {
    using value_type = T;

    // using Traits = std::allocator_traits<ArenaAllocator<T>>;

// #if !defined _MSC_VER    
//     // libstdc++ использует конструктор по умолчанию:
//     // __a == _Alloc()
//     ArenaAllocator() : m_arena(nullptr) {}

//     // libstdc++ требует следующие определения
//     using size_type = typename std::allocator<T>::size_type;
//     using difference_type = typename std::allocator<T>::difference_type;
//     using pointer = typename std::allocator<T>::pointer;
//     using const_pointer = typename std::allocator<T>::const_pointer;
//     // "reference" не входит Allocator Requirements,
//     // но libstdc++ думает что она всегда работает с std::allocator.
//     using reference = typename std::allocator<T>::reference;
//     using const_reference = typename std::allocator<T>::const_reference;
// #endif

    explicit ArenaAllocator(Arena& arena) : m_arena(&arena) {}
    template<class U> ArenaAllocator(const ArenaAllocator<U>& other) : m_arena(other.m_arena) {}

    T* allocate(std::size_t n) { return (T*)m_arena->allocate(n * sizeof(T)); }
    void deallocate(T* p, std::size_t n) { m_arena->deallocate(p, n * sizeof(T)); }

    // требуется в VC++ и libstdc++
    //template<class U, class... Args> void construct(U* p, Args&&... args) { std::allocator<T>().construct(p, std::forward<Args>(args)...); }
    //template<class U> void destroy(U* p) { std::allocator<T>().destroy(p); }
    //template<class U> struct rebind { using other = ArenaAllocator<U>; };

    Arena* m_arena;
};

//template<class T, class U> bool operator==(const ArenaAllocator<T>& lhs, const ArenaAllocator<U>& rhs) { return lhs.m_arena == rhs.m_arena; }
//template<class T, class U> bool operator!=(const ArenaAllocator<T>& lhs, const ArenaAllocator<U>& rhs) { return !(lhs == rhs); }

#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

using a_string = std::basic_string<char, std::char_traits<char>, ArenaAllocator<char>>;

template <class T> using a_vector = std::vector<T, ArenaAllocator<T>>;
template <class T> using a_deque= std::deque<T, ArenaAllocator<T>>;
template <class T> using a_list = std::list<T, ArenaAllocator<T>>;
template <class K> using a_set = std::set<K, std::less<K>, ArenaAllocator<K>>;
template <class K, class V> using a_map = std::map<K, V, std::less<K>, ArenaAllocator<std::pair<const K, V>>>;
template <class K> using a_unordered_set = std::unordered_set<K, std::hash<K>, std::equal_to<K>, ArenaAllocator<K>>;
template <class K, class V> using a_unordered_map = std::unordered_map<K, std::hash<K>, std::equal_to<K>, ArenaAllocator<std::pair<const K, V>>>;

struct X {};

int main() 
{
    Arena arena;
    ArenaAllocator<char> arena_allocator(arena);

    // a_string s_empty(arena_allocator);
    // a_string s_123("123", arena_allocator);

    // a_vector<int> v_int({1, 2, 3}, arena_allocator);
    // a_vector<X> v_x(42, X{}, arena_allocator);
    // a_vector<a_string> v_str({s_empty, s_123}, arena_allocator);
    // a_vector<a_string> v_str_copy(v_str, arena_allocator);
    a_deque<int> d_int({1, 2, 3}, arena_allocator);
    // a_list<int> l_int({1, 2, 3}, arena_allocator);
    // a_set<int> s_int({1, 2, 3}, std::less<int>{}, arena_allocator);
    // a_map<a_string, int> m_str_int(arena_allocator);
    // a_unordered_set<int> us_int(arena_allocator);

    // auto p = std::allocate_shared<int>(arena_allocator, 123);



#if 0 // этот код не работает в VC++ и g++
    a_unordered_map<a_string, int> um_str_int(arena_allocator); 
    std::function<void()> f(std::allocator_arg_t{}, arena_allocator, []{});
#endif
    return 0;
}