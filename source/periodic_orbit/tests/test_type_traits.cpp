#include <iostream>


struct cat
{
    void meow()const
    {
        std::cout << "meow" << std::endl;
    }
};

struct dog
{
    void bark()const
    {
        std::cout << "bark" << std::endl;
    }
};

template <typename, typename = void>
struct has_meow : std::false_type { };
template <typename T>
struct has_meow<T, void_t<decltype(std::declval<T>().meow() )>>
    : std::true_type { };

template <typename, typename = void>
struct has_bark : std::false_type { };
template <typename T>
struct has_bark<T, void_t<decltype(std::declval<T>().bark() )>>
    : std::true_type { };

class A
{
    template <typename T>
    auto make_noise(const T& x)
        -> typename std::enable_if<has_meow<T>{}>::type
    {
        x.meow();
    }
    template <typename T>
    auto make_noise(const T& x)
        -> typename std::enable_if<has_bark<T>{}>::type
    {
        x.bark();
    }

};


int main(int argc, char const *argv[])
{
    cat p;
    doc s;
    A a;
    a.make_noise(p);
    
    return 0;
}