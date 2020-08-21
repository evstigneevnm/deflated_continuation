#include <queue>
#include <deque>
#include <iostream>

template <typename T, int MaxLen, typename Container=std::deque<T>>
class FixedQueue : public std::queue<T, Container> {
public:
    void push(const T& value) 
    {
        if (this->size() == MaxLen) 
        {
           this->c.pop_front();
        }
        std::queue<T, Container>::push(value);
    }
    
    void clear()
    {
        while(!this->c.empty())
        {
            this->c.pop_front();
        }
    }
};

int main() {
    FixedQueue<int, 2> q;
    q.push(1);
    q.push(2);
    q.push(3);
    std::cout << "queue size = " << q.size() << std::endl;
    q.push(4);
    q.push(5);
    q.push(6);
    q.clear();    
    q.push(7);
    
    std::cout << "queue size = " << q.size() << std::endl;
    
    while (q.size() > 0)
    {
        std::cout << q.front() << std::endl;
        q.pop();

    }
}