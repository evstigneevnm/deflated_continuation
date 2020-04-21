#include <iostream>
#include "exception_function.hpp"

int main(int argc, char const *argv[])
{
	if(argc != 2)
	{
		std::cout << argv[0] << " N, where N is a whole number to check exception." << std::endl;
	}

	int val = atoi(argv[1]);
	bool res = false;
	try
	{
		res =  test_function(val);
		res =  test_function1(2-val);
	}
	catch(const std::exception& e)
	{
		std::cout << e.what();
		std::cout << std::endl;
	}

	return 0;
}