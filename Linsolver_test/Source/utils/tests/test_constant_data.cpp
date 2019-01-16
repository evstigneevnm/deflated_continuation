// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#include <cstdio>
#include <utils/constant_data.h>

struct t_test
{
    int x;
};

DEFINE_CONSTANT_BUFFER(t_test, buf)

int main()
{
    t_test  test;
    test.x = 193;

    COPY_TO_CONSTANT_BUFFER(buf, test);

    printf("buf().x = %d\n", buf().x);

    return 0;
}