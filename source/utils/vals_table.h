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

#ifndef __VALS_TABLE_H__
#define __VALS_TABLE_H__

#include <utils/device_tag.h>

namespace utils
{

template<class T, int max_vals_n>
struct vals_table_1d
{
    int     vals_n;
    T       vals[max_vals_n];

    __DEVICE_TAG__  int get_vals_n()const { return vals_n; }
    __DEVICE_TAG__  const T &operator()(int i)const 
    {
        //ISSUE check for range?
        return vals[i];
    }
    __DEVICE_TAG__  T &operator()(int i)
    {
        //ISSUE check for range?
        return vals[i];
    }
};

template<class T, int max_vals_n1, int max_vals_n2>
struct vals_table_2d
{
    int     vals_n1, vals_n2;
    T       vals[max_vals_n1][max_vals_n2];

    __DEVICE_TAG__  int get_vals_n1()const { return vals_n1; }
    __DEVICE_TAG__  int get_vals_n2()const { return vals_n2; }
    __DEVICE_TAG__  const T &operator()(int i1,int i2)const 
    {
        //ISSUE check for range?
        return vals[i1][i2];
    }
    __DEVICE_TAG__  T &operator()(int i1,int i2) 
    {
        //ISSUE check for range?
        return vals[i1][i2];
    }
};

template<class T, int max_vals_n1, int max_vals_n2, int max_vals_n3>
struct vals_table_3d
{
    int     vals_n1, vals_n2, vals_n3;
    T       vals[max_vals_n1][max_vals_n2][max_vals_n3];

    __DEVICE_TAG__  int get_vals_n1()const { return vals_n1; }
    __DEVICE_TAG__  int get_vals_n2()const { return vals_n2; }
    __DEVICE_TAG__  int get_vals_n3()const { return vals_n3; }
    __DEVICE_TAG__  const T &operator()(int i1,int i2,int i3)const 
    {
        //ISSUE check for range?
        return vals[i1][i2][i3];
    }
    __DEVICE_TAG__  T &operator()(int i1,int i2,int i3) 
    {
        //ISSUE check for range?
        return vals[i1][i2][i3];
    }
};

template<class T, int max_vals_n1, int max_vals_n2, int max_vals_n3, int max_vals_n4>
struct vals_table_4d
{
    int     vals_n1, vals_n2, vals_n3, vals_n4;
    T       vals[max_vals_n1][max_vals_n2][max_vals_n3][max_vals_n4];

    __DEVICE_TAG__  int get_vals_n1()const { return vals_n1; }
    __DEVICE_TAG__  int get_vals_n2()const { return vals_n2; }
    __DEVICE_TAG__  int get_vals_n3()const { return vals_n3; }
    __DEVICE_TAG__  int get_vals_n4()const { return vals_n4; }
    __DEVICE_TAG__  const T &operator()(int i1,int i2,int i3,int i4)const 
    {
        //ISSUE check for range?
        return vals[i1][i2][i3][i4];
    }
    __DEVICE_TAG__  T &operator()(int i1,int i2,int i3,int i4) 
    {
        //ISSUE check for range?
        return vals[i1][i2][i3][i4];
    }
};

}

#endif
