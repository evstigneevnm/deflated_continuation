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

#ifndef __SCFD_CREATE_SOLVER_H__
#define __SCFD_CREATE_SOLVER_H__

#include <string>
#include <boost/property_tree/ptree.hpp>
#include "create_monitor.h"
#include "solver_base.h"
#include "cgs.h"

namespace numerical_algos
{
namespace lin_solvers 
{

template<class LinearOperator,class Preconditioner,class VectorOperations,
         bool ManualVerifySize,class Log>
solver_base<LinearOperator,Preconditioner,VectorOperations,ManualVerifySize,default_monitor,Log>*
create_solver(const boost::property_tree::ptree &cfg, Log *log = NULL, int obj_log_lev = 0)
{
    std::string     lin_solver_type_name = cfg.get<std::string>("lin_solver_type_name"),

    if (lin_solver_type_name == "CGS") {
        return new cgs<LinearOperator,Preconditioner,VectorOperations,
                      ManualVerifySize,default_monitor,Log>(log, obj_log_lev);
    } else throw std::runtime_error("create_solver: unsupported lin_solver_type_name: " + 
                                    lin_solver_type_name);
    create_monitor(cfg.get_child("monitor"), log, obj_log_lev);
}


}
}

#endif
