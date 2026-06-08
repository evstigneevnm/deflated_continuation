#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

// problem dependant
#include <common/scalar_math.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/system_operator.h>
// problem dependant ends
#include <numerical_algos/lin_solvers/cgs.h>
#include <numerical_algos/lin_solvers/default_monitor.h>

// problem dependant
#include <common/gpu_file_operations.h>
#include "circle_backend_typedefs.h"
#if defined(CIRCLE_VECTOR_BACKEND_HIP)
#include <scfd/utils/init_hip.h>
#elif !defined(CIRCLE_VECTOR_BACKEND_OMP) && !defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
#include <scfd/utils/init_cuda.h>
#endif
// problem dependant ends

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>

namespace
{

void print_usage(const char* executable)
{
  printf("Usage: %s [path_to_config_file.json] [--analytical-solution] [--seed-exact-curve lambda sign] [--quiet]\n", executable);
}

template<class T>
T parse_scalar(const std::string& value, const char* label)
{
  std::istringstream stream(value);
  T result = T(0);
  stream >> result;
  if(!stream)
  {
    throw std::runtime_error(std::string("failed to parse ") + label + " from '" + value + "'");
  }
  return result;
}

bool backend_needs_device_init()
{
#if defined(CIRCLE_VECTOR_BACKEND_OMP) || defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
  return false;
#else
  return true;
#endif
}

int init_device_from_config(int pci_id)
{
#if defined(CIRCLE_VECTOR_BACKEND_HIP)
  return scfd::utils::init_hip(pci_id);
#elif defined(CIRCLE_VECTOR_BACKEND_OMP) || defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
  (void)pci_id;
  return -1;
#else
  return scfd::utils::init_cuda(pci_id);
#endif
}

} // namespace

int main(int argc, char const *argv[]) {
  size_t Nx = 1; // size of the vector variable. 1 in this case

  typedef gpu_file_operations<vec_ops_real> files_ops_t;
  typedef numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>
      monitor_t;

  typedef typename vec_ops_real::vector_type real_vec;
  typedef nonlinear_operators::circle<vec_ops_real, Blocks_x_> circle_t;

  typedef nonlinear_operators::linear_operator_circle<vec_ops_real, circle_t>
      lin_op_t;

  typedef nonlinear_operators::preconditioner_circle<vec_ops_real, circle_t,
                                                     lin_op_t>
      prec_t;

#if defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
  typedef double parameters_real;
  typedef main_classes::parameters<double> parameters_t;
#else
  typedef real parameters_real;
  typedef main_classes::parameters<real> parameters_t;
#endif
  std::string path_to_config_file = "json_project_files/circle_test.json";
  bool use_analytical_solution = false;
  bool seed_exact_curve = false;
  bool quiet = false;
  real seed_lambda = real(0);
  real seed_sign = real(1);
  for(int argi = 1; argi < argc; ++argi)
  {
    const std::string arg = argv[argi];
    if(arg == "--analytical-solution")
    {
      use_analytical_solution = true;
    }
    else if(arg == "--quiet")
    {
      quiet = true;
    }
    else if(arg == "--seed-exact-curve")
    {
      if(argi + 2 >= argc)
      {
        print_usage(argv[0]);
        return 1;
      }
      try
      {
        seed_lambda = parse_scalar<real>(argv[++argi], "seed lambda");
        seed_sign = parse_scalar<real>(argv[++argi], "seed sign");
      }
      catch(const std::exception& e)
      {
        std::cerr << e.what() << std::endl;
        return 2;
      }
      seed_exact_curve = true;
    }
    else if(path_to_config_file == "json_project_files/circle_test.json")
    {
      path_to_config_file = arg;
    }
    else
    {
      print_usage(argv[0]);
      return 1;
    }
  }
  std::cout << "Using circle backend: " << CIRCLE_BACKEND_NAME << std::endl;
  std::cout << "Analytical seed: " << (use_analytical_solution ? "enabled" : "disabled") << std::endl;
  if(seed_exact_curve)
  {
    std::cout << "Exact regular-continuation seed: lambda=" << seed_lambda
              << ", sign=" << seed_sign << std::endl;
  }
  std::cout << "Reading config file: " << path_to_config_file << std::endl;
  parameters_t parameters =
      main_classes::read_parameters_json<parameters_real>(path_to_config_file);
  if(!quiet)
  {
    parameters.plot_all();
  }

  Nx = parameters.nonlinear_operator.N_size.at(0) == 1
           ? Nx
           : (throw std::runtime_error("incorrect size for problem in config "
                                       "file provided. Expecting 1."));

  unsigned int m_Krylov = parameters.stability_continuation.Krylov_subspace;
  int nvidia_pci_id = parameters.nvidia_pci_id;
  bool use_high_precision_reduction = parameters.use_high_precision_reduction;

  if(backend_needs_device_init())
  {
    const int device = init_device_from_config(nvidia_pci_id);
    std::cout << "Using device " << device << std::endl;
  }
  real norm_wight = common::scalar_math::sqrt(real(Nx));
  (void)norm_wight;
  real Rad = 1.0;

  vec_ops_real vec_ops_R(Nx);
  if (use_high_precision_reduction) {
#if defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
    std::cerr << "Warning: variable-precision vector operations do not expose high-precision reduction toggles; using their native reductions.\n";
#else
    vec_ops_R.use_high_precision();
#endif
  }
  files_ops_t file_ops((vec_ops_real *)&vec_ops_R);

  circle_t CIRCLE(Rad, Nx, (vec_ops_real *)&vec_ops_R);
  log_t log;
  log_t log_linsolver;
  log.set_verbosity(quiet ? 0 : 1);
  log_linsolver.set_verbosity(quiet ? 0 : 1);

  // test continuaiton process of a single curve
  typedef main_classes::deflation_continuation<
      vec_ops_real, files_ops_t, log_t, monitor_t, circle_t, lin_op_t, prec_t,
      numerical_algos::lin_solvers::cgs,
      nonlinear_operators::system_operator, parameters_t>
      deflation_continuation_t;

  deflation_continuation_t DC((vec_ops_real *)&vec_ops_R,
                              (files_ops_t *)&file_ops, (log_t *)&log,
                              (log_t *)&log_linsolver, (circle_t *)&CIRCLE,
                              (parameters_t *)&parameters);

  DC.set_parameters();
  DC.use_analytical_solution(use_analytical_solution);

  if(seed_exact_curve)
  {
    const real radicand = Rad*Rad - seed_lambda*seed_lambda;
    if(radicand < real(0))
    {
      std::cerr << "Exact seed lambda is outside the circle: " << seed_lambda << std::endl;
      return 2;
    }
    real_vec x_seed;
    vec_ops_R.init_vector(x_seed);
    vec_ops_R.start_use_vector(x_seed);
    vec_ops_R.assign_scalar(seed_sign*common::scalar_math::sqrt(radicand), x_seed);
    DC.add_solution_curve(x_seed, seed_lambda);
    vec_ops_R.stop_use_vector(x_seed);
    vec_ops_R.free_vector(x_seed);
  }

  DC.execute();

  return 0;
}
