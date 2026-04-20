#include <algorithm>
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using State4 = std::array<double, 4>;
using Obs2 = std::array<double, 2>;
using Matrix = std::vector<double>;

struct SolverConfig {
  int horizon{50};
  double ts{0.1};
  double v_max{2.0};
  double a_max{1.5};
  int outer_iters{4};
  int inner_iters{6};
  double eps{1e-3};
  double lr{0.003};
  double lambda{100.0};
  double alpha{0.1};
  double beta_grow{1.1};
  double beta_shrink{0.5};
  double trust_pos0{1.0};
  double trust_u0{1.0};
  double min_clearance{4.0};
};

struct SolveResult {
  Matrix u_opt;
  Matrix x_opt;
  std::string solver_status;
};

using OSQPInt = int;
using OSQPFloat = double;

struct OSQPCscMatrix {
  OSQPInt m;
  OSQPInt n;
  OSQPInt * p;
  OSQPInt * i;
  OSQPFloat * x;
  OSQPInt nzmax;
  OSQPInt nz;
  OSQPInt owned;
};

enum osqp_linsys_solver_type {
  OSQP_UNKNOWN_SOLVER = 0,
  OSQP_DIRECT_SOLVER = 1,
  OSQP_INDIRECT_SOLVER = 2,
};

enum osqp_precond_type {
  OSQP_NO_PRECONDITIONER = 0,
  OSQP_DIAGONAL_PRECONDITIONER = 1,
};

struct OSQPSettings {
  OSQPInt device;
  osqp_linsys_solver_type linsys_solver;
  OSQPInt allocate_solution;
  OSQPInt verbose;
  OSQPInt profiler_level;
  OSQPInt warm_starting;
  OSQPInt scaling;
  OSQPInt polishing;
  OSQPFloat rho;
  OSQPInt rho_is_vec;
  OSQPFloat sigma;
  OSQPFloat alpha;
  OSQPInt cg_max_iter;
  OSQPInt cg_tol_reduction;
  OSQPFloat cg_tol_fraction;
  osqp_precond_type cg_precond;
  OSQPInt adaptive_rho;
  OSQPInt adaptive_rho_interval;
  OSQPFloat adaptive_rho_fraction;
  OSQPFloat adaptive_rho_tolerance;
  OSQPInt max_iter;
  OSQPFloat eps_abs;
  OSQPFloat eps_rel;
  OSQPFloat eps_prim_inf;
  OSQPFloat eps_dual_inf;
  OSQPInt scaled_termination;
  OSQPInt check_termination;
  OSQPInt check_dualgap;
  OSQPFloat time_limit;
  OSQPFloat delta;
  OSQPInt polish_refine_iter;
};

struct OSQPInfo {
  char status[32];
  OSQPInt status_val;
  OSQPInt status_polish;
  OSQPFloat obj_val;
  OSQPFloat dual_obj_val;
  OSQPFloat prim_res;
  OSQPFloat dual_res;
  OSQPFloat duality_gap;
  OSQPInt iter;
  OSQPInt rho_updates;
  OSQPFloat rho_estimate;
  OSQPFloat setup_time;
  OSQPFloat solve_time;
  OSQPFloat update_time;
  OSQPFloat polish_time;
  OSQPFloat run_time;
  OSQPFloat primdual_int;
  OSQPFloat rel_kkt_error;
};

struct OSQPSolution {
  OSQPFloat * x;
  OSQPFloat * y;
  OSQPFloat * prim_inf_cert;
  OSQPFloat * dual_inf_cert;
};

struct OSQPWorkspace_;

struct OSQPSolver {
  OSQPSettings * settings;
  OSQPSolution * solution;
  OSQPInfo * info;
  OSQPWorkspace_ * work;
};

constexpr OSQPInt kOsqpSolved = 1;
constexpr OSQPInt kOsqpSolvedInaccurate = 2;
constexpr double kOsqpInf = 1e30;

struct SparseMatrixData {
  std::vector<OSQPFloat> x;
  std::vector<OSQPInt> i;
  std::vector<OSQPInt> p;
};

struct Triplet {
  OSQPInt row;
  OSQPInt col;
  OSQPFloat value;
};

struct OsqpApi {
  using CscMatrixNewFn = OSQPCscMatrix * (*)(OSQPInt, OSQPInt, OSQPInt, OSQPFloat *, OSQPInt *, OSQPInt *);
  using CscMatrixFreeFn = void (*)(OSQPCscMatrix *);
  using SettingsNewFn = OSQPSettings * (*)();
  using SettingsFreeFn = void (*)(OSQPSettings *);
  using SetDefaultSettingsFn = void (*)(OSQPSettings *);
  using SetupFn = OSQPInt (*)(
    OSQPSolver **,
    const OSQPCscMatrix *,
    const OSQPFloat *,
    const OSQPCscMatrix *,
    const OSQPFloat *,
    const OSQPFloat *,
    OSQPInt,
    OSQPInt,
    const OSQPSettings *);
  using SolveFn = OSQPInt (*)(OSQPSolver *);
  using CleanupFn = OSQPInt (*)(OSQPSolver *);
  using WarmStartFn = OSQPInt (*)(OSQPSolver *, const OSQPFloat *, const OSQPFloat *);

  void * handle{nullptr};
  CscMatrixNewFn csc_matrix_new{nullptr};
  CscMatrixFreeFn csc_matrix_free{nullptr};
  SettingsNewFn settings_new{nullptr};
  SettingsFreeFn settings_free{nullptr};
  SetDefaultSettingsFn set_default_settings{nullptr};
  SetupFn setup{nullptr};
  SolveFn solve{nullptr};
  CleanupFn cleanup{nullptr};
  WarmStartFn warm_start{nullptr};
};

struct QpLayout {
  int horizon;
  bool has_obs;
  int u_base{0};
  int term_base;
  int slack_u_base;
  int slack_v_base;
  int slack_tr_u_base;
  int slack_tr_x_base;
  int slack_obs_base;
  int num_vars;

  explicit QpLayout(int horizon_in, bool has_obs_in)
  : horizon(horizon_in), has_obs(has_obs_in),
    term_base(2 * horizon_in),
    slack_u_base(term_base + 4),
    slack_v_base(slack_u_base + 2 * horizon_in),
    slack_tr_u_base(slack_v_base + 2 * horizon_in),
    slack_tr_x_base(slack_tr_u_base + 2 * horizon_in),
    slack_obs_base(slack_tr_x_base + 2 * horizon_in),
    num_vars(slack_obs_base + (has_obs_in ? horizon_in : 0)) {}

  int u(int k, int dim) const {
    return u_base + 2 * k + dim;
  }

  int term_pos(int dim) const {
    return term_base + dim;
  }

  int term_vel(int dim) const {
    return term_base + 2 + dim;
  }

  int slack_u(int k, int dim) const {
    return slack_u_base + 2 * k + dim;
  }

  int slack_v(int k, int dim) const {
    return slack_v_base + 2 * (k - 1) + dim;
  }

  int slack_tr_u(int k, int dim) const {
    return slack_tr_u_base + 2 * k + dim;
  }

  int slack_tr_x(int k, int dim) const {
    return slack_tr_x_base + 2 * (k - 1) + dim;
  }

  int slack_obs(int k) const {
    return slack_obs_base + (k - 1);
  }
};

const OsqpApi & get_osqp_api() {
  static std::once_flag once;
  static std::unique_ptr<OsqpApi> api;
  static std::string error;

  std::call_once(once, []() {
    try {
      py::gil_scoped_acquire acquire;
      const std::string lib_path = py::str(py::module_::import("osqp.ext_builtin").attr("__file__"));
      auto loaded = std::make_unique<OsqpApi>();
      loaded->handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (loaded->handle == nullptr) {
        error = std::string("dlopen failed for osqp.ext_builtin: ") + dlerror();
        return;
      }

      auto require_symbol = [&](auto & fn, const char * name) {
        fn = reinterpret_cast<std::decay_t<decltype(fn)>>(dlsym(loaded->handle, name));
        if (fn == nullptr) {
          throw std::runtime_error(std::string("Missing OSQP symbol: ") + name);
        }
      };

      require_symbol(loaded->csc_matrix_new, "OSQPCscMatrix_new");
      require_symbol(loaded->csc_matrix_free, "OSQPCscMatrix_free");
      require_symbol(loaded->settings_new, "OSQPSettings_new");
      require_symbol(loaded->settings_free, "OSQPSettings_free");
      require_symbol(loaded->set_default_settings, "osqp_set_default_settings");
      require_symbol(loaded->setup, "osqp_setup");
      require_symbol(loaded->solve, "osqp_solve");
      require_symbol(loaded->cleanup, "osqp_cleanup");
      loaded->warm_start = reinterpret_cast<OsqpApi::WarmStartFn>(dlsym(loaded->handle, "osqp_warm_start"));

      api = std::move(loaded);
    } catch (const std::exception & exc) {
      error = exc.what();
    }
  });

  if (!api) {
    throw std::runtime_error(error.empty() ? "Native OSQP API is unavailable" : error);
  }
  return *api;
}

inline double & u_at(Matrix & u, int k, int j) {
  return u[static_cast<std::size_t>(k) * 2U + static_cast<std::size_t>(j)];
}

inline double u_at(const Matrix & u, int k, int j) {
  return u[static_cast<std::size_t>(k) * 2U + static_cast<std::size_t>(j)];
}

inline double & x_at(Matrix & x, int k, int j) {
  return x[static_cast<std::size_t>(k) * 4U + static_cast<std::size_t>(j)];
}

inline double x_at(const Matrix & x, int k, int j) {
  return x[static_cast<std::size_t>(k) * 4U + static_cast<std::size_t>(j)];
}

double read_required_double(const py::dict & config, const char * key) {
  if (!config.contains(py::str(key))) {
    throw std::invalid_argument(std::string("Missing solver config key: ") + key);
  }
  return py::cast<double>(config[py::str(key)]);
}

int read_required_int(const py::dict & config, const char * key) {
  if (!config.contains(py::str(key))) {
    throw std::invalid_argument(std::string("Missing solver config key: ") + key);
  }
  return py::cast<int>(config[py::str(key)]);
}

SolverConfig parse_config(const py::dict & config) {
  SolverConfig cfg;
  cfg.horizon = read_required_int(config, "T_horizon");
  cfg.ts = read_required_double(config, "Ts");
  cfg.v_max = read_required_double(config, "V_MAX");
  cfg.a_max = read_required_double(config, "A_MAX");
  cfg.outer_iters = read_required_int(config, "SCP_MAX_OUTER_ITERS");
  cfg.inner_iters = read_required_int(config, "SCP_INNER_ITERS");
  cfg.eps = read_required_double(config, "SCP_EPS");
  cfg.lr = read_required_double(config, "SCP_LR");
  cfg.lambda = read_required_double(config, "SCP_LAMBDA");
  cfg.alpha = read_required_double(config, "SCP_ALPHA");
  cfg.beta_grow = read_required_double(config, "SCP_BETA_GROW");
  cfg.beta_shrink = read_required_double(config, "SCP_BETA_SHRINK");
  cfg.trust_pos0 = read_required_double(config, "TRUST_POS0");
  cfg.trust_u0 = read_required_double(config, "TRUST_U0");
  cfg.min_clearance = read_required_double(config, "MIN_CLEARANCE");
  return cfg;
}

State4 load_state4(const py::array_t<double, py::array::c_style | py::array::forcecast> & arr) {
  if (arr.ndim() != 1 || arr.shape(0) != 4) {
    throw std::invalid_argument("Expected shape (4,) for state vector");
  }
  auto buf = arr.unchecked<1>();
  return {buf(0), buf(1), buf(2), buf(3)};
}

std::optional<Obs2> load_obs2(const py::object & obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  auto arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(obj);
  if (arr.ndim() != 1 || arr.shape(0) != 2) {
    throw std::invalid_argument("Expected shape (2,) for nearest obstacle point");
  }
  auto buf = arr.unchecked<1>();
  return Obs2{buf(0), buf(1)};
}

Matrix load_u_matrix(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & arr,
  int horizon)
{
  if (arr.ndim() != 2 || arr.shape(0) != horizon || arr.shape(1) != 2) {
    throw std::invalid_argument("Expected shape (T_horizon, 2) for warm-start control");
  }
  Matrix out(static_cast<std::size_t>(horizon) * 2U, 0.0);
  auto buf = arr.unchecked<2>();
  for (int k = 0; k < horizon; ++k) {
    out[static_cast<std::size_t>(k) * 2U] = buf(k, 0);
    out[static_cast<std::size_t>(k) * 2U + 1U] = buf(k, 1);
  }
  return out;
}

py::array_t<double> to_numpy_matrix(const Matrix & data, py::ssize_t rows, py::ssize_t cols) {
  py::array_t<double> out({rows, cols});
  auto buf = out.mutable_unchecked<2>();
  for (py::ssize_t r = 0; r < rows; ++r) {
    for (py::ssize_t c = 0; c < cols; ++c) {
      buf(r, c) = data[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
        static_cast<std::size_t>(c)];
    }
  }
  return out;
}

Matrix rollout(const State4 & x0, const Matrix & u, const SolverConfig & cfg) {
  Matrix x(static_cast<std::size_t>(cfg.horizon + 1) * 4U, 0.0);
  for (int j = 0; j < 4; ++j) {
    x_at(x, 0, j) = x0[static_cast<std::size_t>(j)];
  }
  for (int k = 0; k < cfg.horizon; ++k) {
    x_at(x, k + 1, 0) = x_at(x, k, 0) + cfg.ts * x_at(x, k, 2);
    x_at(x, k + 1, 1) = x_at(x, k, 1) + cfg.ts * x_at(x, k, 3);
    x_at(x, k + 1, 2) = x_at(x, k, 2) + cfg.ts * u_at(u, k, 0);
    x_at(x, k + 1, 3) = x_at(x, k, 3) + cfg.ts * u_at(u, k, 1);
  }
  return x;
}

void project_speed_limits(Matrix & u, const SolverConfig & cfg) {
  for (double & value : u) {
    value = std::clamp(value, -cfg.a_max, cfg.a_max);
  }
}

double hinge(double value) {
  return std::max(0.0, value);
}

double sum_abs2(double a, double b) {
  return std::abs(a) + std::abs(b);
}

double velocity_base(const State4 & x0, int dim) {
  return x0[static_cast<std::size_t>(2 + dim)];
}

double velocity_coeff(int k, int j, const SolverConfig & cfg) {
  return (j < k) ? cfg.ts : 0.0;
}

double position_base(const State4 & x0, int dim, int k, const SolverConfig & cfg) {
  return x0[static_cast<std::size_t>(dim)] + cfg.ts * static_cast<double>(k) * x0[static_cast<std::size_t>(2 + dim)];
}

double position_coeff(int k, int j, const SolverConfig & cfg) {
  return (j < k - 1) ? (cfg.ts * cfg.ts * static_cast<double>(k - 1 - j)) : 0.0;
}

SparseMatrixData diagonal_quadratic_matrix(int num_vars, int num_diag, double diag_value) {
  SparseMatrixData data;
  data.x.reserve(static_cast<std::size_t>(num_diag));
  data.i.reserve(static_cast<std::size_t>(num_diag));
  data.p.resize(static_cast<std::size_t>(num_vars + 1), num_diag);
  for (int col = 0; col < num_vars; ++col) {
    if (col < num_diag) {
      data.p[static_cast<std::size_t>(col)] = col;
      data.i.push_back(col);
      data.x.push_back(diag_value);
    } else {
      data.p[static_cast<std::size_t>(col)] = num_diag;
    }
  }
  data.p[static_cast<std::size_t>(num_vars)] = num_diag;
  return data;
}

SparseMatrixData triplets_to_csc(int num_rows, int num_cols, std::vector<Triplet> triplets) {
  (void)num_rows;
  std::sort(
    triplets.begin(),
    triplets.end(),
    [](const Triplet & lhs, const Triplet & rhs) {
      if (lhs.col != rhs.col) {
        return lhs.col < rhs.col;
      }
      return lhs.row < rhs.row;
    });

  SparseMatrixData data;
  data.x.reserve(triplets.size());
  data.i.reserve(triplets.size());
  data.p.assign(static_cast<std::size_t>(num_cols + 1), 0);
  for (const Triplet & entry : triplets) {
    data.p[static_cast<std::size_t>(entry.col + 1)] += 1;
  }
  for (int col = 0; col < num_cols; ++col) {
    data.p[static_cast<std::size_t>(col + 1)] += data.p[static_cast<std::size_t>(col)];
  }
  for (const Triplet & entry : triplets) {
    data.i.push_back(entry.row);
    data.x.push_back(entry.value);
  }
  return data;
}

std::optional<Matrix> solve_convex_subproblem_osqp(
  const State4 & x0,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  const Matrix & x_now,
  const Matrix & u_now,
  double l_pos,
  double l_u,
  const SolverConfig & cfg,
  std::string * status_out)
{
  const OsqpApi & osqp = get_osqp_api();
  const QpLayout layout(cfg.horizon, x_obs.has_value());

  std::vector<double> q(static_cast<std::size_t>(layout.num_vars), 0.0);
  for (int dim = 0; dim < 2; ++dim) {
    q[static_cast<std::size_t>(layout.term_pos(dim))] = cfg.lambda;
    q[static_cast<std::size_t>(layout.term_vel(dim))] = cfg.lambda;
  }
  for (int k = 0; k < cfg.horizon; ++k) {
    for (int dim = 0; dim < 2; ++dim) {
      q[static_cast<std::size_t>(layout.slack_u(k, dim))] = cfg.lambda;
      q[static_cast<std::size_t>(layout.slack_tr_u(k, dim))] = cfg.lambda;
    }
  }
  for (int k = 1; k <= cfg.horizon; ++k) {
    for (int dim = 0; dim < 2; ++dim) {
      q[static_cast<std::size_t>(layout.slack_v(k, dim))] = cfg.lambda;
      q[static_cast<std::size_t>(layout.slack_tr_x(k, dim))] = cfg.lambda;
    }
    if (x_obs.has_value()) {
      q[static_cast<std::size_t>(layout.slack_obs(k))] = cfg.lambda;
    }
  }

  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(cfg.horizon * 80));
  auto add_coeff = [&](int row, int col, double value) {
    if (std::abs(value) <= 1e-12) {
      return;
    }
    triplets.push_back(Triplet{row, col, value});
  };
  auto add_row = [&](double l, double u) {
    lower.push_back(l);
    upper.push_back(u);
    return static_cast<int>(lower.size() - 1);
  };

  for (int dim = 0; dim < 2; ++dim) {
    {
      const int row = add_row(-kOsqpInf, x_ref[static_cast<std::size_t>(dim)] - position_base(x0, dim, cfg.horizon, cfg));
      for (int j = 0; j < cfg.horizon; ++j) {
        add_coeff(row, layout.u(j, dim), position_coeff(cfg.horizon, j, cfg));
      }
      add_coeff(row, layout.term_pos(dim), -1.0);
    }
    {
      const int row = add_row(-kOsqpInf, position_base(x0, dim, cfg.horizon, cfg) - x_ref[static_cast<std::size_t>(dim)]);
      for (int j = 0; j < cfg.horizon; ++j) {
        add_coeff(row, layout.u(j, dim), -position_coeff(cfg.horizon, j, cfg));
      }
      add_coeff(row, layout.term_pos(dim), -1.0);
    }
    {
      const int row = add_row(-kOsqpInf, x_ref[static_cast<std::size_t>(2 + dim)] - velocity_base(x0, dim));
      for (int j = 0; j < cfg.horizon; ++j) {
        add_coeff(row, layout.u(j, dim), velocity_coeff(cfg.horizon, j, cfg));
      }
      add_coeff(row, layout.term_vel(dim), -1.0);
    }
    {
      const int row = add_row(-kOsqpInf, velocity_base(x0, dim) - x_ref[static_cast<std::size_t>(2 + dim)]);
      for (int j = 0; j < cfg.horizon; ++j) {
        add_coeff(row, layout.u(j, dim), -velocity_coeff(cfg.horizon, j, cfg));
      }
      add_coeff(row, layout.term_vel(dim), -1.0);
    }
  }

  for (int k = 0; k < cfg.horizon; ++k) {
    for (int dim = 0; dim < 2; ++dim) {
      {
        const int row = add_row(-kOsqpInf, cfg.a_max);
        add_coeff(row, layout.u(k, dim), 1.0);
        add_coeff(row, layout.slack_u(k, dim), -1.0);
      }
      {
        const int row = add_row(-kOsqpInf, cfg.a_max);
        add_coeff(row, layout.u(k, dim), -1.0);
        add_coeff(row, layout.slack_u(k, dim), -1.0);
      }
      {
        const int row = add_row(0.0, kOsqpInf);
        add_coeff(row, layout.slack_u(k, dim), 1.0);
      }
      {
        const int row = add_row(-kOsqpInf, l_u + u_at(u_now, k, dim));
        add_coeff(row, layout.u(k, dim), 1.0);
        add_coeff(row, layout.slack_tr_u(k, dim), -1.0);
      }
      {
        const int row = add_row(-kOsqpInf, l_u - u_at(u_now, k, dim));
        add_coeff(row, layout.u(k, dim), -1.0);
        add_coeff(row, layout.slack_tr_u(k, dim), -1.0);
      }
      {
        const int row = add_row(0.0, kOsqpInf);
        add_coeff(row, layout.slack_tr_u(k, dim), 1.0);
      }
    }
  }

  for (int k = 1; k <= cfg.horizon; ++k) {
    for (int dim = 0; dim < 2; ++dim) {
      {
        const int row = add_row(-kOsqpInf, cfg.v_max - velocity_base(x0, dim));
        for (int j = 0; j < cfg.horizon; ++j) {
          add_coeff(row, layout.u(j, dim), velocity_coeff(k, j, cfg));
        }
        add_coeff(row, layout.slack_v(k, dim), -1.0);
      }
      {
        const int row = add_row(-kOsqpInf, cfg.v_max + velocity_base(x0, dim));
        for (int j = 0; j < cfg.horizon; ++j) {
          add_coeff(row, layout.u(j, dim), -velocity_coeff(k, j, cfg));
        }
        add_coeff(row, layout.slack_v(k, dim), -1.0);
      }
      {
        const int row = add_row(0.0, kOsqpInf);
        add_coeff(row, layout.slack_v(k, dim), 1.0);
      }
      {
        const int row = add_row(-kOsqpInf, l_pos + x_at(x_now, k, dim) - position_base(x0, dim, k, cfg));
        for (int j = 0; j < cfg.horizon; ++j) {
          add_coeff(row, layout.u(j, dim), position_coeff(k, j, cfg));
        }
        add_coeff(row, layout.slack_tr_x(k, dim), -1.0);
      }
      {
        const int row = add_row(-kOsqpInf, l_pos - x_at(x_now, k, dim) + position_base(x0, dim, k, cfg));
        for (int j = 0; j < cfg.horizon; ++j) {
          add_coeff(row, layout.u(j, dim), -position_coeff(k, j, cfg));
        }
        add_coeff(row, layout.slack_tr_x(k, dim), -1.0);
      }
      {
        const int row = add_row(0.0, kOsqpInf);
        add_coeff(row, layout.slack_tr_x(k, dim), 1.0);
      }
    }
  }

  if (x_obs.has_value()) {
    for (int k = 1; k <= cfg.horizon; ++k) {
      const double d_now_x = x_at(x_now, k, 0) - x_obs->at(0);
      const double d_now_y = x_at(x_now, k, 1) - x_obs->at(1);
      const double d_now_norm = std::sqrt(d_now_x * d_now_x + d_now_y * d_now_y);
      const double base_x = position_base(x0, 0, k, cfg);
      const double base_y = position_base(x0, 1, k, cfg);
      const double constant = cfg.min_clearance * d_now_norm
        - (d_now_x * (base_x - x_obs->at(0)) + d_now_y * (base_y - x_obs->at(1)));

      const int row = add_row(-kOsqpInf, -constant);
      for (int j = 0; j < cfg.horizon; ++j) {
        add_coeff(row, layout.u(j, 0), -d_now_x * position_coeff(k, j, cfg));
        add_coeff(row, layout.u(j, 1), -d_now_y * position_coeff(k, j, cfg));
      }
      add_coeff(row, layout.slack_obs(k), -1.0);

      const int nonneg_row = add_row(0.0, kOsqpInf);
      add_coeff(nonneg_row, layout.slack_obs(k), 1.0);
    }
  }

  const SparseMatrixData P = diagonal_quadratic_matrix(layout.num_vars, 2 * cfg.horizon, 2.0 * cfg.ts);
  const SparseMatrixData A = triplets_to_csc(static_cast<int>(lower.size()), layout.num_vars, std::move(triplets));

  std::vector<OSQPFloat> q_osqp(q.begin(), q.end());
  std::vector<OSQPFloat> l_osqp(lower.begin(), lower.end());
  std::vector<OSQPFloat> u_osqp(upper.begin(), upper.end());

  OSQPCscMatrix * P_mat = osqp.csc_matrix_new(
    layout.num_vars,
    layout.num_vars,
    static_cast<OSQPInt>(P.x.size()),
    const_cast<OSQPFloat *>(P.x.data()),
    const_cast<OSQPInt *>(P.i.data()),
    const_cast<OSQPInt *>(P.p.data()));
  OSQPCscMatrix * A_mat = osqp.csc_matrix_new(
    static_cast<OSQPInt>(lower.size()),
    layout.num_vars,
    static_cast<OSQPInt>(A.x.size()),
    const_cast<OSQPFloat *>(A.x.data()),
    const_cast<OSQPInt *>(A.i.data()),
    const_cast<OSQPInt *>(A.p.data()));

  if (P_mat == nullptr || A_mat == nullptr) {
    if (P_mat != nullptr) {
      osqp.csc_matrix_free(P_mat);
    }
    if (A_mat != nullptr) {
      osqp.csc_matrix_free(A_mat);
    }
    if (status_out != nullptr) {
      *status_out = "cpp_osqp_matrix_alloc_failed";
    }
    return std::nullopt;
  }

  OSQPSettings * settings = osqp.settings_new();
  if (settings == nullptr) {
    osqp.csc_matrix_free(P_mat);
    osqp.csc_matrix_free(A_mat);
    if (status_out != nullptr) {
      *status_out = "cpp_osqp_settings_alloc_failed";
    }
    return std::nullopt;
  }

  osqp.set_default_settings(settings);
  settings->verbose = 0;
  settings->warm_starting = 1;
  settings->polishing = 0;
  settings->max_iter = 20000;
  settings->eps_abs = 1e-5;
  settings->eps_rel = 1e-5;
  settings->check_termination = 25;

  OSQPSolver * solver = nullptr;
  const OSQPInt setup_status = osqp.setup(
    &solver,
    P_mat,
    q_osqp.data(),
    A_mat,
    l_osqp.data(),
    u_osqp.data(),
    static_cast<OSQPInt>(lower.size()),
    layout.num_vars,
    settings);
  if (setup_status != 0 || solver == nullptr) {
    osqp.settings_free(settings);
    osqp.csc_matrix_free(P_mat);
    osqp.csc_matrix_free(A_mat);
    if (status_out != nullptr) {
      *status_out = "cpp_osqp_setup_failed";
    }
    return std::nullopt;
  }

  std::vector<OSQPFloat> warm_start(static_cast<std::size_t>(layout.num_vars), 0.0);
  for (int k = 0; k < cfg.horizon; ++k) {
    warm_start[static_cast<std::size_t>(layout.u(k, 0))] = u_at(u_now, k, 0);
    warm_start[static_cast<std::size_t>(layout.u(k, 1))] = u_at(u_now, k, 1);
  }
  if (osqp.warm_start != nullptr) {
    osqp.warm_start(solver, warm_start.data(), nullptr);
  }

  const OSQPInt solve_status = osqp.solve(solver);
  std::string status = "cpp_osqp";
  if (solver->info != nullptr) {
    status = solver->info->status;
  } else if (solve_status != 0) {
    status = "cpp_osqp_solve_failed";
  }

  std::optional<Matrix> result;
  if (solve_status == 0 && solver->info != nullptr &&
      (solver->info->status_val == kOsqpSolved || solver->info->status_val == kOsqpSolvedInaccurate) &&
      solver->solution != nullptr && solver->solution->x != nullptr) {
    Matrix u_candidate(static_cast<std::size_t>(cfg.horizon) * 2U, 0.0);
    for (int k = 0; k < cfg.horizon; ++k) {
      u_at(u_candidate, k, 0) = solver->solution->x[layout.u(k, 0)];
      u_at(u_candidate, k, 1) = solver->solution->x[layout.u(k, 1)];
    }
    project_speed_limits(u_candidate, cfg);
    result = std::move(u_candidate);
  }

  if (status_out != nullptr) {
    *status_out = status;
  }

  osqp.cleanup(solver);
  osqp.settings_free(settings);
  osqp.csc_matrix_free(P_mat);
  osqp.csc_matrix_free(A_mat);

  return result;
}

double phi_real(
  const Matrix & x,
  const Matrix & u,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  const SolverConfig & cfg)
{
  double eqns = 0.0;
  eqns += sum_abs2(x_at(x, cfg.horizon, 0) - x_ref[0], x_at(x, cfg.horizon, 1) - x_ref[1]);
  eqns += sum_abs2(x_at(x, cfg.horizon, 2) - x_ref[2], x_at(x, cfg.horizon, 3) - x_ref[3]);

  double ineq_u = 0.0;
  for (int k = 0; k < cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      ineq_u += hinge(std::abs(u_at(u, k, j)) - cfg.a_max);
    }
  }

  double ineq_v = 0.0;
  for (int k = 1; k <= cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      ineq_v += hinge(std::abs(x_at(x, k, j + 2)) - cfg.v_max);
    }
  }

  double ineq_obs = 0.0;
  if (x_obs.has_value()) {
    for (int k = 1; k <= cfg.horizon; ++k) {
      const double dx = x_at(x, k, 0) - x_obs->at(0);
      const double dy = x_at(x, k, 1) - x_obs->at(1);
      const double dist = std::sqrt(dx * dx + dy * dy);
      ineq_obs += hinge(cfg.min_clearance - dist);
    }
  }

  double obj = 0.0;
  for (double value : u) {
    obj += value * value;
  }
  obj *= cfg.ts;
  return obj + cfg.lambda * (eqns + ineq_u + ineq_v + ineq_obs);
}

double phi_hat(
  const Matrix & x,
  const Matrix & u,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  const Matrix & x_now,
  const Matrix & u_now,
  double l_pos,
  double l_u,
  const SolverConfig & cfg)
{
  double eqns = 0.0;
  eqns += sum_abs2(x_at(x, cfg.horizon, 0) - x_ref[0], x_at(x, cfg.horizon, 1) - x_ref[1]);
  eqns += sum_abs2(x_at(x, cfg.horizon, 2) - x_ref[2], x_at(x, cfg.horizon, 3) - x_ref[3]);

  double ineq_u = 0.0;
  double ineq_v = 0.0;
  double ineq_tr_u = 0.0;
  double ineq_tr_x = 0.0;
  for (int k = 0; k < cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      ineq_u += hinge(std::abs(u_at(u, k, j)) - cfg.a_max);
      ineq_tr_u += hinge(std::abs(u_at(u, k, j) - u_at(u_now, k, j)) - l_u);
    }
  }
  for (int k = 1; k <= cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      ineq_v += hinge(std::abs(x_at(x, k, j + 2)) - cfg.v_max);
      ineq_tr_x += hinge(std::abs(x_at(x, k, j) - x_at(x_now, k, j)) - l_pos);
    }
  }

  double ineq_obs = 0.0;
  if (x_obs.has_value()) {
    for (int k = 1; k <= cfg.horizon; ++k) {
      const double d_now_x = x_at(x_now, k, 0) - x_obs->at(0);
      const double d_now_y = x_at(x_now, k, 1) - x_obs->at(1);
      const double d_now_norm = std::sqrt(d_now_x * d_now_x + d_now_y * d_now_y);
      const double d_x = x_at(x, k, 0) - x_obs->at(0);
      const double d_y = x_at(x, k, 1) - x_obs->at(1);
      const double lin_obs = cfg.min_clearance * d_now_norm - (d_now_x * d_x + d_now_y * d_y);
      ineq_obs += hinge(lin_obs);
    }
  }

  double obj = 0.0;
  for (double value : u) {
    obj += value * value;
  }
  obj *= cfg.ts;
  return obj + cfg.lambda * (eqns + ineq_u + ineq_v + ineq_tr_u + ineq_tr_x + ineq_obs);
}

std::pair<Matrix, Matrix> phi_hat_gradient(
  const State4 & x0,
  const Matrix & u,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  const Matrix & x_now,
  const Matrix & u_now,
  double l_pos,
  double l_u,
  const SolverConfig & cfg)
{
  Matrix x = rollout(x0, u, cfg);
  Matrix grad(static_cast<std::size_t>(cfg.horizon) * 2U, 0.0);
  for (int k = 0; k < cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      u_at(grad, k, j) = 2.0 * cfg.ts * u_at(u, k, j);
    }
  }

  const double sign_pos_x = (x_at(x, cfg.horizon, 0) > x_ref[0]) - (x_at(x, cfg.horizon, 0) < x_ref[0]);
  const double sign_pos_y = (x_at(x, cfg.horizon, 1) > x_ref[1]) - (x_at(x, cfg.horizon, 1) < x_ref[1]);
  for (int j = 0; j < cfg.horizon - 1; ++j) {
    const double weight = cfg.ts * cfg.ts * static_cast<double>(cfg.horizon - 1 - j);
    u_at(grad, j, 0) += cfg.lambda * weight * sign_pos_x;
    u_at(grad, j, 1) += cfg.lambda * weight * sign_pos_y;
  }

  const double sign_vel_x = (x_at(x, cfg.horizon, 2) > x_ref[2]) - (x_at(x, cfg.horizon, 2) < x_ref[2]);
  const double sign_vel_y = (x_at(x, cfg.horizon, 3) > x_ref[3]) - (x_at(x, cfg.horizon, 3) < x_ref[3]);
  for (int j = 0; j < cfg.horizon; ++j) {
    u_at(grad, j, 0) += cfg.lambda * cfg.ts * sign_vel_x;
    u_at(grad, j, 1) += cfg.lambda * cfg.ts * sign_vel_y;
  }

  for (int k = 0; k < cfg.horizon; ++k) {
    for (int j = 0; j < 2; ++j) {
      const double u_val = u_at(u, k, j);
      if (std::abs(u_val) - cfg.a_max > 0.0) {
        u_at(grad, k, j) += cfg.lambda * ((u_val > 0.0) - (u_val < 0.0));
      }
      const double du = u_val - u_at(u_now, k, j);
      if (std::abs(du) - l_u > 0.0) {
        u_at(grad, k, j) += cfg.lambda * ((du > 0.0) - (du < 0.0));
      }
    }
  }

  for (int k = 1; k <= cfg.horizon; ++k) {
    double g_v_x = 0.0;
    double g_v_y = 0.0;
    if (std::abs(x_at(x, k, 2)) - cfg.v_max > 0.0) {
      g_v_x = (x_at(x, k, 2) > 0.0) - (x_at(x, k, 2) < 0.0);
    }
    if (std::abs(x_at(x, k, 3)) - cfg.v_max > 0.0) {
      g_v_y = (x_at(x, k, 3) > 0.0) - (x_at(x, k, 3) < 0.0);
    }
    if (g_v_x != 0.0 || g_v_y != 0.0) {
      for (int j = 0; j < k; ++j) {
        u_at(grad, j, 0) += cfg.lambda * cfg.ts * g_v_x;
        u_at(grad, j, 1) += cfg.lambda * cfg.ts * g_v_y;
      }
    }
  }

  for (int k = 1; k <= cfg.horizon; ++k) {
    double g_x = 0.0;
    double g_y = 0.0;
    const double dx = x_at(x, k, 0) - x_at(x_now, k, 0);
    const double dy = x_at(x, k, 1) - x_at(x_now, k, 1);
    if (std::abs(dx) - l_pos > 0.0) {
      g_x = (dx > 0.0) - (dx < 0.0);
    }
    if (std::abs(dy) - l_pos > 0.0) {
      g_y = (dy > 0.0) - (dy < 0.0);
    }
    if (g_x == 0.0 && g_y == 0.0) {
      continue;
    }
    for (int j = 0; j < k - 1; ++j) {
      const double weight = cfg.ts * cfg.ts * static_cast<double>(k - 1 - j);
      u_at(grad, j, 0) += cfg.lambda * weight * g_x;
      u_at(grad, j, 1) += cfg.lambda * weight * g_y;
    }
  }

  if (x_obs.has_value()) {
    for (int k = 0; k < cfg.horizon; ++k) {
      const double d_now_x = x_at(x_now, k + 1, 0) - x_obs->at(0);
      const double d_now_y = x_at(x_now, k + 1, 1) - x_obs->at(1);
      const double d_now_norm = std::sqrt(d_now_x * d_now_x + d_now_y * d_now_y);
      const double d_x = x_at(x, k + 1, 0) - x_obs->at(0);
      const double d_y = x_at(x, k + 1, 1) - x_obs->at(1);
      const double lin_obs = cfg.min_clearance * d_now_norm - (d_now_x * d_x + d_now_y * d_y);
      if (lin_obs <= 0.0) {
        continue;
      }
      const double g_x = -d_now_x;
      const double g_y = -d_now_y;
      for (int j = 0; j < k; ++j) {
        const double weight = cfg.ts * cfg.ts * static_cast<double>(k - j);
        u_at(grad, j, 0) += cfg.lambda * weight * g_x;
        u_at(grad, j, 1) += cfg.lambda * weight * g_y;
      }
    }
  }

  return {std::move(grad), std::move(x)};
}

SolveResult solve_subgradient_impl(
  const State4 & x0,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  Matrix u_warm,
  const SolverConfig & cfg)
{
  project_speed_limits(u_warm, cfg);
  Matrix u_now = std::move(u_warm);
  Matrix x_now = rollout(x0, u_now, cfg);

  double l_pos = cfg.trust_pos0;
  double l_u = cfg.trust_u0;
  std::string status = "cpp_subgradient";

  for (int outer = 0; outer < cfg.outer_iters; ++outer) {
    Matrix u_candidate = u_now;
    for (int inner = 0; inner < cfg.inner_iters; ++inner) {
      auto grad_and_x = phi_hat_gradient(x0, u_candidate, x_ref, x_obs, x_now, u_now, l_pos, l_u, cfg);
      Matrix & grad = grad_and_x.first;
      for (int k = 0; k < cfg.horizon; ++k) {
        for (int j = 0; j < 2; ++j) {
          u_at(u_candidate, k, j) -= cfg.lr * u_at(grad, k, j);
        }
      }
      project_speed_limits(u_candidate, cfg);
    }

    Matrix x_candidate = rollout(x0, u_candidate, cfg);
    const double phi_real_now = phi_real(x_now, u_now, x_ref, x_obs, cfg);
    const double phi_hat_new = phi_hat(x_candidate, u_candidate, x_ref, x_obs, x_now, u_now, l_pos, l_u, cfg);
    const double phi_real_new = phi_real(x_candidate, u_candidate, x_ref, x_obs, cfg);
    const double delta_hat = phi_real_now - phi_hat_new;
    const double delta = phi_real_now - phi_real_new;

    if (delta > cfg.alpha * delta_hat) {
      l_pos *= cfg.beta_grow;
      l_u *= cfg.beta_grow;
      double step_norm = 0.0;
      for (std::size_t idx = 0; idx < x_candidate.size(); ++idx) {
        step_norm = std::max(step_norm, std::abs(x_candidate[idx] - x_now[idx]));
      }
      x_now = std::move(x_candidate);
      u_now = std::move(u_candidate);
      if (step_norm < cfg.eps) {
        status = "cpp_subgradient_converged";
        break;
      }
    } else {
      l_pos *= cfg.beta_shrink;
      l_u *= cfg.beta_shrink;
    }
  }

  return {std::move(u_now), std::move(x_now), std::move(status)};
}

SolveResult solve_osqp_impl(
  const State4 & x0,
  const State4 & x_ref,
  const std::optional<Obs2> & x_obs,
  Matrix u_warm,
  const SolverConfig & cfg)
{
  project_speed_limits(u_warm, cfg);
  Matrix u_now = std::move(u_warm);
  Matrix x_now = rollout(x0, u_now, cfg);

  double l_pos = cfg.trust_pos0;
  double l_u = cfg.trust_u0;
  std::string status = "cpp_osqp";

  for (int outer = 0; outer < cfg.outer_iters; ++outer) {
    std::string subproblem_status;
    std::optional<Matrix> candidate = solve_convex_subproblem_osqp(
      x0, x_ref, x_obs, x_now, u_now, l_pos, l_u, cfg, &subproblem_status);

    if (!candidate.has_value()) {
      SolveResult fallback = solve_subgradient_impl(x0, x_ref, x_obs, std::move(u_now), cfg);
      fallback.solver_status = "cpp_osqp_fallback:" + subproblem_status + "|" + fallback.solver_status;
      return fallback;
    }

    Matrix x_candidate = rollout(x0, *candidate, cfg);
    const double phi_real_now = phi_real(x_now, u_now, x_ref, x_obs, cfg);
    const double phi_hat_new = phi_hat(
      x_candidate, *candidate, x_ref, x_obs, x_now, u_now, l_pos, l_u, cfg);
    const double phi_real_new = phi_real(x_candidate, *candidate, x_ref, x_obs, cfg);
    const double delta_hat = phi_real_now - phi_hat_new;
    const double delta = phi_real_now - phi_real_new;

    status = subproblem_status;
    if (delta > cfg.alpha * delta_hat) {
      l_pos *= cfg.beta_grow;
      l_u *= cfg.beta_grow;
      double step_norm = 0.0;
      for (std::size_t idx = 0; idx < x_candidate.size(); ++idx) {
        step_norm = std::max(step_norm, std::abs(x_candidate[idx] - x_now[idx]));
      }
      x_now = std::move(x_candidate);
      u_now = std::move(*candidate);
      if (step_norm < cfg.eps) {
        status = status + "|converged";
        break;
      }
    } else {
      l_pos *= cfg.beta_shrink;
      l_u *= cfg.beta_shrink;
    }
  }

  return {std::move(u_now), std::move(x_now), std::move(status)};
}

py::dict solve_subgradient(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & x0_in,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & x_ref_in,
  const py::object & x_obs_in,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & u_warm_in,
  const py::dict & config_in)
{
  const SolverConfig cfg = parse_config(config_in);
  const State4 x0 = load_state4(x0_in);
  const State4 x_ref = load_state4(x_ref_in);
  const std::optional<Obs2> x_obs = load_obs2(x_obs_in);
  Matrix u_warm = load_u_matrix(u_warm_in, cfg.horizon);

  SolveResult result;
  {
    // Release the GIL so the Python ROS thread can keep running while the
    // native solver works in the background worker thread.
    py::gil_scoped_release release;
    result = solve_subgradient_impl(x0, x_ref, x_obs, std::move(u_warm), cfg);
  }

  py::dict out;
  out["U_opt"] = to_numpy_matrix(result.u_opt, cfg.horizon, 2);
  out["X_opt"] = to_numpy_matrix(result.x_opt, cfg.horizon + 1, 4);
  out["solver_status"] = py::str(result.solver_status);
  return out;
}

py::dict solve_osqp(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & x0_in,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & x_ref_in,
  const py::object & x_obs_in,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & u_warm_in,
  const py::dict & config_in)
{
  const SolverConfig cfg = parse_config(config_in);
  const State4 x0 = load_state4(x0_in);
  const State4 x_ref = load_state4(x_ref_in);
  const std::optional<Obs2> x_obs = load_obs2(x_obs_in);
  Matrix u_warm = load_u_matrix(u_warm_in, cfg.horizon);

  {
    py::gil_scoped_acquire acquire;
    (void)get_osqp_api();
  }

  SolveResult result;
  {
    py::gil_scoped_release release;
    result = solve_osqp_impl(x0, x_ref, x_obs, std::move(u_warm), cfg);
  }

  py::dict out;
  out["U_opt"] = to_numpy_matrix(result.u_opt, cfg.horizon, 2);
  out["X_opt"] = to_numpy_matrix(result.x_opt, cfg.horizon + 1, 4);
  out["solver_status"] = py::str(result.solver_status);
  return out;
}

}  // namespace

PYBIND11_MODULE(_mpc_native, m) {
  m.doc() = "Native MPC solver helpers for llm_drone";
  m.def(
    "solve_subgradient",
    &solve_subgradient,
    py::arg("x0"),
    py::arg("x_ref"),
    py::arg("x_obs"),
    py::arg("u_warm"),
    py::arg("config"),
    "Solve the planar MPC subgradient backend in native C++.");
  m.def(
    "solve_osqp",
    &solve_osqp,
    py::arg("x0"),
    py::arg("x_ref"),
    py::arg("x_obs"),
    py::arg("u_warm"),
    py::arg("config"),
    "Solve the planar MPC convex subproblem with native OSQP and fall back to subgradient.");
}
