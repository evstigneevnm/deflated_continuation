#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <main/deflation_continuation/solver_bundle.h>
#include <main/stability_continuation/solver_bundle.h>

namespace
{

struct tracked
{
    inline static std::set<const tracked *> live;
    inline static int                       constructed         = 0;
    inline static int                       fail_at             = -1;
    inline static bool                      invalid_destruction = false;
    std::vector<const tracked *>            dependencies;

    tracked( std::initializer_list<const tracked *> dependencies = {} ) : dependencies( dependencies )
    {
        if ( constructed++ == fail_at )
            throw std::runtime_error( "injected construction failure" );
        live.insert( this );
    }
    ~tracked()
    {
        for ( const auto *dependency : dependencies )
            if ( dependency && live.count( dependency ) == 0 )
                invalid_destruction = true;
        live.erase( this );
    }
};

struct vector_operations : tracked
{
    using scalar_type = double;
    int get_l2_size() const
    {
        return 8;
    }
};

struct file_operations : tracked
{
};
struct log_type : tracked
{
};
struct nonlinear_operations : tracked
{
};

struct linear_operator : tracked
{
    explicit linear_operator( nonlinear_operations *&problem ) : tracked( { problem } )
    {
    }
};

struct preconditioner : tracked
{
    explicit preconditioner( nonlinear_operations *&problem ) : tracked( { problem } )
    {
    }
};

struct linear_system : tracked
{
    linear_system( preconditioner *&prec, vector_operations *&operations, log_type *&log )
        : tracked( { prec, operations, log } )
    {
    }
};

struct linear_solver : tracked
{
    linear_solver( vector_operations *&operations, log_type *&log ) : tracked( { operations, log } )
    {
    }
    void set_preconditioner( preconditioner *prec )
    {
        dependencies.push_back( prec );
    }
};

struct convergence : tracked
{
    convergence( vector_operations *&operations, log_type *&log ) : tracked( { operations, log } )
    {
    }
};

struct system_operator : tracked
{
    template <class Solver>
    system_operator( vector_operations *&operations, linear_operator *&op, Solver *&solver )
        : tracked( { operations, op, solver } )
    {
    }
};

struct newton : tracked
{
    newton( vector_operations *&operations, system_operator *&op, convergence *&strategy )
        : tracked( { operations, op, strategy } )
    {
    }
};

struct knots : tracked
{
};

struct solution_storage : tracked
{
    solution_storage()
    {
        ++alive;
    }

    solution_storage( vector_operations *&operations, int, int, double, log_type *&log )
        : tracked( { operations, log } )
    {
        ++alive;
    }

    ~solution_storage()
    {
        --alive;
    }

    static int alive;
};

int solution_storage::alive = 0;

struct continuation : tracked
{
    continuation(
        vector_operations *&operations, file_operations *&files, log_type *&log, nonlinear_operations *&problem,
        linear_operator *&op, knots *&parameters, linear_system *&solver, newton *&nonlinear_solver
    )
        : tracked( { operations, files, log, problem, op, parameters, solver, nonlinear_solver } )
    {
    }
};

struct diagram : tracked
{
    diagram(
        vector_operations *&operations, file_operations *&files, log_type *&log, nonlinear_operations *&problem,
        newton *&solver, const std::string &, unsigned int
    )
        : tracked( { operations, files, log, problem, solver } )
    {
    }
};

struct deflation : tracked
{
    deflation(
        vector_operations *&operations, file_operations *&files, log_type *&log, nonlinear_operations *&problem,
        linear_operator *&op, linear_system *&solver, solution_storage *&storage
    )
        : tracked( { operations, files, log, problem, op, solver, storage } )
    {
    }
};

struct types
{
    using vector_operations_type       = vector_operations;
    using vector_file_operations_type  = file_operations;
    using log_type                     = ::log_type;
    using nonlinear_operations_type    = nonlinear_operations;
    using linear_operator_type         = linear_operator;
    using preconditioner_type          = preconditioner;
    using linear_system_type           = linear_system;
    using linear_solver_type           = linear_solver;
    using convergence_type             = convergence;
    using system_operator_type         = system_operator;
    using newton_type                  = newton;
    using knots_type                   = knots;
    using solution_storage_type        = solution_storage;
    using continuation_type            = continuation;
    using analytical_continuation_type = continuation;
    using diagram_type                 = diagram;
    using deflation_type               = deflation;
};

void require( bool condition, const std::string &message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

using deflation_bundle = main_classes::deflation_continuation_detail::solver_bundle<types>;
using stability_bundle = main_classes::stability_continuation_detail::solver_bundle<types>;

struct fixture
{
    log_type                              log;
    std::shared_ptr<vector_operations>    operations = std::make_shared<vector_operations>();
    std::shared_ptr<file_operations>      files      = std::make_shared<file_operations>();
    std::shared_ptr<nonlinear_operations> problem    = std::make_shared<nonlinear_operations>();
};

std::unique_ptr<deflation_bundle> make_deflation_bundle( fixture &f )
{
    return std::make_unique<deflation_bundle>( f.operations, f.files, &f.log, &f.log, f.problem, "project", 3 );
}

std::unique_ptr<stability_bundle> make_stability_bundle( fixture &f )
{
    return std::make_unique<stability_bundle>( f.operations, &f.log, &f.log, f.problem );
}

template <class Bundle>
void test_shared_ownership( std::unique_ptr<Bundle> ( *create )( fixture & ) )
{
    static_assert( !std::is_copy_constructible_v<Bundle> );
    fixture                             f;
    std::weak_ptr<vector_operations>    operations = f.operations;
    std::weak_ptr<nonlinear_operations> problem    = f.problem;
    auto                                bundle     = create( f );
    f.operations.reset();
    f.problem.reset();
    f.files.reset();
    require( !operations.expired() && !problem.expired(), "retained shared dependencies" );
    require( bundle->linear_operator() && bundle->newton(), "borrowed solver access" );

    fixture replacement;
    auto    next = create( replacement );
    bundle       = std::move( next );
    require( operations.expired() && problem.expired(), "replacing a bundle releases the previous graph" );
    std::weak_ptr<vector_operations> next_operations = replacement.operations;
    replacement.operations.reset();
    replacement.problem.reset();
    replacement.files.reset();
    bundle.reset();
    require( next_operations.expired(), "no retained dependency cycle" );
}

template <class Bundle>
void test_construction_failure( std::unique_ptr<Bundle> ( *create )( fixture & ) )
{
    fixture    f;
    const auto alive         = tracked::live.size();
    const int  before        = tracked::constructed;
    auto       bundle        = create( f );
    const int  constructions = tracked::constructed - before;
    bundle.reset();
    for ( int index = 0; index < constructions; ++index )
    {
        tracked::fail_at = tracked::constructed + index;
        bool failed      = false;
        try
        {
            auto unexpected = create( f );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        tracked::fail_at = -1;
        require( failed, "injected node construction failure" );
        require( tracked::live.size() == alive, "partially constructed graph cleanup" );
    }
}

void test_shared_external_storage()
{
    fixture                         f;
    auto                            storage  = std::make_shared<solution_storage>();
    std::weak_ptr<solution_storage> observed = storage;
    {
        deflation_bundle bundle( f.operations, f.files, &f.log, &f.log, f.problem, "project", 3, storage );
        require( !bundle.owns_solution_storage(), "supplied storage remains externally managed" );
        require( bundle.solution_storage() == storage.get(), "same storage instance" );
        storage.reset();
        require( !observed.expired(), "shared external storage retained" );
    }
    require( observed.expired(), "shared external storage released" );
}

}

int main()
{
    using bundle_type = main_classes::deflation_continuation_detail::solver_bundle<types>;

    try
    {
        vector_operations    vector_ops;
        file_operations      file_ops;
        log_type             log;
        nonlinear_operations nonlinear_ops;

        {
            bundle_type bundle( &vector_ops, &file_ops, &log, &log, &nonlinear_ops, "project", 3 );
            require( bundle.owns_solution_storage(), "internal storage ownership" );
            require( solution_storage::alive == 1, "internal storage lifetime" );
            require( bundle.linear_operator() != nullptr, "linear operator handle" );
            require( bundle.continuation() != nullptr, "continuation handle" );
            require( bundle.analytical_continuation() != nullptr, "analytical continuation handle" );
            require( bundle.diagram() != nullptr, "diagram handle" );
            require( bundle.deflation() != nullptr, "deflation handle" );
        }
        require( solution_storage::alive == 0, "internal storage cleanup" );

        {
            solution_storage external;
            require( solution_storage::alive == 1, "external storage setup" );
            {
                bundle_type bundle( &vector_ops, &file_ops, &log, &log, &nonlinear_ops, "project", 3, &external );
                require( !bundle.owns_solution_storage(), "external storage ownership" );
                require( bundle.solution_storage() == &external, "external storage handle" );
            }
            require( solution_storage::alive == 1, "external storage survives bundle" );
        }
        require( solution_storage::alive == 0, "external storage final cleanup" );
        test_shared_ownership( make_deflation_bundle );
        test_shared_ownership( make_stability_bundle );
        test_construction_failure( make_deflation_bundle );
        test_construction_failure( make_stability_bundle );
        test_shared_external_storage();
        require( !tracked::invalid_destruction, "consumers destroyed before their dependencies" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    if ( !tracked::live.empty() )
    {
        std::cerr << "FAILED: leaked test resources\n";
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
