#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <chrono>
#include "pgmlink/pgm.h"
#include "pgmlink/inferencemodel/constraint_pool.hxx"
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>

typedef opengm::Adder OperatorType;

template<typename ITERATOR>
double countNumViolatedSoftConstraints(pgmlink::PertGmType& model, ITERATOR labels)
{
    size_t numViolated = 0;
    std::vector<size_t> factor_state(model.factorOrder()+1);
    for(size_t j = 0; j < model.numberOfFactors(); ++j)
    {
        // skip if this is not a constraint function!
        // (function index determined by counting in the function type list of PertGm)
        if(model[j].functionType() > 10)
            continue;

        factor_state[0]=0;
        // construct labeling
        for(size_t i = 0; i < model.numberOfVariables(j); ++i)
        {
            factor_state[i] = labels[model.variableOfFactor(j, i)];
        }

        // evaluate function
        if(model[j](factor_state.begin()) > 0)
            numViolated += 1;
    }
    return numViolated;
}

int main(int argc, char** argv)
{
    if(argc < 4 || argc > 5)
    {
        std::cout << "Inference runner on stored conservation tracking models with constraints. 2014 (c) Carsten Haubold" << std::endl;
        std::cout << "\nUSAGE: " << argv[0] << " model.h5 constraints.cp CPLEX|ICM|LP|LP+ICM <big-m>" << std::endl;
        return 0;
    }

    std::string filename_model(argv[1]);
    std::string filename_constraints(argv[2]);
    std::string inference_type(argv[3]);

    double big_m = 10000000.0;
    if(argc == 5)
    {
        big_m = atof(argv[4]);
    }

    // load model and constraints from disk
//    pgmlink::pgm::OpengmModelDeprecated::ogmGraphicalModel model;
    pgmlink::PertGmType model;
    opengm::hdf5::load(model, filename_model, "model");

    std::ifstream constraint_pool_input(filename_constraints);
    pgmlink::pgm::ConstraintPool cp;

    {
        boost::archive::text_iarchive ia(constraint_pool_input);
        ia & cp;
    }
    constraint_pool_input.close();

    // dump statistics
    std::cout << "Loaded Model from " << filename_model << std::endl;
    std::cout << "\tVariables: " << model.numberOfVariables() << std::endl;
    std::cout << "\tFactors: " << model.numberOfFactors() << std::endl;

    std::cout << "\nLoaded ConstraintPool from " << filename_constraints << std::endl;
    std::cout << "\tNum Constraints: " << cp.get_num_constraints() << std::endl;

    std::vector<pgmlink::pgm::OpengmModelDeprecated::ogmInference::LabelType> solution;
    double solution_value = -999;
    double evaluate_value = -999;

    if(inference_type == "CPLEX")
    {
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator>::Parameter param;
        param.verbose_ = true;
        param.integerConstraint_ = true;
        param.epGap_ = 0.0;
        param.numberOfThreads_ = 1;
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model, param);
        cp.add_constraints_to_problem(model, inf);

        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("CPLEX optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from CPLEX");
        }

//        for(size_t i = 0; i < solution.size(); i++)
//        {
//            opengm::IndependentFactor<double, size_t, size_t> values;
//            inf.variable(i, values);
//            std::cout << "Variable " << i << ": ";
//            for(size_t state = 0; state < model.numberOfLabels(i); state++)
//            {
//                std::cout << "(" << state << ")=" << values(state) << " ";
//            }
//            std::cout << std::endl;
//        }

        solution_value = inf.value();
    }
    else if(inference_type == "LP")
    {
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator>::Parameter param;
        param.verbose_ = true;
        param.integerConstraint_ = false;
        param.epGap_ = 0.0;
        param.numberOfThreads_ = 1;
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model, param);
        cp.add_constraints_to_problem(model, inf);

        std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
        opengm::InferenceTermination status = inf.infer();
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_seconds = endTime - startTime;
        std::cout << "Solving the LP relaxation took " << elapsed_seconds.count() << " secs" << std::endl;

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("CPLEX optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from CPLEX");
        }

        size_t numIntegralVariables = 0;
        for(size_t i = 0; i < solution.size(); i++)
        {
            opengm::IndependentFactor<double, size_t, size_t> values;
            inf.variable(i, values);
            double v = values(solution[i]);
            if(v == 0.0 || v == 1.0)
                numIntegralVariables++;
        }
        std::cout << numIntegralVariables << " variables of " << model.numberOfVariables() << " are integral! "
                  << 100.0 * float(numIntegralVariables) / model.numberOfVariables() << "%" << std::endl;
        solution_value = inf.value();
    }
    else if(inference_type == "ICM")
    {
        size_t numRuns = 10;
        std::srand(std::time(0));
        solution_value = std::numeric_limits<double>::max();
        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> infDummy(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, infDummy);
        std::cout << "After inserting soft constraints, model has " << model.numberOfFactors() << " factors" << std::endl;

        for(size_t run = 0; run < numRuns; run++)
        {
            std::vector<size_t> currentSolution;
            double currentSolutionValue;

            // set up a random initialization
            std::vector<size_t> randomInit(model.numberOfVariables(), 0);
            if(run > 0)
            {
                for(size_t v = 0; v < randomInit.size(); v++)
                {
                    randomInit[v] = std::rand() % model.numberOfLabels(v);
                }
            }

            // configure ICM
            opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model, randomInit);


            // run inference
            opengm::InferenceTermination status = inf.infer();

            if (status != opengm::NORMAL)
            {
                throw std::runtime_error("ICM optimizer terminated abnormally");
            }

            // extract and print solution
            status = inf.arg(currentSolution);

            if (status != opengm::NORMAL)
            {
                throw std::runtime_error("Could not extract solution from ICM");
            }

            currentSolutionValue = inf.value();

            std::cout << "Found solution with value " << currentSolutionValue << " and "
                      << countNumViolatedSoftConstraints(model, currentSolution) << " violated constraints" << std::endl;

            if(solution_value > currentSolutionValue)
            {
                solution_value = currentSolutionValue;
                solution.clear();
                solution.insert(solution.begin(), currentSolution.begin(), currentSolution.end());
            }
        }
    }
    else if (inference_type == "LP+ICM")
    {
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator>::Parameter param;
        param.verbose_ = true;
        param.integerConstraint_ = false;
        param.epGap_ = 0.0;
        opengm::LPCplex<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model, param);
        cp.add_constraints_to_problem(model, inf);

        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("CPLEX optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from CPLEX");
        }

        std::cout << "Value of LP solution: " << inf.value() << std::endl;

        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator>::Parameter param2(solution);
        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> infDummy(model, param2);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, infDummy);
        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf2(model, param2);

        status = inf2.infer();
        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("ICM optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf2.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from ICM");
        }
        solution_value = inf2.value();
    }
    else if(inference_type == "LazyFlip")
    {
        // configure Lazyflip
        opengm::LazyFlipper<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> infDummy(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, infDummy);
        opengm::LazyFlipper<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model);

        // run inference
        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("LazyFlipper optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from LazyFlipper");
        }

        solution_value = inf.value();
    }
    else
    {
        throw std::runtime_error("No valid inference type specified!");
    }

    std::cout << "\n===============================\nFound Solution:" << std::endl;

//    for(auto it = solution.begin(); it != solution.end(); ++it)
//    {
//        std::cout << *it << " ";
//    }
//    std::cout << std::endl;

    std::cout << "Solution has value: " << solution_value << std::endl;

    if(inference_type == "CPLEX" || inference_type == "LP")
    {
        std::cout << "Adding constraints" << std::endl;
        opengm::LazyFlipper<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> infDummy(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, infDummy);
    }

    evaluate_value = model.evaluate(solution);
    std::cout << "Evaluating the model with that solution: " << evaluate_value << std::endl;
    std::cout << "Number of violated soft constraints: " << countNumViolatedSoftConstraints(model, solution) << std::endl;

    return 0;
}
