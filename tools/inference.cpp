#include <iostream>
#include <stdlib.h>
#include "pgmlink/pgm.h"
#include "pgmlink/inferencemodel/constraint_pool.hxx"
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>

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

        for(size_t i = 0; i < solution.size(); i++)
        {
            opengm::IndependentFactor<double, size_t, size_t> values;
            inf.variable(i, values);
            std::cout << "Variable " << i << ": ";
            for(size_t state = 0; state < model.numberOfLabels(i); state++)
            {
                std::cout << "(" << state << ")=" << values(state) << " ";
            }
            std::cout << std::endl;
        }

        solution_value = inf.value();
    }
    else if(inference_type == "LP")
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

        for(size_t i = 0; i < solution.size(); i++)
        {
            opengm::IndependentFactor<double, size_t, size_t> values;
            inf.variable(i, values);
            std::cout << "Variable " << i << ": ";
            for(size_t state = 0; state < model.numberOfLabels(i); state++)
            {
                std::cout << "(" << state << ")=" << values(state) << " ";
            }
            std::cout << std::endl;
        }

        solution_value = inf.value();
    }
    else if(inference_type == "ICM")
    {
        // configure ICM
        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, inf);

        // run inference
        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("ICM optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(solution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from ICM");
        }

        solution_value = inf.value();
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
        opengm::ICM<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf2(model, param2);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, inf2);

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
        opengm::LazyFlipper<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, inf);

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

    for(auto it = solution.begin(); it != solution.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Solution has value: " << solution_value << std::endl;

    if(inference_type == "CPLEX" || inference_type == "LP")
    {
        std::cout << "Adding constraints" << std::endl;
        opengm::LazyFlipper<pgmlink::PertGmType, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model);
        cp.set_big_m(big_m);
        cp.add_constraints_to_problem(model, inf);
    }

    evaluate_value = model.evaluate(solution);
    std::cout << "Evaluating the model with that solution: " << evaluate_value << std::endl;

    return 0;
}
