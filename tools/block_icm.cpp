
#include <iostream>
#include <algorithm>
#include <stack>

#include "pgmlink/pgm.h"
#include "pgmlink/inferencemodel/constraint_pool.hxx"
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/icm.hxx>
// #include <vigra/timing.hxx>

typedef pgmlink::PertGmType GraphicalModel;
typedef opengm::LPCplex<GraphicalModel, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> CPLEX;
typedef std::map<size_t, size_t> IndexMapping;
typedef std::vector<size_t> Solution;

std::vector<size_t> map_factor_indices(std::map<size_t, size_t>& index_mapping, const GraphicalModel& model, size_t factor_id)
{
    std::vector<size_t> new_indices;

    for(size_t i = 0; i < model[factor_id].numberOfVariables(); i++)
    {
        if(index_mapping.find(model[factor_id].variableIndex(i)) == index_mapping.end())
        {
            return std::vector<size_t>();
        }
        else
        {
            new_indices.push_back(index_mapping[model[factor_id].variableIndex(i)]);
        }
    }

    return new_indices;
}

template<class CONTAINER>
std::pair<Solution, IndexMapping> inference_on_submodel(const CONTAINER& nodes, const GraphicalModel& model, pgmlink::pgm::ConstraintPool& cp, const Solution& solution, const std::string& inference_type)
{
    IndexMapping index_mapping;
    GraphicalModel submodel;

    // copy all nodes and store a mapping
    for(auto node_it = nodes.begin(); node_it != nodes.end(); ++node_it)
    {
        index_mapping[*node_it] = submodel.addVariable(model.numberOfLabels(*node_it));
    }

    // copy all contained factors
    for(size_t factor_id = 0; factor_id < model.numberOfFactors(); ++factor_id)
    {
        std::vector<size_t> remapped_factor_indices = map_factor_indices(index_mapping, model, factor_id);
        if(model[factor_id].numberOfVariables() == 0 || remapped_factor_indices.size() > 0)
        {
            pgmlink::pgm::OpengmModelDeprecated::FunctionIdentifier sub_id;

            switch(model[factor_id].functionType())
            {
                case 0:
                    sub_id = submodel.addFunction(model[factor_id].function<0>());
                    break;
                case 1:
                    sub_id = submodel.addFunction(model[factor_id].function<1>());
                    break;
                case 2:
                    sub_id = submodel.addFunction(model[factor_id].function<2>());
                    break;
                case 3:
                    sub_id = submodel.addFunction(model[factor_id].function<3>());
                    break;
                case 4:
                    sub_id = submodel.addFunction(model[factor_id].function<4>());
                    break;
                default:
                    throw std::runtime_error("Function ID too high (>4)");
            }
            submodel.addFactor(sub_id, remapped_factor_indices.begin(), remapped_factor_indices.end());
        }
    }

    Solution subsolution;

    if(inference_type == "CPLEX")
    {
        // create inference type and add constraints
        CPLEX::Parameter param;
        param.verbose_ = false;
        param.integerConstraint_ = true;
        param.epGap_ = 0.0;
        CPLEX inf(submodel, param);
        cp.add_constraints_to_problem(submodel, inf, index_mapping);

        // set a starting point for the optimizer, based on prior solutions to these variables
        std::vector<size_t> starting_point(nodes.size(), 0);
        for(auto node_it = nodes.begin(); node_it != nodes.end(); ++node_it)
        {
            starting_point[index_mapping[*node_it]] = solution[*node_it];
        }
        inf.setStartingPoint(starting_point.begin());

        // infer
        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("CPLEX optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(subsolution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from CPLEX");
        }
    }
    else if(inference_type == "ICM")
    {
        opengm::ICM<GraphicalModel, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(submodel);
        cp.add_constraints_to_problem(submodel, inf, index_mapping);

        // set a starting point for the optimizer, based on prior solutions to these variables
        std::vector<size_t> starting_point(nodes.size(), 0);
        for(auto node_it = nodes.begin(); node_it != nodes.end(); ++node_it)
        {
            starting_point[index_mapping[*node_it]] = solution[*node_it];
        }
        inf.setStartingPoint(starting_point.begin());

        // infer
        opengm::InferenceTermination status = inf.infer();

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("CPLEX optimizer terminated abnormally");
        }

        // extract and print solution
        status = inf.arg(subsolution);

        if (status != opengm::NORMAL)
        {
            throw std::runtime_error("Could not extract solution from CPLEX");
        }
    }

    return std::make_pair(subsolution, index_mapping);
}

int main(int argc, char** argv)
{
    // USETICTOC
    if(argc != 6 && argc != 5)
    {
        std::cout << "Perform something similar to block-ICM on stored conservation tracking models with constraints. 2014 (c) Carsten Haubold" << std::endl;
        std::cout << "\nUSAGE: " << argv[0] << " model.h5 constraints.cp YES|NO (for hierarchical) CPLEX|ICM big-m" << std::endl;
        return 0;
    }

    std::string filename_model(argv[1]);
    std::string filename_constraints(argv[2]);
    bool use_hierarchical = std::string(argv[3]) == "YES";
    std::string inference_type(argv[4]);

    double big_m = 10000000.0;
    if(argc == 6)
    {
        big_m = atof(argv[5]);
    }

    // load model and constraints from disk
    GraphicalModel model;
    opengm::hdf5::load(model, filename_model, "model");

    std::ifstream constraint_pool_input(filename_constraints);
    std::map< size_t, std::vector<size_t> > nodes_per_timestep;
    pgmlink::pgm::ConstraintPool cp;

    {
        boost::archive::text_iarchive ia(constraint_pool_input);
        ia & cp;
        ia & nodes_per_timestep;
    }
    constraint_pool_input.close();
    cp.set_big_m(big_m);

    // dump statistics
    std::cout << "Loaded Model from " << filename_model << std::endl;
    std::cout << "\tVariables: " << model.numberOfVariables() << std::endl;
    std::cout << "\tFactors: " << model.numberOfFactors() << std::endl;

    std::cout << "\nLoaded ConstraintPool from " << filename_constraints << std::endl;
    std::cout << "\tNum Constraints: " << cp.get_num_constraints() << std::endl;

    Solution solution(model.numberOfVariables(), 0);
    Solution differences(model.numberOfVariables(), 2);
    std::vector<int> prior_label_from_block(model.numberOfVariables(), -1);
    std::set< std::set<size_t> > blocks_to_compute;
    std::vector< std::set<size_t> > blocks_done;

    // TIC
    // iterate over time steps, create submodel (view?) just for pairwise time frames, CPLEX inference on submodels only
    for(auto timestep_it = nodes_per_timestep.begin(); timestep_it != nodes_per_timestep.end(); ++timestep_it)
    {
// use with MS VS 2015 and later
//        blocks_to_compute.insert({timestep_it->first});
        std::set<size_t> first_set;
        first_set.insert(timestep_it->first);
        blocks_to_compute.insert(first_set);
// end MS VS 2012 fix
    }

    while(!blocks_to_compute.empty())
    {
        std::set<size_t> timesteps = *(blocks_to_compute.begin());
        std::vector<size_t> nodes;
        std::stringstream timesteps_desc;
        for(size_t t : timesteps)
        {
            timesteps_desc << t << " ";
            nodes.insert(nodes.end(), nodes_per_timestep[t].begin(), nodes_per_timestep[t].end());
        }

        std::pair<Solution, IndexMapping> inf_result = inference_on_submodel(nodes, model, cp, solution, inference_type);
        Solution subsolution = inf_result.first;
        IndexMapping index_mapping = inf_result.second;

        // check for disagreements with prior solution:
        for(auto node_it = nodes.begin(); node_it != nodes.end(); ++node_it)
        {
            if(solution[*node_it] != subsolution[index_mapping[*node_it]] && differences[*node_it] != 2)
            {
                differences[*node_it] = 1;

                if(use_hierarchical)
                {
                    std::set<size_t> new_timesteps;
                    new_timesteps.insert(prior_label_from_block[*node_it]);
                    new_timesteps.insert(timesteps.begin(), timesteps.end());
                    blocks_to_compute.insert(new_timesteps);
                }
            }
            else
            {
                differences[*node_it] = 0;
            }
        }

        // map solution back
        for(auto node_it = nodes.begin(); node_it != nodes.end(); ++node_it)
        {
            solution[*node_it] = subsolution[index_mapping[*node_it]];
            prior_label_from_block[*node_it] = blocks_done.size();
        }

        blocks_done.push_back(*blocks_to_compute.begin());
        blocks_to_compute.erase(blocks_to_compute.begin());

        // evaluate model with current solution
        std::cout << "\n\n++++++++++++++++++++\nSolution after optimizing timesteps " << timesteps_desc.str() << " has energy: " << model.evaluate(solution) << "\n\n" << std::endl;
    }

    std::cout << "\n===============================\nFound Solution:" << std::endl;

    for(auto it = solution.begin(); it != solution.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "\n===============================\nDifferences:" << std::endl;

    for(auto it = solution.begin(); it != solution.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "\nNumber of elements in solution: " << solution.size() << std::endl;

    opengm::ICM<GraphicalModel, pgmlink::pgm::OpengmModelDeprecated::ogmAccumulator> inf(model);
    cp.add_constraints_to_problem(model, inf);
    std::cout << "Solution after optimizing " << nodes_per_timestep.size() << " timesteps has energy: " << model.evaluate(solution) << std::endl;
}
