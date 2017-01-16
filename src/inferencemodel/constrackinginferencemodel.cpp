#include "pgmlink/inferencemodel/constrackinginferencemodel.h"
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <boost/python.hpp>
#include <iso646.h> // for not, and, or on MSVC

namespace pgmlink
{

#ifndef NO_ILP

ConsTrackingInferenceModel::ConsTrackingInferenceModel(Parameter& param):
    InferenceModel(param),
    number_of_transition_nodes_(0),
    number_of_division_nodes_(0),
    number_of_appearance_nodes_(0),
    number_of_disappearance_nodes_(0),
    ground_truth_filename_(""),
    weights_(5)
{
    cplex_param_.verbose_ = true;
    cplex_param_.integerConstraint_ = true;
    cplex_param_.epGap_ = param_.ep_gap;
    cplex_param_.timeLimit_ = param_.cplex_timeout;
    cplex_param_.numberOfThreads_ = param_.num_threads;
}

void ConsTrackingInferenceModel::build_from_graph(const HypothesesGraph& hypotheses)
{
    LOG(logDEBUG) << "ConsTrackingInferenceModel::formulate: entered";

    LOG(logDEBUG) << "ConsTrackingInferenceModel::formulate: add_transition_nodes";
    add_transition_nodes(hypotheses);
    LOG(logDEBUG) << "ConsTrackingInferenceModel::formulate: add_appearance_nodes";
    add_appearance_nodes(hypotheses);
    LOG(logDEBUG) << "ConsTrackingInferenceModel::formulate: add_disappearance_nodes";
    add_disappearance_nodes(hypotheses);

    LOG(logDEBUG) << "ConsTrackingInferenceModel::formulate: add_division_nodes";
    if (param_.with_divisions)
    {
        add_division_nodes(hypotheses);
    }

    LOG(logINFO) << "number_of_transition_nodes_ = " << number_of_transition_nodes_;
    LOG(logINFO) << "number_of_appearance_nodes_ = " << number_of_appearance_nodes_;
    LOG(logINFO) << "number_of_disappearance_nodes_ = " << number_of_disappearance_nodes_;
    LOG(logINFO) << "number_of_division_nodes_ = " << number_of_division_nodes_;

    add_finite_factors(hypotheses);
    add_constraints_to_pool(hypotheses);
}

void ConsTrackingInferenceModel::fixFirstDisappearanceNodesToLabels(
        const HypothesesGraph &g,
        const HypothesesGraph& tracklet_graph,
        std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> >& traxel2tracklet_map
        )
{

    assert(g.has_property(appearance_label()));
    property_map<appearance_label, HypothesesGraph::base_graph>::type &appearance_labels = g.get(appearance_label());

    if(!param_.with_tracklets)
    {
        property_map<node_timestep, HypothesesGraph::base_graph>::type &timestep_map = g.get(node_timestep());
        int earliest_timestep = *(timestep_map.beginValue());

        for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
        {
            if(timestep_map[n] == earliest_timestep)
            {
                constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(app_node_map_[n], appearance_labels[n]));
                constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(dis_node_map_[n], appearance_labels[n]));
//                linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(app_node_map_[n], appearance_labels[n]));
//                linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(dis_node_map_[n], appearance_labels[n]));
            }
        }
    }
    else
    {
        // in the tracklet graph, the respective label is overwritten by later traxels in the tracklet,
        // get the first original node and use its label
        property_map<node_timestep, HypothesesGraph::base_graph>::type &timestep_map = tracklet_graph.get(node_timestep());
        int earliest_timestep = *(timestep_map.beginValue());

        for (HypothesesGraph::NodeIt n(tracklet_graph); n != lemon::INVALID; ++n)
        {
            if(timestep_map[n] == earliest_timestep)
            {
                HypothesesGraph::Node orig_n = traxel2tracklet_map[n][0];
                constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(app_node_map_[n], appearance_labels[orig_n]));
                constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(dis_node_map_[n], appearance_labels[orig_n]));
//                linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(app_node_map_[n], appearance_labels[orig_n]));
//                linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(dis_node_map_[n], appearance_labels[orig_n]));
            }
        }
    }
}

void ConsTrackingInferenceModel::fixNodesToLabels( HypothesesGraph& g)
{
    typedef property_map<node_traxel, HypothesesGraph::base_graph>::type node_traxel_map;
    node_traxel_map& traxel_map = g.get(node_traxel());

    assert(g.has_property(appearance_label()));
    property_map<appearance_label, HypothesesGraph::base_graph>::type &appearance_labels = g.get(appearance_label());
    typedef property_map<appearance_label, HypothesesGraph::base_graph>::type::ValueIt value_it_type;
    for (value_it_type value_it = ++appearance_labels.beginValue(); value_it != appearance_labels.endValue(); ++value_it)
    {
        property_map<appearance_label, HypothesesGraph::base_graph>::type::ItemIt node_it(appearance_labels, *value_it);
        for (; node_it != lemon::INVALID; ++node_it)
        {
            constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(app_node_map_[node_it], appearance_labels[node_it]-1));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(app_node_map_[node_it], appearance_labels[node_it]-1));
        }
    }

    assert(g.has_property(disappearance_label()));
    property_map<disappearance_label, HypothesesGraph::base_graph>::type &disappearance_labels = g.get(disappearance_label());
    typedef property_map<disappearance_label, HypothesesGraph::base_graph>::type::ValueIt value_it_type;
    for (value_it_type value_it = ++disappearance_labels.beginValue(); value_it != disappearance_labels.endValue(); ++value_it)
    {
        property_map<disappearance_label, HypothesesGraph::base_graph>::type::ItemIt node_it(disappearance_labels, *value_it);
        for (; node_it != lemon::INVALID; ++node_it)
        {
            constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(dis_node_map_[node_it], disappearance_labels[node_it]-1));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(dis_node_map_[node_it], disappearance_labels[node_it]-1));
        }
    }

    assert(g.has_property(division_label()));
    property_map<division_label, HypothesesGraph::base_graph>::type &division_labels = g.get(division_label());
    typedef property_map<division_label, HypothesesGraph::base_graph>::type::ValueIt value_it_type;
    for (value_it_type value_it = ++division_labels.beginValue(); value_it != division_labels.endValue(); ++value_it)
    {
        property_map<division_label, HypothesesGraph::base_graph>::type::ItemIt node_it(division_labels, *value_it);
        for (; node_it != lemon::INVALID; ++node_it)
        {
            constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(div_node_map_[node_it], division_labels[node_it]-1));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(div_node_map_[node_it], division_labels[node_it]-1));
        }
    }

    assert(g.has_property(arc_label()));
    property_map<arc_label, HypothesesGraph::base_graph>::type &arc_labels = g.get(arc_label());
    typedef property_map<arc_label, HypothesesGraph::base_graph>::type::ValueIt arc_value_it_type;
    for (arc_value_it_type value_it = ++arc_labels.beginValue(); value_it != arc_labels.endValue(); ++value_it)
    {
        property_map<arc_label, HypothesesGraph::base_graph>::type::ItemIt arc_it(arc_labels, *value_it);
        for (; arc_it != lemon::INVALID; ++arc_it)
        {
            constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueConstraint(arc_map_[arc_it], arc_labels[arc_it]-1));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::FixNodeValueLinearConstraint(arc_map_[arc_it], arc_labels[arc_it]-1));

        }
    }
}

void ConsTrackingInferenceModel::save(const std::string& filename)
{
    // opengm model
    opengm::hdf5::save(model_, filename, "model");

#ifndef NO_ILP
    // constraint pool
    std::ofstream cp_out_str(filename + ".cp");
    boost::archive::text_oarchive oa(cp_out_str);
    oa & constraint_pool_;
#endif
}


ConsTrackingInferenceModel::GraphicalModelType& ConsTrackingInferenceModel::get_model()
{
    return model_;
}

ConsTrackingInferenceModel::HypothesesGraphNodeMap &ConsTrackingInferenceModel::get_division_node_map()
{
    return div_node_map_;
}

ConsTrackingInferenceModel::HypothesesGraphNodeMap &ConsTrackingInferenceModel::get_appearance_node_map()
{
    return app_node_map_;
}

ConsTrackingInferenceModel::HypothesesGraphNodeMap &ConsTrackingInferenceModel::get_disappearance_node_map()
{
    return dis_node_map_;
}

ConsTrackingInferenceModel::HypothesesGraphArcMap &ConsTrackingInferenceModel::get_arc_map()
{
    return arc_map_;
}

std::map<HypothesesGraph::Node, size_t>& ConsTrackingInferenceModel::get_detection_factor_node_map()
{
    return detection_f_node_map_;
}

void ConsTrackingInferenceModel::add_appearance_nodes(const HypothesesGraph& g)
{
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        model_.addVariable(param_.max_number_objects + 1);
        app_node_map_[n] = model_.numberOfVariables() - 1;

        HypothesesGraph::node_timestep_map& timestep_map = g.get(node_timestep());
        nodes_per_timestep_[timestep_map[n]].push_back(model_.numberOfVariables() - 1);

        assert(model_.numberOfLabels(app_node_map_[n]) == param_.max_number_objects + 1);

//        property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
        ++count;
    }
    number_of_appearance_nodes_ = count;
}

void ConsTrackingInferenceModel::add_disappearance_nodes(const HypothesesGraph& g)
{
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        model_.addVariable(param_.max_number_objects + 1);
        dis_node_map_[n] = model_.numberOfVariables() - 1;

        HypothesesGraph::node_timestep_map& timestep_map = g.get(node_timestep());
        nodes_per_timestep_[timestep_map[n]].push_back(model_.numberOfVariables() - 1);

        assert(model_.numberOfLabels(dis_node_map_[n]) == param_.max_number_objects + 1);

//        property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
        ++count;
    }
    number_of_disappearance_nodes_ = count;
}

void ConsTrackingInferenceModel::add_transition_nodes(const HypothesesGraph& g)
{
    size_t count = 0;
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        model_.addVariable(param_.max_number_objects + 1);
        arc_map_[a] = model_.numberOfVariables() - 1;
        // store these nodes by the timestep of the base-appearance node,
        // as well as in the timestep of the disappearance node they enter
        HypothesesGraph::node_timestep_map& timestep_map = g.get(node_timestep());
        HypothesesGraph::OutArcIt out_arc(g, a);
        HypothesesGraph::Node n = g.source(out_arc);
        nodes_per_timestep_[timestep_map[n]].push_back(model_.numberOfVariables() - 1);
        n = g.target(out_arc);
        nodes_per_timestep_[timestep_map[n]].push_back(model_.numberOfVariables() - 1);

        assert(model_.numberOfLabels(arc_map_[a]) == param_.max_number_objects + 1);

//        property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
        ++count;
    }
    number_of_transition_nodes_ = count;
}

void ConsTrackingInferenceModel::add_division_nodes(const HypothesesGraph& g)
{
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        size_t number_of_outarcs = 0;
        for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a)
        {
            ++number_of_outarcs;
        }
        if (number_of_outarcs > 1)
        {
            model_.addVariable(2);
            div_node_map_[n] = model_.numberOfVariables() - 1;
            HypothesesGraph::node_timestep_map& timestep_map = g.get(node_timestep());
            nodes_per_timestep_[timestep_map[n]].push_back(model_.numberOfVariables() - 1);

            assert(model_.numberOfLabels(div_node_map_[n]) == 2);

//            property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
            ++count;
        }
    }
    number_of_division_nodes_ = count;
}

size_t ConsTrackingInferenceModel::get_number_of_division_nodes(){
    return number_of_division_nodes_;
}

size_t ConsTrackingInferenceModel::get_number_of_transition_nodes(){
    return number_of_transition_nodes_;
}

size_t ConsTrackingInferenceModel::get_number_of_appearance_nodes(){
    return number_of_appearance_nodes_;
}

size_t ConsTrackingInferenceModel::get_number_of_disappearance_nodes(){
    return number_of_disappearance_nodes_;
}

void ConsTrackingInferenceModel::printResults(const HypothesesGraph& g)
{
    property_map<arc_active_count, HypothesesGraph::base_graph>::type& active_arcs_count =
        g.get(arc_active_count());
    property_map<node_active_count, HypothesesGraph::base_graph>::type& active_nodes_count =
        g.get(node_active_count());
    property_map<division_active_count, HypothesesGraph::base_graph>::type& active_divisions_count =
        g.get(division_active_count());

    int c = 0;
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        c = 0;
        for( std::vector<bool>::const_iterator i = active_arcs_count[a].begin();
                i != active_arcs_count[a].end();
                ++i)
        {
            LOG(logDEBUG4) << *i;
            if (*i)
            {
                c++;
            }
        }
        LOG(logDEBUG4) << "total= " << c;
    }

    for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
            it != app_node_map_.end(); ++it)
    {
        c = 0;
        for( std::vector<size_t>::const_iterator i = active_nodes_count[it->first].begin();
                i != active_nodes_count[it->first].end();
                ++i)
        {
            LOG(logINFO) << *i;
            if (*i)
            {
                c++;
            }
        }
        LOG(logINFO) << "total= " << c << std::endl;
    }
    LOG(logDEBUG4) << "division nodes " << c << std::endl;
    for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
            it != app_node_map_.end(); ++it)
    {
        c = 0;
        for( std::vector<bool>::const_iterator i = active_divisions_count[it->first].begin();
                i != active_divisions_count[it->first].end();
                ++i)
        {
            LOG(logDEBUG4) << *i << " ";
            if (*i)
            {
                c++;
            }
        }
        LOG(logDEBUG4) << "total= " << c << std::endl;
    }
}

ConsTrackingInferenceModel::GraphicalModelType::FunctionIdentifier ConsTrackingInferenceModel::add_marray_as_explicit_function(
    const std::vector<size_t>& shape,
    const marray::Marray<double>& energies)
{
    pgm::OpengmModelDeprecated::ExplicitFunctionType func(shape.begin(), shape.end());
        
    auto func_it = func.begin();
    for(auto energies_it = energies.begin(); energies_it != energies.end(); ++energies_it)
    {
        *func_it = *energies_it;
         ++func_it;
    }
    
    return model_.addFunction(func);
}

size_t ConsTrackingInferenceModel::add_detection_factors(const HypothesesGraph& g, size_t factorIndex)
{
    ////
    //// add detection factors
    ////
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map_ = g.get(node_traxel());
    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map_ =
        g.get(node_tracklet());

    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_finite_factors: add detection factors";
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        size_t num_vars = 0;
        std::vector<size_t> vi;
        std::vector<double> cost;

        int node_begin_time = -1;
        int node_end_time = -1;
        if (param_.with_tracklets)
        {
            node_begin_time = tracklet_map_[n].front().Timestep;
            node_end_time = tracklet_map_[n].back().Timestep;
        }
        else
        {
            node_begin_time = traxel_map_[n].Timestep;
            node_end_time = traxel_map_[n].Timestep;
        }


        double energy, e;//, f, w;
        if (app_node_map_.count(n) > 0)
        {
            vi.push_back(app_node_map_[n]);
            // "<" holds if there are only tracklets in the first frame
            if (node_begin_time < g.earliest_timestep())
            {
                // pay no appearance costs in the first timestep
                cost.push_back(0.);
            }
            else
            {
                if (param_.with_tracklets)
                {
                    energy = param_.appearance_cost_fn(tracklet_map_[n].front());
                }
                else
                {
                    energy = param_.appearance_cost_fn(traxel_map_[n]);
                }

                energy += generateRandomOffset(Appearance);
                cost.push_back(energy);
            }
            ++num_vars;
        }

        if (dis_node_map_.count(n) > 0)
        {
            vi.push_back(dis_node_map_[n]);
            // "<" holds if there are only tracklets in the last frame
            if (node_end_time < g.latest_timestep())
            {
                if (param_.with_tracklets)
                {
                    energy = param_.disappearance_cost_fn(tracklet_map_[n].back());
                }
                else
                {
                    energy = param_.disappearance_cost_fn(traxel_map_[n]);
                }

                energy += generateRandomOffset(Disappearance);
                cost.push_back(energy);
            }
            else
            {
                cost.push_back(0);
            }
            ++num_vars;
        }
        // convert vector to array
        std::vector<size_t> coords(num_vars, 0); // number of variables
        std::vector<size_t> shape(num_vars, (param_.max_number_objects + 1));
        marray::Marray<double> energies(shape.begin(), shape.end(), param_.forbidden_cost);

        for (size_t state = 0; state <= param_.max_number_objects; ++state)
        {
            if (param_.with_tracklets)
            {
                energy = 0;
                // add all detection factors of the internal nodes
                for (std::vector<Traxel>::const_iterator trax_it = tracklet_map_[n].begin();
                        trax_it != tracklet_map_[n].end(); ++trax_it)
                {
                    e = param_.detection(*trax_it, state);
                    energy += e;

                    energy += generateRandomOffset(Detection, e, *trax_it, 0, state);
                }
                // add all transition factors of the internal arcs
                Traxel tr_prev;
                bool first = true;
                for (std::vector<Traxel>::const_iterator trax_it = tracklet_map_[n].begin();
                        trax_it != tracklet_map_[n].end(); ++trax_it)
                {
                    LOG(logDEBUG4) << "internal arcs traxel " << *trax_it;
                    Traxel tr = *trax_it;
                    if (!first)
                    {
                        e = param_.transition( get_transition_probability(tr_prev, tr, state) );
                        //e = param_.transition( tr_prev, tr, state);
                        energy += e;
                        energy += generateRandomOffset(Transition, e, tr_prev, tr);
                    }
                    else
                    {
                        first = false;
                    }
                    tr_prev = tr;
                }

            }
            else
            {
                e = param_.detection(traxel_map_[n], state);

                energy = e;
                energy += generateRandomOffset(Detection, e, traxel_map_[n], 0, state);
            }
            LOG(logDEBUG2) << "ConsTrackingInferenceModel::add_finite_factors: detection[" << state
                           << "] = " << energy;
            for (size_t var_idx = 0; var_idx < num_vars; ++var_idx)
            {
                coords[var_idx] = state;
                // if only one of the variables is > 0, then it is an appearance in this time frame
                // or a disappearance in the next timeframe. Hence, add the cost of appearance/disappearance
                // to the detection cost
                energies(coords.begin()) = energy + state * cost[var_idx];

                coords[var_idx] = 0;
                LOG(logDEBUG4) << "ConsTrackingInferenceModel::add_finite_factors: var_idx "
                               << var_idx << " = " << energy;
            }
            // also this energy if both variables have the same state
            if (num_vars == 2)
            {
                coords[0] = state;
                coords[1] = state;
                // only pay detection energy if both variables are on
                energies(coords.begin()) = energy;

                coords[0] = 0;
                coords[1] = 0;

                LOG(logDEBUG4) << "ConsTrackingInferenceModel::add_finite_factors: var_idxs 0 and var_idx 1 = "
                               << energy;
            }
        }

        LOG(logDEBUG3) << "ConsTrackingInferenceModel::add_finite_factors: adding table to pgm";
        //functor add detection table
        factorIndex = add_div_m_best_perturbation(energies, Detection, factorIndex);
        
        GraphicalModelType::FunctionIdentifier funcId = add_marray_as_explicit_function(shape, energies);

        // sorting only works because appearance nodes have lower variable indices than disappearances
        // and the matrix is constructed such that appearances are along coords[0], ...
        sort(vi.begin(), vi.end());
        model_.addFactor(funcId, vi.begin(), vi.end());
        detection_f_node_map_[n] = model_.numberOfFactors() - 1;
    }

    return factorIndex;
}

size_t ConsTrackingInferenceModel::add_transition_factors(const HypothesesGraph& g, size_t factorIndex)
{
    ////
    //// add transition factors
    ////
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map_ = g.get(node_traxel());
    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map_ =
        g.get(node_tracklet());

    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_finite_factors: add transition factors";

    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        size_t vi[] = { arc_map_[a] };
        std::vector<size_t> coords(1, 0); // number of variables

        std::vector<size_t> shape(1, (param_.max_number_objects + 1));
        marray::Marray<double> energies(shape.begin(), shape.end(), param_.forbidden_cost);

        for (size_t state = 0; state <= param_.max_number_objects; ++state)
        {
            Traxel tr1, tr2;
            if (param_.with_tracklets)
            {
                tr1 = tracklet_map_[g.source(a)].back();
                tr2 = tracklet_map_[g.target(a)].front();
            }
            else
            {
                tr1 = traxel_map_[g.source(a)];
                tr2 = traxel_map_[g.target(a)];
            }

            double energy = param_.transition(get_transition_probability(tr1, tr2, state));
            //double energy = param_.transition(tr1, tr2, state);
            energy += generateRandomOffset(Transition, energy, tr1, tr2);

            LOG(logDEBUG2) << "ConsTrackingInferenceModel::add_finite_factors: transition[" << state
                           << "] = " << energy;
            coords[0] = state;
            energies(coords.begin()) = energy;
            coords[0] = 0;
        }
        factorIndex = add_div_m_best_perturbation(energies, Transition, factorIndex);

        GraphicalModelType::FunctionIdentifier funcId = add_marray_as_explicit_function(shape, energies);
        model_.addFactor(funcId, vi, vi + 1);
    }

    return factorIndex;
}

size_t ConsTrackingInferenceModel::add_division_factors(const HypothesesGraph& g, size_t factorIndex)
{
    if(!param_.with_divisions)
    {
        return factorIndex;
    }

    ////
    //// add division factors
    ////
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map_ = g.get(node_traxel());
    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map_ =
        g.get(node_tracklet());

    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_finite_factors: add division factors";
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        if (div_node_map_.count(n) == 0)
        {
            continue;
        }
        size_t vi[] = { div_node_map_[n] };
        std::vector<size_t> coords(1, 0); // number of variables
        std::vector<size_t> shape(1, 2);
        marray::Marray<double> energies(shape.begin(), shape.end(), param_.forbidden_cost);

        for (size_t state = 0; state <= 1; ++state)
        {
            double energy;
            Traxel tr;
            if (param_.with_tracklets)
            {
                tr = tracklet_map_[n].back();
            }
            else
            {
                tr = traxel_map_[n];
            }
            energy = param_.division(tr, state);
            energy += generateRandomOffset(Division, energy,  tr);

            LOG(logDEBUG2) << "ConsTrackingInferenceModel::add_finite_factors: division[" << state
                           << "] = " << energy;
            coords[0] = state;
            energies(coords.begin()) = energy;
            coords[0] = 0;
        }
        factorIndex = add_div_m_best_perturbation(energies, Division, factorIndex);

        GraphicalModelType::FunctionIdentifier funcId = add_marray_as_explicit_function(shape, energies);
        model_.addFactor(funcId, vi, vi + 1);
    }

    return factorIndex;
}

void ConsTrackingInferenceModel::add_finite_factors(const HypothesesGraph& g)
{
    // refactor this:

    // we could use this method to calculate the label-specific offset, also.
    // in order to do so, we would have to write a functor that assigns
    // either the energy or the offset to a table
    // this table is than treated either as an explicit factor table
    // or as an offset marray.
    //
    // Note that this makes it necessary to ensure that factors are added nowhere else in the code
    //
    // Also, this implies that there is some functor choosing either energy or offset

    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_finite_factors: entered";
    size_t factorIndex = 0;
    factorIndex = add_detection_factors(g, factorIndex);
    factorIndex = add_transition_factors(g, factorIndex);
    factorIndex = add_division_factors(g, factorIndex);
    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_finite_factors: finished";
}

void ConsTrackingInferenceModel::add_constraints_to_pool(const HypothesesGraph& g)
{
    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_constraints_to_pool";

    constraint_pool_ = pgm::ConstraintPool(param_.forbidden_cost,
                                           param_.with_divisions,
                                           param_.with_appearance,
                                           param_.with_disappearance,
                                           param_.with_misdetections_allowed);

    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        ////
        //// outgoing transitions
        ////
        {
            std::vector<size_t> transition_nodes;
            for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a)
            {
                transition_nodes.push_back(arc_map_[a]);
            }

            int division_node = -1;
            if(div_node_map_.count(n) > 0)
            {
                division_node = div_node_map_[n];
            }
            size_t appearance_node = app_node_map_[n];

            constraint_pool_.add_constraint(pgm::ConstraintPool::OutgoingConstraint(appearance_node,division_node,transition_nodes));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::OutgoingLinearConstraint(appearance_node,division_node,transition_nodes));
        }

        ////
        //// incoming transitions
        ////
        {
            std::vector<size_t> transition_nodes;
            for (HypothesesGraph::InArcIt a(g, n); a != lemon::INVALID; ++a)
            {
                transition_nodes.push_back(arc_map_[a]);
            }
            size_t disappearance_node = dis_node_map_[n];

            constraint_pool_.add_constraint(pgm::ConstraintPool::IncomingConstraint(transition_nodes,disappearance_node));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::IncomingLinearConstraint(transition_nodes,disappearance_node));
        }

        ////
        //// disappearance/appearance coupling
        ////
        if (app_node_map_.count(n) > 0 && dis_node_map_.count(n) > 0)
        {
            constraint_pool_.add_constraint(pgm::ConstraintPool::DetectionConstraint((size_t)dis_node_map_[n],(size_t)app_node_map_[n]));
//            linear_constraint_pool_.add_constraint(pgm::ConstraintPool::DetectionLinearConstraint((size_t)dis_node_map_[n],(size_t)app_node_map_[n]));
        }
    }

    constraint_pool_.force_softconstraint(!param_.with_constraints);
}

void ConsTrackingInferenceModel::set_inference_params(size_t numberOfSolutions,
        const std::string &feature_filename,
        const std::string &constraints_filename,
        const std::string &ground_truth_filename)
{
    ground_truth_filename_ = ground_truth_filename;
    cplex_param_.verbose_ = true;
    //cplex_param_.integerConstraintNodeVar_ = true;
    //cplex_param_.relaxation_ = cplex_param_.TightPolytope;
    //cplex_param_.useSoftConstraints_ = false;

#ifdef WITH_MODIFIED_OPENGM
    optimizer_ = boost::shared_ptr<cplex_optimizer>(new cplex_optimizer(get_model(),
                 cplex_param_,
                 numberOfSolutions,
                 feature_filename,
                 constraints_filename,
                 ground_truth_filename));

    cplex_variable_id_map_ = optimizer_->get_cplex_variable_id_map();
    cplex_factor_id_map_ = optimizer_->get_cplex_factor_id_map();
#else
    optimizer_ = boost::shared_ptr<cplex_optimizer>(new cplex_optimizer(get_model(), cplex_param_));
#endif

    if(param_.with_constraints)
    {
        LOG(logINFO) << "[ConsTrackingInferenceModel] add_constraints ";
        add_constraints(*optimizer_);
        LOG(logINFO) << "[ConsTrackingInferenceModel] add_constraints";
    }
    else
    {
        throw std::runtime_error("GraphicalModel::infer(): inference with soft constraints is not implemented yet. "
                                 "The conservation tracking factor graph has been saved to file");
    }
}

ConsTrackingInferenceModel::IlpSolution ConsTrackingInferenceModel::infer()
{
    opengm::InferenceTermination status = optimizer_->infer();
    if (status != opengm::NORMAL)
    {
        throw std::runtime_error("GraphicalModel::infer(): optimizer terminated abnormally");
    }

    IlpSolution solution;
    opengm::InferenceTermination statusExtract = optimizer_->arg(solution);
    if (statusExtract != opengm::NORMAL)
    {
        throw std::runtime_error("GraphicalModel::infer(): solution extraction terminated abnormally");
    }

    return solution;
}

ConsTrackingInferenceModel::IlpSolution ConsTrackingInferenceModel::extractSolution(size_t k,
        const std::string &ground_truth_filename)
{
    IlpSolution solution;
#ifdef WITH_MODIFIED_OPENGM
    optimizer_->set_export_file_names("", "", ground_truth_filename);
#endif
    opengm::InferenceTermination status = optimizer_->arg(solution, k);

    if (status != opengm::NORMAL)
    {
        throw std::runtime_error("GraphicalModel::infer(): solution extraction terminated abnormally");
    }

    return solution;
}

void ConsTrackingInferenceModel::write_labeledgraph_to_file(const HypothesesGraph &g,
        const std::string &ground_truth_filename)
{
    using namespace std;

    cout << "write_labeledgraph_to_file" << endl;
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    // if(with_tracklets_)
    // {
    // property_map<traxel_arc_id, HypothesesGraph::base_graph>::type& traxel_arc_id_map = tracklet_graph_.get(traxel_arc_id());
    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map = g.get(node_tracklet());
    // }


    //write this map to label file
    map<int, label_type> cplexid_label_map;
    map<int, stringstream> cplexid_label_info_map;
    map<size_t, std::vector<std::pair<size_t, double> > > cplexid_weight_class_map;

    // fill labels of Variables
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        //appearance
        for (size_t state = 0;
                state < get_model().numberOfLabels(get_appearance_node_map()[n]);
                ++state)
        {
            int id = cplex_variable_id_map_[make_pair(get_appearance_node_map()[n], state)];
            cplexid_label_map[id] = ((g.get(appearance_label())[n] == state) ? 1 : 0);
            LOG(logDEBUG4) << "app\t" << cplexid_label_map[id] << "  " << id << "  "
                           <<  get_appearance_node_map()[n] << "  " << state;
            cplexid_weight_class_map[id].clear();
            cplexid_weight_class_map[id].push_back(std::make_pair(0, 1.));
        }
        //disappearance
        for (size_t state = 0;
                state < get_model().numberOfLabels(get_disappearance_node_map()[n]);
                ++state)
        {
            int id = cplex_variable_id_map_[make_pair(get_disappearance_node_map()[n], state)];
            cplexid_label_map[id] = ((g.get(disappearance_label())[n] == state) ? 1 : 0);
            LOG(logDEBUG4) << "dis\t" << cplexid_label_map[id] << "  " << id << "  "
                           <<  get_disappearance_node_map()[n] << "  " << state;
            cplexid_weight_class_map[id].clear();
            cplexid_weight_class_map[id].push_back(std::make_pair(1, 1.));
        }
        //division
        if(param_.with_divisions and get_division_node_map().count(n) != 0)
        {

            for (size_t state = 0;
                    state < get_model().numberOfLabels(get_division_node_map()[n]);
                    ++state)
            {
                int id = cplex_variable_id_map_[make_pair(get_division_node_map()[n], state)];
                cplexid_label_map[id] = ((g.get(division_label())[n] == state) ? 1 : 0);
                LOG(logDEBUG4) << "div\t" << cplexid_label_map[id] << "  " << id << "  "
                               << get_division_node_map()[n] << "  " << state << "   "; // << number_of_division_nodes_;
                cplexid_weight_class_map[id].clear();
                cplexid_weight_class_map[id].push_back(std::make_pair(2, 1.));
            }
        }
    }
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        //move
        for (size_t state = 0; state < get_model().numberOfLabels(get_arc_map()[a]); ++state)
        {
            int id = cplex_variable_id_map_[make_pair(get_arc_map()[a], state)];
            cplexid_label_map[id] = ((g.get(arc_label())[a] == state) ? 1 : 0);
            LOG(logDEBUG4) << "arc\t" << cplexid_label_map[id] << "  " << id << "  " <<  get_arc_map()[a] << "  " << state;
            cplexid_weight_class_map[id].clear();
            if (param_.with_tracklets and (tracklet_map[g.source(a)]).size() > 1)
            {
                cplexid_weight_class_map[id].push_back(std::make_pair(3, (tracklet_map[g.source(a)]).size() - 1));
                cplexid_weight_class_map[id].push_back(std::make_pair(4, (tracklet_map[g.source(a)]).size()));
            }
            else
            {
                cplexid_weight_class_map[id].push_back(std::make_pair(3, 1.));
            }
        }
    }

    // fill labels of Factors (only second order factors need to be exported (others accounted for in variable states))
    std::map<HypothesesGraph::Node, size_t>& detection_f_node_map = get_detection_factor_node_map();

    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        //detection factor detection_node_map_
        for (size_t s1 = 0; s1 < get_model().numberOfLabels(detection_f_node_map[n]); ++s1)
        {
            for (size_t s2 = 0; s2 < get_model().numberOfLabels(detection_f_node_map[n]); ++s2)
            {
                int id = cplex_factor_id_map_[make_pair(detection_f_node_map[n], make_pair(s1, s2))];
                cplexid_label_map[id] = ((g.get(appearance_label())[n] == s1 and g.get(disappearance_label())[n] == s2) ? 1 : 0);
                LOG(logDEBUG4) << "detection\t" << cplexid_label_map[id] << "  " << id << "  "
                               <<  detection_f_node_map[n] << "  " << s1 << "  " << s2 << endl;
                cplexid_weight_class_map[id].clear();
                if (param_.with_tracklets and (tracklet_map[n]).size() > 1)
                {
                    cplexid_weight_class_map[id].push_back(std::make_pair(3, (tracklet_map[n]).size() - 1));
                    cplexid_weight_class_map[id].push_back(std::make_pair(4, (tracklet_map[n]).size()));
                }
                else
                {
                    cplexid_weight_class_map[id].push_back(std::make_pair(4, 1.));
                }
            }
        }
    }

    //write labels to file
    std::ofstream ground_truth_file;

    ground_truth_file.open (ground_truth_filename, std::ios::app);

    for(std::map<int, size_t>::iterator iterator = cplexid_label_map.begin(); iterator != cplexid_label_map.end(); ++iterator)
    {
        ground_truth_file << iterator->second << "\t\t";
        for (std::vector<std::pair<size_t, double>>::iterator class_weight_pair = cplexid_weight_class_map[iterator->first].begin();
                class_weight_pair != cplexid_weight_class_map[iterator->first].end();
                ++class_weight_pair)
        {
            ground_truth_file << "#c" << class_weight_pair->first << ":" << class_weight_pair->second << " ";
        }
        ground_truth_file << "\tc#\tcplexid:" << iterator->first
                          // << cplexid_label_info_map[iterator->first].str()
                          << endl;
    }
    ground_truth_file.close();
}

void ConsTrackingInferenceModel::conclude( HypothesesGraph& g,
        HypothesesGraph& tracklet_graph,
        std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> >& tracklet2traxel_node_map,
        IlpSolution& solution)
{
    // add 'active' properties to graph
    g.add(node_active2()).add(arc_active()).add(division_active());

    property_map<node_active2, HypothesesGraph::base_graph>::type& active_nodes =
        g.get(node_active2());
    property_map<arc_active, HypothesesGraph::base_graph>::type& active_arcs = g.get(arc_active());
    property_map<division_active, HypothesesGraph::base_graph>::type& division_nodes =
        g.get(division_active());

    // add counting properties for analysis of perturbed models
    g.add(arc_active_count()).add(node_active_count()).add(division_active_count()).add(arc_value_count());

    property_map<arc_active_count, HypothesesGraph::base_graph>::type& active_arcs_count =
        g.get(arc_active_count());
    property_map<node_active_count, HypothesesGraph::base_graph>::type& active_nodes_count =
        g.get(node_active_count());
    property_map<division_active_count, HypothesesGraph::base_graph>::type& active_divisions_count =
        g.get(division_active_count());
    property_map<arc_value_count, HypothesesGraph::base_graph>::type& arc_values =
        g.get(arc_value_count());

    if (!param_.with_tracklets)
    {
        tracklet_graph.add(tracklet_intern_arc_ids()).add(traxel_arc_id());
    }
    property_map<tracklet_intern_arc_ids, HypothesesGraph::base_graph>::type& tracklet_arc_id_map =
        tracklet_graph.get(tracklet_intern_arc_ids());
    property_map<traxel_arc_id, HypothesesGraph::base_graph>::type& traxel_arc_id_map =
        tracklet_graph.get(traxel_arc_id());

    int iterStep = active_nodes_count[get_appearance_node_map().begin()->first].size();
    bool isMAP = (iterStep == 0);

    if (isMAP)
    {
        //initialize vectors for storing optimizer results
        for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
        {
            active_arcs_count.set(a, std::vector<bool>());
            arc_values.set(a, std::vector<size_t>());
        }
        for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
        {
            active_nodes_count.set(n, std::vector<size_t>());
            active_divisions_count.set(n, std::vector<bool>());
        }
    }

    //initialize node counts by 0
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n)
    {
        active_nodes_count.get_value(n).push_back(0);
        active_divisions_count.get_value(n).push_back(0);
    }

    //initialize arc counts by 0
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a)
    {
        active_arcs.set(a, false);
        active_arcs_count.get_value(a).push_back(0);
        arc_values.get_value(a).push_back(0);
    }

    // write state after inference into 'active'-property maps
    // the node is also active if its appearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = get_appearance_node_map().begin();
            it != get_appearance_node_map().end();
            ++it)
    {
        if (param_.with_tracklets)
        {
            // set state of tracklet nodes
            std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map[it->first];

            for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
                    tr_n_it != traxel_nodes.end();
                    ++tr_n_it)
            {
                HypothesesGraph::Node n = *tr_n_it;
                active_nodes.set(n, solution[it->second]);
                active_nodes_count.get_value(n)[iterStep] = solution[it->second];
                //TODO: active_nodes_vector
            }

            // set state of tracklet internal arcs
            std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
            for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                    arc_id_it != arc_ids.end(); ++arc_id_it)
            {
                HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                assert(active_arcs[a] == false);
                if (solution[it->second] > 0)
                {

                    active_arcs.set(a, true);
                    active_arcs_count.get_value(a)[iterStep] = true;
                    arc_values.get_value(a)[iterStep] = solution[it->second];

                    assert(active_nodes[g.source(a)] == solution[it->second]
                           && "tracklet internal arcs must have the same flow as their connected nodes");
                    assert(active_nodes[g.target(a)] == solution[it->second]
                           && "tracklet internal arcs must have the same flow as their connected nodes");
                }
            }
        }
        else
        {
            active_nodes.set(it->first, solution[it->second]);
            active_nodes_count.get_value(it->first)[iterStep] = solution[it->second];
        }
    }
    // the node is also active if its disappearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = get_disappearance_node_map().begin();
            it != get_disappearance_node_map().end(); ++it)
    {
        if (solution[it->second] > 0)
        {
            if (param_.with_tracklets)
            {
                // set state of tracklet nodes
                std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map[it->first];
                for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it =
                            traxel_nodes.begin(); tr_n_it != traxel_nodes.end(); ++tr_n_it)
                {
                    HypothesesGraph::Node n = *tr_n_it;

                    if (active_nodes[n] == 0)
                    {
                        active_nodes.set(n, solution[it->second]);
                        active_nodes_count.get_value(n)[iterStep] = solution[it->second];

                    }
                    else
                    {
                        assert(active_nodes[n] == solution[it->second]);
                    }
                }
                // set state of tracklet internal arcs
                std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
                for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                        arc_id_it != arc_ids.end(); ++arc_id_it)
                {
                    HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                    if (solution[it->second] > 0)
                    {

                        active_arcs.set(a, true);
                        active_arcs_count.get_value(a)[iterStep] = true;
                        arc_values.get_value(a)[iterStep] = solution[it->second];

                        assert(active_nodes[g.source(a)] == solution[it->second]
                               && "tracklet internal arcs must have the same flow as their connected nodes");
                        assert(active_nodes[g.target(a)] == solution[it->second]
                               && "tracklet internal arcs must have the same flow as their connected nodes");
                    }
                }
            }
            else
            {

                if (active_nodes[it->first] == 0)
                {
                    active_nodes.set(it->first, solution[it->second]);
                    active_nodes_count.get_value(it->first)[iterStep] = solution[it->second];

                }
                else
                {
                    assert(active_nodes[it->first] == solution[it->second]);
                }
            }
        }
    }

    for (std::map<HypothesesGraph::Arc, size_t>::const_iterator it = get_arc_map().begin();
            it != get_arc_map().end(); ++it)
    {
        if (solution[it->second] >= 1)
        {
            if (param_.with_tracklets)
            {
                active_arcs.set(g.arcFromId((traxel_arc_id_map[it->first])), true);
                active_arcs_count.get_value(g.arcFromId((traxel_arc_id_map[it->first])))[iterStep] = true;
                arc_values.get_value(g.arcFromId((traxel_arc_id_map[it->first])))[iterStep] = solution[it->second];
            }
            else
            {
                active_arcs.set(it->first, true);
                active_arcs_count.get_value(it->first)[iterStep] = true;
                arc_values.get_value(it->first)[iterStep] = solution[it->second];
            }
        }
    }
    // write division node map
    if (param_.with_divisions)
    {
        for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = get_division_node_map().begin();
                it != get_division_node_map().end(); ++it)
        {
            division_nodes.set(it->first, false);
        }
        for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = get_division_node_map().begin();
                it != get_division_node_map().end(); ++it)
        {

            if (solution[it->second] >= 1)
            {
                if (param_.with_tracklets)
                {
                    // set division property for the last node in the tracklet
                    HypothesesGraph::Node n = tracklet2traxel_node_map[it->first].back();
                    division_nodes.set(n, true);
                    active_divisions_count.get_value(n)[iterStep] = true;
                }
                else
                {
                    division_nodes.set(it->first, true);
                    active_divisions_count.get_value(it->first)[iterStep] = true;
                }
            }
        }
    }
}

ConsTrackingInferenceModel::GraphicalModelType ConsTrackingInferenceModel::model(){
    return model_;
}
ConsTrackingInferenceModel::IlpSolution ConsTrackingInferenceModel::extract_solution_from_graph(
  const HypothesesGraph &g,
  const HypothesesGraph &tracklet_graph,
  const std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &tracklet2traxel_node_map,
  size_t solutionIndex) const
{
    // allocate enough space
    IlpSolution sol(number_of_division_nodes_ + number_of_appearance_nodes_ + number_of_disappearance_nodes_ + number_of_transition_nodes_);

    // WARNING: this only works with the _count maps, and not the single ones!
    if(!g.has_property(arc_value_count()))
    {
        throw std::runtime_error("Can only extract solution from graph if arc_value_count map is present!");
    }
    if(!g.has_property(node_active_count()))
    {
        throw std::runtime_error("Can only extract solution from graph if node_active_count map is present!");
    }
    if(!g.has_property(division_active_count()))
    {
        throw std::runtime_error("Can only extract solution from graph if division_active_count map is present!");
    }

    property_map<arc_value_count, HypothesesGraph::base_graph>::type& arc_values =
        g.get(arc_value_count());
    property_map<node_active_count, HypothesesGraph::base_graph>::type& active_nodes_count =
        g.get(node_active_count());
    property_map<division_active_count, HypothesesGraph::base_graph>::type& active_divisions_count =
        g.get(division_active_count());  
    property_map<node_timestep, HypothesesGraph::base_graph>::type& timestep_map = g.get(node_timestep());  
    int earliest_timestep = *(timestep_map.beginValue());
    int latest_timestep = *(timestep_map.endValue());

    // extract divisions
    for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = div_node_map_.begin();
         it != div_node_map_.end(); 
         ++it)
    {
        assert(it->second < sol.size());
        
        HypothesesGraph::Node node = it->first;
        if(param_.with_tracklets)
            node = tracklet2traxel_node_map.at(node).back();

        assert(active_divisions_count[node].size() > solutionIndex);
        if(active_divisions_count[node][solutionIndex])
            assert(active_nodes_count[node][solutionIndex]);

        sol[it->second] = active_divisions_count[node][solutionIndex] ? 1 : 0;
    }

    std::function<size_t(HypothesesGraph::Node&)> findFlowAlongOutArcs([&](HypothesesGraph::Node& n){
        size_t num_arcs = 0;
        for (HypothesesGraph::OutArcIt oa(g, n); oa != lemon::INVALID; ++oa)
        {
            assert(active_nodes_count[n][solutionIndex] >= arc_values[oa][solutionIndex]);
            num_arcs += arc_values[oa][solutionIndex];
        }
        return num_arcs;
    });

    std::function<size_t(HypothesesGraph::Node&)> findFlowAlongInArcs([&](HypothesesGraph::Node& n){
        size_t num_arcs = 0;
        for (HypothesesGraph::InArcIt ia(g, n); ia != lemon::INVALID; ++ia)
        {
            assert(active_nodes_count[n][solutionIndex] >= arc_values[ia][solutionIndex]);
            num_arcs += arc_values[ia][solutionIndex];
        }
        return num_arcs;
    });

    // extract appearances/disappearances
    HypothesesGraph::NodeIt n = HypothesesGraph::NodeIt(g);
    if(param_.with_tracklets)
        n = HypothesesGraph::NodeIt(tracklet_graph);

    for(; n != lemon::INVALID; ++n)
    {
        assert(app_node_map_.find(n) != app_node_map_.end());
        assert(dis_node_map_.find(n) != dis_node_map_.end());

        HypothesesGraph::Node node_begin = n;
        HypothesesGraph::Node node_end = n;

        if(param_.with_tracklets)
        {
            node_begin = tracklet2traxel_node_map.at(n).front();
            node_end = tracklet2traxel_node_map.at(n).back();
        }

        size_t node_state = active_nodes_count[node_end][solutionIndex];
        size_t incoming = findFlowAlongInArcs(node_begin);
        size_t outgoing = findFlowAlongOutArcs(node_end);
        size_t division = 0;
        if(div_node_map_.find(n) != div_node_map_.end())
            division = sol[div_node_map_.at(n)];

        if(incoming != node_state && incoming != 0)
            throw std::runtime_error("Invalid configuration, incoming transitions don't sum up to node label");
        if(outgoing != node_state + division && outgoing != 0)
            throw std::runtime_error("Invalid configuration, outgoing transitions don't sum to node state + divisions");

        if(timestep_map[n] == earliest_timestep)
            incoming = outgoing - division;
        if(timestep_map[n] == latest_timestep)
        {
            assert(division == 0);
            outgoing = incoming;
        }

        sol[dis_node_map_.at(n)] = incoming;
        sol[app_node_map_.at(n)] = outgoing - division;
    }

    // extract transitions
    for(std::map<HypothesesGraph::Arc, size_t>::const_iterator it = arc_map_.begin();
         it != arc_map_.end(); 
         ++it)
    {
        HypothesesGraph::Arc arc = it->first;
        
        assert(it->second < sol.size());
        assert(arc_values[arc].size() > solutionIndex);
        
        if(param_.with_tracklets)
        {
            HypothesesGraph::Node s = tracklet2traxel_node_map.at(tracklet_graph.source(arc)).back();
            HypothesesGraph::Node t = tracklet2traxel_node_map.at(tracklet_graph.target(arc)).front();

            bool found = false;
            // find proper arc in original hypothesesgraph
            for (HypothesesGraph::OutArcIt oa(g, s); oa != lemon::INVALID; ++oa)
            {
                if(g.target(oa) == t)
                {
                    found = true;
                    arc = oa;
                    break;
                }
            }

            if(!found)
                throw std::runtime_error("Did not find corresponding arc in non-tracklet hypothesesgraph!");

            assert(active_nodes_count[s][solutionIndex] >= arc_values[arc][solutionIndex]);
            assert(active_nodes_count[t][solutionIndex] >= arc_values[arc][solutionIndex]);
        }
        else
        {
            HypothesesGraph::Node s = g.source(arc);
            HypothesesGraph::Node t = g.target(arc);

            if(active_nodes_count[t][solutionIndex] >= arc_values[arc][solutionIndex] && 
                active_nodes_count[s][solutionIndex] >= arc_values[arc][solutionIndex])
            {
                LOG(logDEBUG4) << "all good";
            }
            else
            {
                std::cout << "Problem with arc " << g.id(arc) << " between nodes " << g.id(s) << " and " << g.id(t) << std::endl;
                std::cout << "\tTimesteps " << timestep_map[s] << " and " << timestep_map[t] << std::endl;
                std::cout << "\tActive States " << active_nodes_count[s][solutionIndex] << " and " << active_nodes_count[s][solutionIndex] << std::endl;
                std::cout << "\tArc State " << arc_values[arc][solutionIndex] << std::endl;
            }
            assert(active_nodes_count[t][solutionIndex] >= arc_values[arc][solutionIndex]);
            assert(active_nodes_count[s][solutionIndex] >= arc_values[arc][solutionIndex]);
        }
        
        sol[it->second] = arc_values[arc][solutionIndex];
    }

    return sol;
}

void ConsTrackingInferenceModel::set_starting_point(const IlpSolution& solution)
{
    optimizer_->setStartingPoint(solution.begin());
}

#endif

} // namespace pgmlink
