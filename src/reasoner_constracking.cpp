#include <algorithm>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <string.h>
#include <sstream> 
#include <memory.h>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/datastructures/marray/marray.hxx>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/python.hpp>

#include "pgmlink/hypotheses.h"
#include "pgmlink/log.h"
#include "pgmlink/reasoner_constracking.h"
#include "pgmlink/traxels.h"
#include "pgmlink/inferencemodel/constrackinginferencemodel.h"
#include "pgmlink/inferencemodel/perturbedinferencemodel.h"

//added for view-support
#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/functions/modelviewfunction.hxx"
#include "opengm/functions/view.hxx"

//for computing inverse_sigmoid
#include <boost/math/distributions/normal.hpp>


using namespace std;

namespace pgmlink {

typedef opengm::ModelViewFunction
	<pgm::OpengmModelDeprecated::ogmGraphicalModel, marray::Marray<ValueType> >
	ViewFunctionType;

typedef opengm::LPCplex<pgm::OpengmModelDeprecated::ogmGraphicalModel,
			pgm::OpengmModelDeprecated::ogmAccumulator> cplex_optimizerHG;


ConservationTracking::ConservationTracking(const Parameter &param)
    : max_number_objects_(param.max_number_objects),
      detection_(param.detection),
      division_(param.division),
      transition_(param.transition),
      forbidden_cost_(param.forbidden_cost),
      ep_gap_(param.ep_gap),
      with_tracklets_(param.with_tracklets),
      with_divisions_(param.with_divisions),
      disappearance_cost_(param.disappearance_cost_fn),
      appearance_cost_(param.appearance_cost_fn),
      with_misdetections_allowed_(param.with_misdetections_allowed),
      with_appearance_(param.with_appearance),
      with_disappearance_(param.with_disappearance),
      transition_parameter_(param.transition_parameter),
      with_constraints_(param.with_constraints),
      uncertainty_param_(param.uncertainty_param),
      cplex_timeout_(param.cplex_timeout),
      export_from_labeled_graph_(false),
      isMAP_(true),
      division_weight_(param.division_weight),
      detection_weight_(param.detection_weight),
      transition_weight_(param.transition_weight),
      transition_classifier_(param.transition_classifier),
      with_optical_correction_(param.with_optical_correction)
{
    inference_model_param_.max_number_objects = max_number_objects_;

    inference_model_param_.with_constraints = with_constraints_;
    inference_model_param_.with_tracklets = with_tracklets_;
    inference_model_param_.with_divisions = with_divisions_;
    inference_model_param_.with_appearance = with_appearance_;
    inference_model_param_.with_disappearance = with_disappearance_;
    inference_model_param_.with_misdetections_allowed = with_misdetections_allowed_;
    inference_model_param_.with_optical_correction = with_optical_correction_;

    inference_model_param_.detection = detection_;
    inference_model_param_.division = division_;
    inference_model_param_.transition = transition_;
    inference_model_param_.transition_parameter = transition_parameter_;
    inference_model_param_.transition_classifier = transition_classifier_;

    inference_model_param_.forbidden_cost = forbidden_cost_;
    inference_model_param_.appearance_cost = appearance_cost_;
    inference_model_param_.disappearance_cost = disappearance_cost_;

    perturbed_inference_model_param_.distributionId = uncertainty_param_.distributionId;
    perturbed_inference_model_param_.distributionParam = uncertainty_param_.distributionParam;
    perturbed_inference_model_param_.detection_weight = detection_weight_;
    perturbed_inference_model_param_.division_weight = division_weight_;
    perturbed_inference_model_param_.transition_weight = transition_weight_;
}

ConservationTracking::~ConservationTracking() {
}

double ConservationTracking::forbidden_cost() const {
    return forbidden_cost_;
}


std::string ConservationTracking::get_export_filename(size_t iteration, const std::string& orig_file_name)
{
    std::stringstream export_filename;
    if(!orig_file_name.empty())
    {
        std::string orig_basename = orig_file_name;
        std::string extension = ".txt";

        // remove extension
        std::string::size_type extension_pos = orig_file_name.find_last_of(".");
        if(extension_pos != orig_file_name.npos)
        {
            extension = orig_file_name.substr(extension_pos);
            orig_basename = orig_file_name.substr(0, extension_pos);
        }

        export_filename << orig_basename << "_" << std::setfill('0') << std::setw(3) << iteration << extension;
    }

    return export_filename.str();
}

void ConservationTracking::perturbedInference(HypothesesGraph & hypotheses, bool with_inference)
{
    if (with_inference)
        solutions_.clear();

    HypothesesGraph *graph;

    // for formulate, add_constraints, add_finite_factors: distinguish graph & tracklet_graph
    if (with_tracklets_)
    {
        LOG(logINFO) << "ConservationTracking::perturbedInference: generating tracklet graph";
        tracklet2traxel_node_map_ = generateTrackletGraph2(hypotheses, tracklet_graph_);
        graph = &tracklet_graph_;
    }
    else
    {
        graph = &hypotheses;
    }

    graph->add(relative_uncertainty()).add(node_active_count());

    LOG(logINFO) << "ConservationTracking::perturbedInference: number of iterations: " << uncertainty_param_.numberOfIterations;
    LOG(logINFO) << "ConservationTracking::perturbedInference: perturb using method with Id " << uncertainty_param_.distributionId;
    LOG(logDEBUG) << "ConservationTracking::perturbedInference: formulate ";

    LOG(logDEBUG) << "ConservationTracking::perturbedInference: uncertainty parameter print";
    uncertainty_param_.print();

    //m-best: if perturbation is set to m-best, specify number of solutions. Otherwise, we expect only one solution.
    size_t numberOfSolutions = 1;
    if (uncertainty_param_.distributionId == MbestCPLEX)
    {
        numberOfSolutions = uncertainty_param_.numberOfIterations;
    }

    boost::shared_ptr<ConsTrackingInferenceModel> inference_model = boost::shared_ptr<ConsTrackingInferenceModel>(
            new ConsTrackingInferenceModel(inference_model_param_, ep_gap_, cplex_timeout_));
    inference_model->build_from_graph(*graph);

    IlpSolution sol = inference_model->infer(numberOfSolutions,
                                             get_export_filename(0, features_file_),
                                             constraints_file_,
                                             get_export_filename(0, ground_truth_file_),
                                             with_inference,
                                             export_from_labeled_graph_);

    if (with_inference)
    {
        solutions_.push_back(sol);

        if (export_from_labeled_graph_ and not ground_truth_file_.empty())
        {
            LOG(logINFO) << "export graph labels to " << ground_truth_file_ << std::endl;
            inference_model->write_labeledgraph_to_file(*graph, ground_truth_file_);
            ground_truth_file_.clear();
        }

        LOG(logINFO) << "conclude MAP";
        conclude(hypotheses, inference_model);

        for (size_t k = 1; k < numberOfSolutions; ++k)
        {
            LOG(logINFO) << "conclude " << k + 1 << "-best solution";
            solutions_.push_back(inference_model->extractSolution(k, get_export_filename(k, ground_truth_file_)));

            conclude(hypotheses, inference_model);
        }
    }

    //store offset by factor index
    //technical assumption: the number & order of factors remains the same for the perturbed models
    //performance assumption: the number of pertubations is rather low, so iterating over
    //a list of offset for each iteration Step is not the bottle nec (this is O(n*n) time in the number of pertubations)

    size_t numberOfIterations = uncertainty_param_.numberOfIterations;
    if (uncertainty_param_.distributionId == MbestCPLEX)
    {
        numberOfIterations = 1;
    }

    // get number of factors and then we can dispose of the inference model
    size_t num_factors = inference_model->get_model().numberOfFactors();
    boost::shared_ptr<PerturbedInferenceModel> perturbed_inference_model;

    // offsets for DivMBest
    vector<vector<vector<size_t> > >deterministic_offset(num_factors);

    //deterministic & non-deterministic perturbation
    for (size_t iterStep = 1; iterStep < numberOfIterations; ++iterStep)
    {
        perturbed_inference_model = boost::shared_ptr<PerturbedInferenceModel>(
                new PerturbedInferenceModel(inference_model_param_,
                                            perturbed_inference_model_param_,
                                            ep_gap_, cplex_timeout_));

        // Push away from the solution of the last iteration
        if (uncertainty_param_.distributionId == DiverseMbest)
        {
            for (size_t factorId = 0; factorId < num_factors; ++factorId)
            {
                PertGmType::FactorType factor = inference_model->get_model()[factorId];
                vector<size_t> varIndices;
                for (PertGmType::FactorType::VariablesIteratorType ind = factor.variableIndicesBegin();
                     ind != factor.variableIndicesEnd();
                     ++ind)
                {
                    varIndices.push_back(solutions_[iterStep - 1][*ind]);
                }
                deterministic_offset[factorId].push_back(varIndices);
            }
            perturbed_inference_model->perturb(&deterministic_offset);
        }

        perturbed_inference_model->use_transition_prediction_cache(inference_model.get());
        perturbed_inference_model->build_from_graph(*graph);

        IlpSolution sol = perturbed_inference_model->infer(1,
                                                           get_export_filename(iterStep, features_file_),
                                                           "",
                                                           get_export_filename(iterStep, ground_truth_file_),
                                                           with_inference,
                                                           false);

        if (with_inference)
        {
            solutions_.push_back(sol);
            LOG(logINFO) << "conclude";
            conclude(hypotheses, perturbed_inference_model);
        }
    }

    compute_relative_uncertainty(graph);
}

void ConservationTracking::compute_relative_uncertainty(HypothesesGraph* graph)
{
    graph->add(relative_uncertainty());

    property_map<node_active_count, HypothesesGraph::base_graph>::type &active_nodes = graph->get(node_active_count());
    property_map<relative_uncertainty, HypothesesGraph::base_graph>::type &rel_uncertainty = graph->get(relative_uncertainty());
    for (HypothesesGraph::NodeIt n(*graph); n != lemon::INVALID; ++n)
    {
        double count = 0;
        vector<size_t> *active_list = &active_nodes.get_value(n);
        for (vector<size_t>::iterator is_active = active_list->begin(); is_active != active_list->end(); is_active++)
        {
            if (*is_active != 0)
            {
                ++count;
            }
        }

        rel_uncertainty.set(n, count / uncertainty_param_.numberOfIterations);
    }
}

void ConservationTracking::infer()
{
    throw std::runtime_error("Not implemented");
}

void ConservationTracking::conclude(HypothesesGraph &)
{
    throw std::runtime_error("Not implemented");
}

void ConservationTracking::conclude( HypothesesGraph& g, boost::shared_ptr<ConsTrackingInferenceModel> inference_model) {

    // add 'active' properties to graph
    g.add(node_active2()).add(arc_active()).add(division_active());

    property_map<node_active2, HypothesesGraph::base_graph>::type& active_nodes =
            g.get(node_active2());
    property_map<arc_active, HypothesesGraph::base_graph>::type& active_arcs = g.get(arc_active());
    property_map<division_active, HypothesesGraph::base_graph>::type& division_nodes =
            g.get(division_active());

    // add counting properties for analysis of perturbed models
    g.add(arc_active_count()).add(node_active_count()).add(division_active_count());

    property_map<arc_active_count, HypothesesGraph::base_graph>::type& active_arcs_count =
        g.get(arc_active_count());
    property_map<node_active_count, HypothesesGraph::base_graph>::type& active_nodes_count =
        g.get(node_active_count());
    property_map<division_active_count, HypothesesGraph::base_graph>::type& active_divisions_count =
        g.get(division_active_count());

    if (!with_tracklets_) {
        tracklet_graph_.add(tracklet_intern_arc_ids()).add(traxel_arc_id());
    }
    property_map<tracklet_intern_arc_ids, HypothesesGraph::base_graph>::type& tracklet_arc_id_map =
            tracklet_graph_.get(tracklet_intern_arc_ids());
    property_map<traxel_arc_id, HypothesesGraph::base_graph>::type& traxel_arc_id_map =
            tracklet_graph_.get(traxel_arc_id());

    int iterStep = active_nodes_count[inference_model->get_appearance_node_map().begin()->first].size();
    bool isMAP = (iterStep==0);

    if (isMAP){
        //initialize vectors for storing optimizer results
        for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
        active_arcs_count.set(a,std::vector<bool>());
        }
        for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
            active_nodes_count.set(n,std::vector<long unsigned int>());
            active_divisions_count.set(n, std::vector<bool>());
        }
    }

    //initialize node counts by 0
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        active_nodes_count.get_value(n).push_back(0);
        active_divisions_count.get_value(n).push_back(0);
    }


    //initialize arc counts by 0
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
        active_arcs.set(a, false);
        active_arcs_count.get_value(a).push_back(0);
    }
    // write state after inference into 'active'-property maps
    // the node is also active if its appearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = inference_model->get_appearance_node_map().begin();
            it != inference_model->get_appearance_node_map().end(); ++it) {
        if (with_tracklets_) {
            // set state of tracklet nodes
            std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];

            for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
                    tr_n_it != traxel_nodes.end(); ++tr_n_it) {
                HypothesesGraph::Node n = *tr_n_it;
                active_nodes.set(n, solutions_.back()[it->second]);
                active_nodes_count.get_value(n)[iterStep]=solutions_.back()[it->second];
                //TODO: active_nodes_vector
            }

            // set state of tracklet internal arcs
            std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
            for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                    arc_id_it != arc_ids.end(); ++arc_id_it) {
                HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                assert(active_arcs[a] == false);
                if (solutions_.back()[it->second] > 0) {

                    active_arcs.set(a, true);
                    active_arcs_count.get_value(a)[iterStep]=true;

                    assert(active_nodes[g.source(a)] == solutions_.back()[it->second]
                            && "tracklet internal arcs must have the same flow as their connected nodes");
                    assert(active_nodes[g.target(a)] == solutions_.back()[it->second]
                            && "tracklet internal arcs must have the same flow as their connected nodes");
                }
            }
        } else {
            active_nodes.set(it->first, solutions_.back()[it->second]);
            active_nodes_count.get_value(it->first)[iterStep]=solutions_.back()[it->second];
        }
    }
    // the node is also active if its disappearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = inference_model->get_disappearance_node_map().begin();
            it != inference_model->get_disappearance_node_map().end(); ++it) {
        if (solutions_.back()[it->second] > 0) {
            if (with_tracklets_) {
                // set state of tracklet nodes
                std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
                for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it =
                        traxel_nodes.begin(); tr_n_it != traxel_nodes.end(); ++tr_n_it) {
                    HypothesesGraph::Node n = *tr_n_it;

                    if (active_nodes[n] == 0) {
                        active_nodes.set(n, solutions_.back()[it->second]);
                        active_nodes_count.get_value(n)[iterStep]=solutions_.back()[it->second];

                    } else {
                        assert(active_nodes[n] == solutions_.back()[it->second]);
                    }
                }
                // set state of tracklet internal arcs
                std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
                for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                        arc_id_it != arc_ids.end(); ++arc_id_it) {
                    HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                    if (solutions_.back()[it->second] > 0) {

                        active_arcs.set(a, true);
                        active_arcs_count.get_value(a)[iterStep]=true;

                        assert(active_nodes[g.source(a)] == solutions_.back()[it->second]
                                && "tracklet internal arcs must have the same flow as their connected nodes");
                        assert(active_nodes[g.target(a)] == solutions_.back()[it->second]
                                && "tracklet internal arcs must have the same flow as their connected nodes");
                    }
                }
            } else {

                if (active_nodes[it->first] == 0) {
                    active_nodes.set(it->first, solutions_.back()[it->second]);
                    active_nodes_count.get_value(it->first)[iterStep]=solutions_.back()[it->second];

                } else{
                    assert(active_nodes[it->first] == solutions_.back()[it->second]);
                }
            }
        }
    }

    for (std::map<HypothesesGraph::Arc, size_t>::const_iterator it = inference_model->get_arc_map().begin();
            it != inference_model->get_arc_map().end(); ++it) {
        if (solutions_.back()[it->second] >= 1) {
            if (with_tracklets_) {
                active_arcs.set(g.arcFromId((traxel_arc_id_map[it->first])), true);
                active_arcs_count.get_value(g.arcFromId((traxel_arc_id_map[it->first])))[iterStep]=true;
            } else {
                active_arcs.set(it->first, true);
                active_arcs_count.get_value(it->first)[iterStep]=true;
            }
        }
    }
    // write division node map
    if (with_divisions_) {
        for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = inference_model->get_division_node_map().begin();
                        it != inference_model->get_division_node_map().end(); ++it) {
                    division_nodes.set(it->first, false);
                }
        for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = inference_model->get_division_node_map().begin();
                it != inference_model->get_division_node_map().end(); ++it) {

            if (solutions_.back()[it->second] >= 1) {
                if (with_tracklets_) {
                    // set division property for the last node in the tracklet
                    HypothesesGraph::Node n = tracklet2traxel_node_map_[it->first].back();
                    division_nodes.set(n, true);
                    active_divisions_count.get_value(n)[iterStep]=true;
                } else {
                    division_nodes.set(it->first, true);
                    active_divisions_count.get_value(it->first)[iterStep]=true;
                }
            }
        }
    }
}

void ConservationTracking::formulate(const HypothesesGraph &)
{
    // nothing to do
}

const std::vector<ConservationTracking::IlpSolution> &ConservationTracking::get_ilp_solutions() const
{
    return solutions_;
}

void ConservationTracking::set_ilp_solutions(const std::vector<ConservationTracking::IlpSolution>& solutions)
{
    if(solutions.size() == 0)
    {
        LOG(logWARNING) << "No solutions given to set";
        return;
    }

    solutions_.clear();
    for(std::vector<ConservationTracking::IlpSolution>::const_iterator sol_it = solutions.begin();
        sol_it != solutions.end();
        ++sol_it)
    {
        solutions_.push_back(ConservationTracking::IlpSolution(*sol_it));
    }
}

void ConservationTracking::reset() {
//    solutions_.clear();
}

boost::python::dict convertFeatureMapToPyDict(FeatureMap map){
    boost::python::dict dictionary;
    for (FeatureMap::iterator iter = map.begin(); iter != map.end(); ++iter) {
            dictionary[iter->first] = iter->second;
        }
    return dictionary;
}

} /* namespace pgmlink */
