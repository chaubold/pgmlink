#ifndef INFERENCEMODEL_H
#define INFERENCEMODEL_H

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "pgmlink/hypotheses.h"
#include "pgmlink/pgm.h"
#include "pgmlink/conservationtracking_parameter.h"

#include <boost/python.hpp>

namespace pgmlink
{
enum EnergyType {Appearance = 0, Disappearance = 1, Detection = 2, Transition = 3, Division = 4 };

/**
 * @brief The InferenceModel class is the base for several inference methods using the conservation
 * tracking hypotheses graph: standard CPLEX solver, perturbed CPLEX solver, standard dynamic programming,
 * and perturbed dynamic programming to date.
 */
class InferenceModel
{
public: // typedefs
    typedef std::map<std::pair<Traxel, Traxel >, std::pair<double, double > > TransitionPredictionsMap;

public: // API
    InferenceModel(Parameter& param);
    ~InferenceModel();

    // set a transition_predictions_map that was cached from a previous inference
    void use_transition_prediction_cache(InferenceModel* other);

    // build the inference model from the given graph
    virtual void build_from_graph(const HypothesesGraph&) = 0;

    // write results to hypotheses graph
    virtual void conclude(HypothesesGraph &g,
                          HypothesesGraph &tracklet_graph,
                          std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &tracklet2traxel_node_map,
                          std::vector<size_t> &solution) = 0;

    virtual std::vector<size_t> infer() = 0;
    virtual void fixFirstDisappearanceNodesToLabels(
            const HypothesesGraph& g,
            const HypothesesGraph &tracklet_graph,
            std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &tracklet2traxel_map) = 0;

    virtual void write_labeledgraph_to_file(const HypothesesGraph &,
                                            const std::string&) {}

protected: // methods
    double get_transition_prob(double distance,
                               size_t state,
                               double alpha);
    virtual double get_transition_probability(Traxel &tr1,
            Traxel &tr2,
            size_t state);
    virtual double generateRandomOffset(EnergyType parameterIndex,
                                        double energy = 0,
                                        Traxel tr = 0,
                                        Traxel tr2 = 0,
                                        size_t state = 0);
    virtual size_t add_div_m_best_perturbation(marray::Marray<double> &energies,
            EnergyType energy_type,
            size_t factorIndex);
    bool callable(boost::python::object object);

protected: // members
    boost::shared_ptr<TransitionPredictionsMap> transition_predictions_;
public:
    Parameter& param_;
};

} // namespace pgmlink

#endif // INFERENCEMODEL_H