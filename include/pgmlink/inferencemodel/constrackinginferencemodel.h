#ifndef CONSTRACKINGINFERENCEMODEL_H
#define CONSTRACKINGINFERENCEMODEL_H

#include <boost/function.hpp>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>

#ifndef NO_ILP
#include <opengm/inference/lpcplex.hxx>
//#include <opengm/inference/lpcplex2.hxx>
#include "pgmlink/inferencemodel/constraint_pool.hxx"
#endif

#include "pgmlink/hypotheses.h"
#include "pgmlink/pgm.h"
#include "pgmlink/inferencemodel/inferencemodel.h"
#include "../pgmlink_export.h"

namespace pgmlink
{

/**
 * @brief The ConsTrackingInferenceModel class builds the OpenGM model needed to run basic conservation tracking.
 * Derived classes such as PerturbedInferenceModel can extend the functionality to support more advanced models.
 * The general usage is to set up this inference model from a hypotheses graph, retrieve the OpenGM model, create
 * an optimizer, add the constraints, and run inference.
 */
class ConsTrackingInferenceModel : public InferenceModel
{
public:
    typedef double ValueType;
    typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::OperatorType OperatorType;
    typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::LabelType LabelType;
    typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::IndexType IndexType;
    typedef std::vector<LabelType> IlpSolution;
    typedef PertGmType GraphicalModelType;
#ifndef NO_ILP
    typedef opengm::LPCplex<PertGmType, pgm::OpengmModelDeprecated::ogmAccumulator> cplex_optimizer;
    typedef opengm::LPCplex2<PertGmType, opengm::Minimizer> cplex2_optimizer;
#endif
    typedef std::map<HypothesesGraph::Node, size_t> HypothesesGraphNodeMap;
    typedef std::map<HypothesesGraph::Arc, size_t> HypothesesGraphArcMap;

public: // API
    // constructor
    PGMLINK_EXPORT ConsTrackingInferenceModel(Parameter& param);

    virtual PGMLINK_EXPORT void build_from_graph(const HypothesesGraph&);

    virtual PGMLINK_EXPORT void fixFirstDisappearanceNodesToLabels(
            const HypothesesGraph& g,
            const HypothesesGraph &tracklet_graph,
            std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &traxel2tracklet_map);
    PGMLINK_EXPORT void fixNodesToLabels( HypothesesGraph& hypothesesGraph );

    PGMLINK_EXPORT GraphicalModelType& get_model();

    virtual PGMLINK_EXPORT IlpSolution infer();
    PGMLINK_EXPORT void set_inference_params(size_t numberOfSolutions,
                              const std::string& feature_filename,
                              const std::string& constraints_filename,
                              const std::string& ground_truth_filename);

    PGMLINK_EXPORT IlpSolution extractSolution(size_t k, const std::string& ground_truth_filename);
    PGMLINK_EXPORT void set_starting_point(const IlpSolution& solution);

    // write or extract results to hypotheses graph
    virtual PGMLINK_EXPORT void conclude(HypothesesGraph &g,
                          HypothesesGraph &tracklet_graph,
                          std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &tracklet2traxel_node_map,
                          IlpSolution &solution);

    PGMLINK_EXPORT IlpSolution extract_solution_from_graph(const HypothesesGraph &g,
                          const HypothesesGraph &tracklet_graph,
                          const std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > &tracklet2traxel_node_map,
                          size_t solutionIndex=0) const;

    // output
    PGMLINK_EXPORT void printResults(const HypothesesGraph &g);
    PGMLINK_EXPORT void write_labeledgraph_to_file(const HypothesesGraph & g,
                                    const std::string &ground_truth_filename);

    PGMLINK_EXPORT void save(const std::string& filename);

    opengm::learning::Weights<double> weights_;
    PGMLINK_EXPORT void setWeight ( size_t, double);
    PGMLINK_EXPORT GraphicalModelType model();
    PGMLINK_EXPORT size_t get_number_of_division_nodes();
    PGMLINK_EXPORT size_t get_number_of_transition_nodes();
    PGMLINK_EXPORT size_t get_number_of_appearance_nodes();
    PGMLINK_EXPORT size_t get_number_of_disappearance_nodes();

protected:
    void add_appearance_nodes( const HypothesesGraph& );
    void add_disappearance_nodes( const HypothesesGraph& );
    void add_transition_nodes( const HypothesesGraph& );
    void add_division_nodes(const HypothesesGraph& );

    void add_finite_factors(const HypothesesGraph& );
    virtual size_t add_division_factors(const HypothesesGraph &g, size_t factorIndex);
    virtual size_t add_transition_factors(const HypothesesGraph &g, size_t factorIndex);
    virtual size_t add_detection_factors(const HypothesesGraph &g, size_t factorIndex);

    virtual void add_constraints_to_pool(const HypothesesGraph& );

    GraphicalModelType::FunctionIdentifier add_marray_as_explicit_function(
        const std::vector<size_t>& shape,
        const marray::Marray<double>& energies);

    // retrieve node and arc maps
    HypothesesGraphNodeMap& get_division_node_map();
    HypothesesGraphNodeMap& get_appearance_node_map();
    HypothesesGraphNodeMap& get_disappearance_node_map();
    HypothesesGraphArcMap& get_arc_map();
    std::map<HypothesesGraph::Node, size_t>& get_detection_factor_node_map();

    template<class INF>
    void add_constraints(INF& optimizer);
    template<class INF>
    void add_linear_constraints(INF& optimizer);

protected:
    GraphicalModelType model_;

    HypothesesGraphNodeMap div_node_map_;
    HypothesesGraphNodeMap app_node_map_;
    HypothesesGraphNodeMap dis_node_map_;
    HypothesesGraphArcMap arc_map_;
    std::map<HypothesesGraph::Node, size_t> detection_f_node_map_;

#ifndef NO_ILP
    cplex_optimizer::Parameter cplex_param_;
    boost::shared_ptr<cplex_optimizer> optimizer_;
    pgm::ConstraintPool constraint_pool_;

    cplex2_optimizer::Parameter cplex2_param_;
    boost::shared_ptr<cplex2_optimizer> optimizer2_;
    pgm::ConstraintPool linear_constraint_pool_;
#endif

    unsigned int number_of_transition_nodes_, number_of_division_nodes_;
    unsigned int number_of_appearance_nodes_, number_of_disappearance_nodes_;
    std::map< size_t, std::vector<size_t> > nodes_per_timestep_;

    bool export_from_labeled_graph_;
    std::string ground_truth_filename_;
    std::map<std::pair<size_t, size_t>, size_t > cplex_variable_id_map_;
    std::map<std::pair<size_t, std::pair<size_t, size_t> >, size_t> cplex_factor_id_map_;
};

#ifndef NO_ILP
template<class INF>
void ConsTrackingInferenceModel::add_constraints(INF &optimizer)
{
    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_constraints";
    constraint_pool_.add_constraints_to_problem(model_, optimizer);
}

template<class INF>
void ConsTrackingInferenceModel::add_linear_constraints(INF &optimizer)
{
    LOG(logDEBUG) << "ConsTrackingInferenceModel::add_linear_constraints";
    linear_constraint_pool_.add_constraints_to_model(model_, optimizer);
}
#endif

} // namespace pgmlink

#endif // CONSTRACKINGINFERENCEMODEL_H
