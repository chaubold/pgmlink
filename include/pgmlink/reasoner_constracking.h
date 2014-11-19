#ifndef CONSTRACKING_REASONER_H
#define CONSTRACKING_REASONER_H

#include <map>
#include <boost/function.hpp>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/lpcplex.hxx>

#include "boost/python.hpp"

#include "pgmlink/pgm.h"
#include "pgmlink/hypotheses.h"
#include "pgmlink/reasoner.h"
#include "pgmlink/feature.h"
#include "pgmlink/uncertaintyParameter.h"
#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/functions/modelviewfunction.hxx"
#include "opengm/functions/view.hxx"

//Random:
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>


namespace pgmlink {	

typedef double ValueType;
typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::OperatorType OperatorType;
typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::LabelType LabelType;
typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::IndexType IndexType;


typedef pgm::OpengmModelDeprecated::ogmGraphicalModel MAPGmType;

typedef opengm::GraphicalModel
		<ValueType, OperatorType,  typename opengm::meta::TypeListGenerator
		<opengm::ModelViewFunction<pgm::OpengmModelDeprecated::ogmGraphicalModel, marray::Marray<ValueType> > , marray::Marray<ValueType> >::type,
		opengm::DiscreteSpace<IndexType,LabelType> > 
		PertGmType;
		

typedef opengm::LPCplex
	<	PertGmType,
			pgm::OpengmModelDeprecated::ogmAccumulator		>
    cplex_optimizer;



typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::FactorType factorType;

typedef typename boost::variate_generator<boost::mt19937, boost::normal_distribution<> > normalRNGType; 
typedef typename boost::variate_generator<boost::mt19937, boost::uniform_real<> > uniformRNGType;    

enum EnergyType {Appearance=0, Disappearance=1, Detection=2, Transition=3, Division=4 };


class Traxel;

class ConservationTracking : public Reasoner {
    public:
    typedef std::vector<pgm::OpengmModelDeprecated::ogmInference::LabelType> IlpSolution;

	ConservationTracking(
                             unsigned int max_number_objects,
                             boost::function<double (const Traxel&, const size_t)> detection,
                             boost::function<double (const Traxel&, const size_t)> division,
                             boost::function<double (const double)> transition,
                             double forbidden_cost = 0,
                             double ep_gap = 0.01,                             
                             bool with_tracklets = false,
                             bool with_divisions = true,
                             boost::function<double (const Traxel&)> disappearance_cost_fn = ConstantFeature(500.0),
                             boost::function<double (const Traxel&)> appearance_cost_fn = ConstantFeature(500.0),
                             bool with_misdetections_allowed = true,
                             bool with_appearance = true,
                             bool with_disappearance = true,
                             double transition_parameter = 5,
                             bool with_constraints = true,
                             UncertaintyParameter param = UncertaintyParameter(),
                             double cplex_timeout = 1e75,
                             double division_weight = 10,
                             double detection_weight = 10,
                             double transition_weight = 10,
                             boost::python::object transition_classifier = boost::python::object(),
                             bool with_optical_correction = false
                             )
        : max_number_objects_(max_number_objects),
          detection_(detection),
          division_(division),
          transition_(transition),
          forbidden_cost_(forbidden_cost),
          optimizer_(NULL),
          ep_gap_(ep_gap),
          with_tracklets_(with_tracklets),
          with_divisions_(with_divisions),
          disappearance_cost_(disappearance_cost_fn),
          appearance_cost_(appearance_cost_fn),
          number_of_transition_nodes_(0), 
          number_of_division_nodes_(0),
          number_of_appearance_nodes_(0),
          number_of_disappearance_nodes_(0),
          with_misdetections_allowed_(with_misdetections_allowed),
          with_appearance_(with_appearance),
          with_disappearance_(with_disappearance),
          transition_parameter_(transition_parameter),
          with_constraints_(with_constraints),
          uncertainty_param_(param),
          cplex_timeout_(cplex_timeout),
          export_from_labeled_graph_(false),
          isMAP_(true),
          division_weight_(division_weight),
          detection_weight_(detection_weight),
          transition_weight_(transition_weight),
          rng_(42),
          random_normal_(rng_,boost::normal_distribution<>(0, 1)),
          random_uniform_(rng_,boost::uniform_real<>(0,1)),
          transition_classifier_(transition_classifier),
          with_optical_correction_(with_optical_correction)
    {

    };
    ~ConservationTracking();

    virtual void formulate( const HypothesesGraph& );
    virtual void infer();
    virtual void conclude( HypothesesGraph& );
    virtual void perturbedInference(HypothesesGraph&, bool with_inference = true);
    
    double forbidden_cost() const;
    bool with_constraints() const;

    /** Return current state of graphical model
     *
     * The returned pointer may be NULL before formulate() is called
     * the first time.
     **/
//    const pgm::OpengmModelDeprecated* get_graphical_model() const;

    /** Return mapping from HypothesesGraph nodes to graphical model variable ids
     *
     * The map is populated after the first call to formulate().
     */
//    const std::map<HypothesesGraph::Node, size_t>& get_node_map() const;

    /** Return mapping from HypothesesGraph arcs to graphical model variable ids
     *
     * The map is populated after the first call to formulate().
     */
    const std::map<HypothesesGraph::Arc, size_t>& get_arc_map() const;
    
    /// Return reference to all CPLEX solution vectors
    const std::vector<IlpSolution>& get_ilp_solutions() const;
    void set_ilp_solutions(const std::vector<IlpSolution>&);

    //cplex export file names
    std::string features_file_;
    std::string constraints_file_;
    std::string ground_truth_file_;

    bool export_from_labeled_graph_;

    void write_labeledgraph_to_file(const HypothesesGraph&);

    static std::string get_export_filename(size_t iteration, const std::string &orig_file_name);
private:
    // copy and assingment have to be implemented, yet
    
    ConservationTracking(const ConservationTracking&):
		random_normal_(rng_,boost::normal_distribution<>(0, 1)),
		random_uniform_(rng_,boost::uniform_real<>(0, 1))
		{};
    ConservationTracking& operator=(const ConservationTracking&) { return *this;};

    void reset();
    void add_constraints(const HypothesesGraph& );
    void add_detection_nodes( const HypothesesGraph& );
    void add_appearance_nodes( const HypothesesGraph& );
    void add_disappearance_nodes( const HypothesesGraph& );
    void add_transition_nodes( const HypothesesGraph& );
    void add_division_nodes(const HypothesesGraph& );
    template <typename ModelType> void add_finite_factors( const HypothesesGraph&, ModelType* model, bool perturb= false, vector<vector<vector<size_t> > >* detoff=NULL );
    double getEnergyByEvent(EnergyType event, HypothesesGraph::NodeIt n,bool perturb=false,size_t state=0);
    void printResults( HypothesesGraph&);
    double sample_with_classifier_variance(double mean, double variance);
    double generateRandomOffset(EnergyType parameterIndex,  double energy=0, Traxel tr=0, Traxel tr2=0, size_t state=0);
    double get_transition_probability(Traxel& tr1, Traxel& tr2, size_t num_outgoing, size_t state);
    double get_transition_variance(Traxel& tr1, Traxel& tr2);
    const marray::Marray<ValueType>  perturbFactor(const factorType* factor,size_t factorId,std::vector<marray::Marray<ValueType> >* detoffset);
    void write_hypotheses_graph_state(const HypothesesGraph& g, const std::string out_fn);

    // funky export maps
    std::map<std::pair<size_t,size_t>,size_t > clpex_variable_id_map_;
    std::map<std::pair<size_t,std::pair<size_t,size_t> >,size_t> clpex_factor_id_map_;
    
    unsigned int max_number_objects_;

    // energy functions
    boost::function<double (const Traxel&, const size_t)> detection_;
    boost::function<double (const Traxel&, const size_t)> division_;
    boost::function<double (const double)> transition_;

    double forbidden_cost_;
    
    boost::shared_ptr<pgm::OpengmModelDeprecated> pgm_;
    //opengm::LPCplex<pgm::OpengmModelDeprecated::ogmGraphicalModel, pgm::OpengmModelDeprecated::ogmAccumulator>* optimizer_;
	boost::shared_ptr<cplex_optimizer> optimizer_;

    std::vector<IlpSolution> solutions_;
	
    std::map<HypothesesGraph::Node, size_t> div_node_map_;
    std::map<HypothesesGraph::Node, size_t> app_node_map_;
    std::map<HypothesesGraph::Node, size_t> dis_node_map_;
    std::map<HypothesesGraph::Arc, size_t> arc_map_;

    typedef std::map<std::pair<Traxel, Traxel >, std::pair<double, double > > TransitionPredictionsMap;
    TransitionPredictionsMap transition_predictions_;

    //factor id maps
    std::map<HypothesesGraph::Node, size_t> detection_f_node_map_;

    double ep_gap_;    

    bool with_tracklets_, with_divisions_;
    boost::function<double (const Traxel&)> disappearance_cost_;
    boost::function<double (const Traxel&)> appearance_cost_;

    unsigned int number_of_transition_nodes_, number_of_division_nodes_;
    unsigned int number_of_appearance_nodes_, number_of_disappearance_nodes_;

    bool with_misdetections_allowed_;
    bool with_appearance_;
    bool with_disappearance_;

    double transition_parameter_;

    bool with_constraints_;

    
    UncertaintyParameter uncertainty_param_;
    double cplex_timeout_;
    bool isMAP_;
    
    double division_weight_; // these cannot be read from the division/detection variable since
    double detection_weight_;// those were converted to boost::function objects in tracking
    double transition_weight_;

    boost::mt19937 rng_;
    normalRNGType random_normal_;
    uniformRNGType random_uniform_;

    boost::python::object transition_classifier_;

    bool with_optical_correction_;

	HypothesesGraph tracklet_graph_;
    std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > tracklet2traxel_node_map_;

    std::map< size_t, std::vector<size_t> > nodes_per_timestep_;
};



/******************/
/* Implementation */
/******************/
 
// template< typename table_t, typename const_iter >
//   void ConservationTracking::add_factor( const table_t& table, const_iter first_idx, const_iter last_idx ){
//   OpengmModelDeprecated::FunctionIdentifier id=pgm_->Model()->addFunction(table);
//   pgm_->Model()->addFactor(id, first_idx, last_idx);
// }
 
} /* namespace pgmlink */
#endif /* MRF_REASONER_H */
  
