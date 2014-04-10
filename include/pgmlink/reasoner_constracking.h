#ifndef CONSTRACKING_REASONER_H
#define CONSTRACKING_REASONER_H

#include <map>
#include <boost/function.hpp>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/lpcplex.hxx>

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


typedef opengm::GraphicalModel
		<ValueType, OperatorType,  typename opengm::meta::TypeListGenerator
		<opengm::ModelViewFunction<pgm::OpengmModelDeprecated::ogmGraphicalModel, marray::Marray<ValueType> > >::type, 
		opengm::DiscreteSpace<IndexType,LabelType> > 
		SubGmType;
		
//typedef pgm::OpengmModelDeprecated::ogmGraphicalModel SubGmType;

typedef opengm::LPCplex
	<	SubGmType,
			pgm::OpengmModelDeprecated::ogmAccumulator		>
    cplex_optimizer;


typedef pgm::OpengmModelDeprecated::ogmGraphicalModel::FactorType factorType;

typedef typename boost::variate_generator<boost::mt19937, boost::normal_distribution<> > normalRNGType; 
typedef typename boost::variate_generator<boost::mt19937, boost::uniform_real<> > uniformRNGType;    


class Traxel;

class ConservationTracking : public Reasoner {
    public:
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
                             UncertaintyParameter param = UncertaintyParameter())
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
          param_(param),
		  /*number_of_iterations_(number_of_iterations),
		  distribution_(distribution),
		  distribution_param_(distribution_param),
		  diverse_lambda_(diverse_lambda),
		  m_in_mbest_(m_in_mbest),  */
          isMAP_(true),
          random_normal_(rng_,boost::normal_distribution<>(0, 1)),
		  random_uniform_(rng_,boost::uniform_real<>(0,1))

     {};
    ~ConservationTracking();

    virtual void formulate( const HypothesesGraph& );
    virtual void infer();
    virtual void conclude( HypothesesGraph& );
    virtual void perturbedInference( HypothesesGraph&);
    
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
    void add_finite_factors( const HypothesesGraph& );
    void printResults( HypothesesGraph&);
    double generateRandomOffset(size_t parameterIndex=0,marray::Marray<ValueType>* determOffset = 0,int k=0);
    const marray::Marray<ValueType>  perturbFactor(const factorType* factor,size_t factorId,std::vector<marray::Marray<ValueType> >* detoffset);

    // helper
    size_t cplex_id(size_t opengm_id, size_t state);


    unsigned int max_number_objects_;

    // energy functions
    boost::function<double (const Traxel&, const size_t)> detection_;
    boost::function<double (const Traxel&, const size_t)> division_;
    boost::function<double (const double)> transition_;

    double forbidden_cost_;
    
    shared_ptr<pgm::OpengmModelDeprecated> pgm_;
    //opengm::LPCplex<pgm::OpengmModelDeprecated::ogmGraphicalModel, pgm::OpengmModelDeprecated::ogmAccumulator>* optimizer_;
	cplex_optimizer* optimizer_;
	std::vector<pgm::OpengmModelDeprecated::ogmInference::LabelType> solution_;
	
    std::map<HypothesesGraph::Node, size_t> div_node_map_;
    std::map<HypothesesGraph::Node, size_t> app_node_map_;
    std::map<HypothesesGraph::Node, size_t> dis_node_map_;
    std::map<HypothesesGraph::Arc, size_t> arc_map_;

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
    
    UncertaintyParameter param_;
    bool isMAP_;
    
    boost::mt19937 rng_;
    normalRNGType random_normal_;
    uniformRNGType random_uniform_;
    
	HypothesesGraph tracklet_graph_;
    std::map<HypothesesGraph::Node, std::vector<HypothesesGraph::Node> > tracklet2traxel_node_map_;
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
  
