/**
   @file
   @ingroup pgm
   @brief The Chaingraph model.
*/

#ifndef PGMLINK_PGM_CHAINGRAPH_H
#define PGMLINK_PGM_CHAINGRAPH_H

#include <algorithm>
#include <iterator>
#include <map>
#include <vector>
#include <utility>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bimap.hpp>
#include <opengm/inference/inference.hxx>

#ifdef WITH_GUROBI
#include <opengm/inference/lpgurobi.hxx>
#else
#include <opengm/inference/lpcplex.hxx>
#endif

#include "pgmlink/pgm.h"
#include "pgmlink/hypotheses.h"
#include "pgmlink/features/feature.h"

// Both boost and lemon define the same template ignore_unused_variable_warning<T>.
// Using boost templates with lemon types triggers ADL and the (MSVC++) compiler can't
// resolve the name. Therefore with explicitly define an overload for the two
// involved lemon types in the boost namespace.
namespace boost
{
inline void ignore_unused_variable_warning(const lemon::ListDigraphBase::Node& x)
{
    boost::ignore_unused_variable_warning(x);
}
inline void ignore_unused_variable_warning(const lemon::ListDigraphBase::Arc& x)
{
    boost::ignore_unused_variable_warning(x);
}
}


namespace pgmlink
{
namespace pgm
{
namespace chaingraph
{

#ifdef WITH_GUROBI
  typedef opengm::LPGurobi<OpengmModel, opengm::Minimizer> OpengmLPCplex;
#else
  typedef opengm::LPCplex<OpengmModel, opengm::Minimizer> OpengmLPCplex;
#endif

using boost::function;
using std::map;
using std::vector;

class ModelBuilder;
/**
   @brief Chaingraph model formulated as an Opengm graphical model.

   Represents an opengm model to solve the matching problem in a
   HypothesesGraph. Use a chaingraph::ModelBuilder to construct the
   model. During construction of the chaingraph::Model a random
   variable is added to the graphical model for every node and
   every arc in the HypothesesGraph. The mapping between nodes
   resp. arcs and random variables is stored in the fields
   node_var and arc_var.

   A node in the HypothesesGraph describes a detection in the link
   model. The corresponding random variable determines wether it
   is an actual object or a misdetection. Similarly, an arc is
   interpreted as a possible link between two objects whose state is
   determined by the corresponding random variable.

   @see chaingraph::ModelBuilder
   @see HypothesesGraph
*/
class Model
{
public:
    typedef HypothesesGraph::Node node_t;
    typedef HypothesesGraph::Arc arc_t;
    typedef OpengmModel::IndexType var_t;
    typedef boost::bimap<node_t, var_t>::left_map node_var_map;
    typedef boost::bimap<node_t, var_t>::right_map var_node_map;
    typedef boost::bimap<arc_t, var_t>::left_map arc_var_map;
    typedef boost::bimap<arc_t, var_t>::right_map var_arc_map;

    Model();
    Model( boost::shared_ptr<OpengmModel>,
           const node_var_map&,
           const arc_var_map&
         );

    boost::shared_ptr<OpengmModel> opengm_model; ///< opengm model usually constructed by chaingraph::ModelBuilder

    const node_var_map& var_of_node() const; ///< maps nodes to random variables representing detections
    const var_node_map& node_of_var() const;
    const arc_var_map& var_of_arc() const; ///< maps arcs to random variables representing links
    const var_arc_map& arc_of_var() const;

    var_t var_of_node(node_t) const;
    var_t var_of_arc(arc_t) const;
    node_t node_of_var(var_t) const;
    arc_t arc_of_var(var_t) const;

    enum VarCategory {node_var, arc_var};
    VarCategory var_category(var_t) const;

    enum WeightType {det_weight, mov_weight, div_weight, app_weight, dis_weight, opp_weight};
    map<WeightType, vector<OpengmModel::IndexType> > weight_map; ///< associates events with their corresponding weight ids

    //void set_weights( WeightType, vector<OpengmModel::ValueType> );
    //const vector<OpengmModel::ValueType>& get_weights( WeightType );
private:
    friend class ModelBuilder;

    void init();

    boost::bimap<node_t, var_t> node_var_;
    boost::bimap<arc_t, var_t> arc_var_;
};

class ModelBuilder
{
public:
    ModelBuilder(boost::function<double (const Traxel&)> appearance = ConstantFeature(1000),
                 boost::function<double (const Traxel&)> disappearance = ConstantFeature(1000),
                 boost::function<double (const Traxel&, const Traxel&)> move = SquaredDistance(),
                 double opportunity_cost = 0,
                 double forbidden_cost = 100000)
        : with_detection_vars_(false),
          with_divisions_(false),
          appearance_(appearance),
          disappearance_(disappearance),
          move_(move),
          opportunity_cost_(opportunity_cost),
          forbidden_cost_(forbidden_cost),
          cplex_timeout_(1e+75) {}

    virtual chaingraph::ModelBuilder* clone() const = 0;
    virtual ~ModelBuilder() {}

    // mandatory parameters
    function<double (const Traxel&)> appearance() const
    {
        return appearance_;
    }
    ModelBuilder& appearance( function<double (const Traxel&)> );

    function<double (const Traxel&)> disappearance() const
    {
        return disappearance_;
    }
    ModelBuilder& disappearance( function<double (const Traxel&)> );

    function<double (const Traxel&, const Traxel&)> move() const
    {
        return move_;
    }
    ModelBuilder& move( function<double (const Traxel&, const Traxel&)> );

    double opportunity_cost() const
    {
        return opportunity_cost_;
    }
    ModelBuilder& opportunity_cost( double c )
    {
        opportunity_cost_ = c;
        return *this;
    }

    double forbidden_cost() const
    {
        return forbidden_cost_;
    }
    ModelBuilder& forbidden_cost( double c )
    {
        forbidden_cost_ = c;
        return *this;
    }

    //// optional parameters
    // detection vars
    ModelBuilder& with_detection_vars( function<double (const Traxel&)> detection = ConstantFeature(10),
                                       function<double (const Traxel&)> non_detection = ConstantFeature(200));
    ModelBuilder& without_detection_vars();
    bool has_detection_vars() const
    {
        return with_detection_vars_;
    }
    function<double (const Traxel&)> detection() const
    {
        return detection_;
    }
    function<double (const Traxel&)> non_detection() const
    {
        return non_detection_;
    }

    // divisions
    ModelBuilder& with_divisions( function<double (const Traxel&, const Traxel&, const Traxel&)> div = KasterDivision(10) );
    ModelBuilder& without_divisions();
    bool has_divisions() const
    {
        return with_divisions_;
    }
    function<double (const Traxel&, const Traxel&, const Traxel&)> division() const
    {
        return division_;
    }

    // build
    virtual chaingraph::Model* build( const HypothesesGraph& ) const = 0;

    // refinement
    void add_hard_constraints( const Model&, const HypothesesGraph&, OpengmLPCplex& );
    void fix_detections( const Model&, const HypothesesGraph&, OpengmLPCplex& );

    // cplex parameters
    void set_cplex_timeout( double seconds );


protected:
    void add_detection_vars( const HypothesesGraph&, Model& ) const;
    void add_assignment_vars( const HypothesesGraph&, Model& ) const;

    vector<OpengmModel::IndexType> vars_for_outgoing_factor( const HypothesesGraph&,
            const Model&,
            const HypothesesGraph::Node&) const;
    vector<OpengmModel::IndexType> vars_for_incoming_factor( const HypothesesGraph&,
            const Model&,
            const HypothesesGraph::Node&) const;


private:
    static void couple( const chaingraph::Model&, const HypothesesGraph::Node&, const HypothesesGraph::Arc&, OpengmLPCplex& );

    bool with_detection_vars_;
    bool with_divisions_;

    function<double (const Traxel&)> detection_;
    function<double (const Traxel&)> non_detection_;
    function<double (const Traxel&)> appearance_;
    function<double (const Traxel&)> disappearance_;
    function<double (const Traxel&, const Traxel&)> move_;
    function<double (const Traxel&, const Traxel&, const Traxel&)> division_;
    double opportunity_cost_;
    double forbidden_cost_;
    double cplex_timeout_;
};

class TrainableModelBuilder : public chaingraph::ModelBuilder
{
public:
    TrainableModelBuilder(boost::function<double (const Traxel&)> appearance = ConstantFeature(1000),
                          boost::function<double (const Traxel&)> disappearance = ConstantFeature(1000),
                          boost::function<double (const Traxel&, const Traxel&)> move = SquaredDistance(),
                          double opportunity_cost = 0,
                          double forbidden_cost = 100000)
        : chaingraph::ModelBuilder(appearance, disappearance, move, opportunity_cost, forbidden_cost) {}
    virtual TrainableModelBuilder* clone() const;

    // build
    virtual chaingraph::Model* build( const HypothesesGraph& ) const;

private:
    void add_detection_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;
    void add_outgoing_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;
    void add_incoming_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;


};

class ECCV12ModelBuilder : public chaingraph::ModelBuilder
{
public:
    ECCV12ModelBuilder(boost::function<double (const Traxel&)> appearance = ConstantFeature(1000),
                       boost::function<double (const Traxel&)> disappearance = ConstantFeature(1000),
                       boost::function<double (const Traxel&, const Traxel&)> move = SquaredDistance(),
                       double opportunity_cost = 0,
                       double forbidden_cost = 100000)
        : chaingraph::ModelBuilder(appearance, disappearance, move, opportunity_cost, forbidden_cost) {}
    virtual ECCV12ModelBuilder* clone() const;

    // build
    virtual chaingraph::Model* build( const HypothesesGraph& ) const;

private:
    void add_detection_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;
    void add_outgoing_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;
    void add_incoming_factor( const HypothesesGraph&, Model&, const HypothesesGraph::Node& ) const;
};

/* class ModelTrainer { */
/* public: */
/*   template<class IT1, class IT2, class IT3> */
/* 	std::vector<OpengmModel::ValueType> train(IT1 samples_begin, IT1 samples_end, IT2 node_labels, IT3 arc_labels) const; */
/* }; */

} /* namespace chaingraph */
} /* namespace pgm */
} /* namespace pgmlink */


/**/
/* implementation */
/**/
#include <boost/ptr_container/ptr_vector.hpp>
#include <pgmlink/ext_opengm/structsvm.hxx>
#include <iostream>

#endif /* PGMLINK_PGM_CHAINGRAPH_H */
