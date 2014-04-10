#include <algorithm>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/datastructures/marray/marray.hxx>
//#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
//#include <random>
//#include <cstdlib>

#include "pgmlink/hypotheses.h"
#include "pgmlink/log.h"
#include "pgmlink/reasoner_constracking.h"
#include "pgmlink/traxels.h"
//added for view-support
#include "opengm/opengm.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/functions/modelviewfunction.hxx"
#include "opengm/functions/view.hxx"


using namespace std;

namespace pgmlink {

typedef opengm::ModelViewFunction
	<pgm::OpengmModelDeprecated::ogmGraphicalModel, marray::Marray<ValueType> >
	ViewFunctionType;

typedef opengm::LPCplex<pgm::OpengmModelDeprecated::ogmGraphicalModel,
			pgm::OpengmModelDeprecated::ogmAccumulator> cplex_optimizerHG;


/*typedef opengm::GraphicalModel
		<ValueType, OperatorType,  typename opengm::meta::TypeListGenerator
		<opengm::ModelViewFunction<pgm::OpengmModelDeprecated::ogmGraphicalModel, marray::Marray<ValueType> > >::type, 
		opengm::DiscreteSpace<IndexType,LabelType> > 
		SubGmType;*/


/*typedef opengm::LPCplex
	<	pgm::OpengmModelDeprecated::ogmGraphicalModel,
			pgm::OpengmModelDeprecated::ogmAccumulator		>
    cplex_optimizer;*/

ConservationTracking::~ConservationTracking() {
    	/*if (pgm_ != NULL) {
    		delete pgm_;
    		pgm_ = NULL;
    	}
       if (optimizer_ != NULL) {
          delete optimizer_;
          optimizer_ = NULL;
       }*/
}


double ConservationTracking::forbidden_cost() const {
    return forbidden_cost_;
}

void ConservationTracking::printResults(HypothesesGraph& g){
	//very verbose print of solution
	property_map<arc_active_count, HypothesesGraph::base_graph>::type& active_arcs_count =
		g.get(arc_active_count());
	property_map<node_active_count, HypothesesGraph::base_graph>::type& active_nodes_count =
		g.get(node_active_count());
	property_map<division_active_count, HypothesesGraph::base_graph>::type& active_divisions_count =
	        g.get(division_active_count());
	
	int c;
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
		c=0;
		for( std::vector<bool>::const_iterator i = active_arcs_count[a].begin(); i != active_arcs_count[a].end(); ++i){
			LOG(logDEBUG4) << *i;
			if (*i) {
				c++;
			}
		}
		LOG(logDEBUG4) << "total= "<<c;
	}
	
	for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
		it != app_node_map_.end(); ++it) {
		c=0;
		for( std::vector<long unsigned int>::const_iterator i = active_nodes_count[it->first].begin(); i != active_nodes_count[it->first].end(); ++i){
			LOG(logDEBUG4) << *i;
			if (*i>0){
				c++;
			}
		}
		LOG(logDEBUG4) << "total= "<<c<<std::endl;
	}
	LOG(logDEBUG4) << "division nodes "<<c<<std::endl;
		for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
			it != app_node_map_.end(); ++it) {
			c=0;
			for( std::vector<bool>::const_iterator i = active_divisions_count[it->first].begin(); i != active_divisions_count[it->first].end(); ++i){
				LOG(logDEBUG4) << *i<<" ";
				if (*i>0){
					c++;
				}
			}
		LOG(logDEBUG4) << "total= "<<c<<std::endl;
		}
	}

const marray::Marray<ValueType> ConservationTracking::perturbFactor(const factorType* factor,size_t factorId, std::vector<marray::Marray<ValueType> >* detoffset){
	size_t nOV = factor->numberOfVariables();

	if (nOV>2){
		//only perturb factors with one or two variables
		return marray::Marray<ValueType>();}

	vector<int> off;
	for (size_t ind=0;ind<nOV;ind++){
		off.push_back(factor->numberOfLabels(ind));
	}

	if (nOV==2){
		//This is potentially a factor consisting of an appearance and a disappearance-variable.
		//Check if this is the case, and if so perturb depending on the number of cells

		marray::Marray<ValueType> offset(off.begin(),off.end(),0);
		LOG(logDEBUG4)<<(offset).size()<<" , "<<factor->size();
		LOG(logDEBUG4)<<(offset).dimension()<<" , "<<factor->numberOfVariables();

		//Make sure this is really a dis/app-factor:
		//Look for variable indices: are these app/dis-vars?
		int indexVar1 = factor->variableIndex(0);
		int indexVar2 = factor->variableIndex(1);

		// which is the appearance node, which is the disappearance node?
		int disIndex = -1, appIndex = -1;
		for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();it != app_node_map_.end(); ++it) {
			int indexVar = it->second;
			if (indexVar == indexVar1 || indexVar == indexVar2){
				disIndex = indexVar;
				break;}
		}
		for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = dis_node_map_.begin();it != dis_node_map_.end(); ++it) {
			int indexVar = it->second;
			if (indexVar == indexVar1 || indexVar == indexVar2){
				appIndex = indexVar;
				break;}
		}

		if (disIndex==-1 || appIndex==-1){
			//This isn't a dis/app-factor, as either a disappearance or an appearance node is missing
			return marray::Marray<ValueType>();
		}

		LOG(logDEBUG4)<<"This is a d/a-node";
		//This is a dis/app-node, perturb it with individual sigma
		for(size_t disNum=0;disNum<factor->numberOfLabels(0);disNum++){
			for(size_t appNum=0;appNum<factor->numberOfLabels(1);appNum++){
				if(appNum>0 && disNum>0 && appNum!=disNum) continue;

				int maxnum = max(disNum,appNum);
				offset(disNum,appNum) = generateRandomOffset(maxnum+1);
			}
		}
		return offset;
	} else {
		int nOL = factor->numberOfLabels(0);
		marray::Marray<ValueType> offset(off.begin(),off.end(),0);

		for (int k=0;k<nOL;++k){
			offset(k) = generateRandomOffset(0,&(*detoffset)[factorId],  k);
		}
		return offset;
	}

}

void ConservationTracking::perturbedInference(HypothesesGraph& hypotheses){

	HypothesesGraph *graph;
	if (with_tracklets_) {
	  LOG(logINFO) << "ConservationTracking::perturbedInference: generating tracklet graph";
	  tracklet2traxel_node_map_ = generateTrackletGraph2(hypotheses, tracklet_graph_);
	  graph = &tracklet_graph_; 
	} else {
	  graph = &hypotheses;
	}
	LOG(logDEBUG) << "ConservationTracking::perturbedInference: formulate ";
	formulate(*graph);
	cplex_optimizer::Parameter param;
	param.verbose_ = true;
	param.integerConstraint_ = true;
	param.epGap_ = ep_gap_;
	pgm::OpengmModelDeprecated::ogmGraphicalModel* model = pgm_->Model();
	size_t nOF = model->numberOfFactors();
	

	std::vector<marray::Marray<ValueType> > deterministic_offset;
	
	SubGmType PertMod = SubGmType(model[0].space());
	for(size_t factorId=0; factorId<nOF; ++factorId) {
		
		ViewFunctionType view(*model,factorId,1.0);
		const typename SubGmType::FunctionIdentifier funcId = PertMod.addFunction(view);
		
		PertMod.addFactor(funcId,(*model)[factorId].variableIndicesBegin(),(*model)[factorId].variableIndicesEnd());
		
		deterministic_offset.push_back(marray::Vector<ValueType>((*model)[factorId].numberOfLabels(0),0));
		
    }
    
	LOG(logDEBUG4) << "ConservationTracking::perturbedInference: information about original model: number of factors" << PertMod.numberOfFactors();
	for (size_t j=0;j<PertMod.numberOfFactors();j++){
		LOG(logDEBUG4) << "number of Variables " << PertMod[j].numberOfVariables();
		for (size_t k=0;k<PertMod[j].numberOfVariables();k++){
			LOG(logDEBUG4) << "potential at variable "<<k<<" is " << PertMod[j](k);
		}
	}
    
	//m-best: if perturbation is set to m-best, specify number of solutions. Otherwise, we expect only one solution.
	size_t numberOfSolutions = 1;
	if (param_.distributionId==MbestCPLEX){
		numberOfSolutions = param_.numberOfIterations;
	}

	optimizer_ = new cplex_optimizer(PertMod, param, numberOfSolutions);
	LOG(logINFO) << "add_constraints";
	 if (with_constraints_) {
	  add_constraints(*graph);
	}
	
	LOG(logINFO) << "infer MAP";
	infer();
	LOG(logINFO) << "conclude MAP";
	conclude(*graph);


	for (size_t k=1;k<numberOfSolutions;++k){
		LOG(logINFO) << "conclude "<<k+1<<"-best solution";
		opengm::InferenceTermination status = optimizer_->arg(solution_,k);
		if (status != opengm::NORMAL) {
			throw runtime_error("GraphicalModel::infer(): solution extraction terminated abnormally");
		}
		conclude(*graph);
	}
	size_t numberOfIterations = param_.numberOfIterations;
	if (param_.distributionId==MbestCPLEX){
		numberOfIterations = 1;
	}
	//deterministic & non-deterministic perturbation
	for (size_t iterStep=1;iterStep<numberOfIterations;++iterStep){
		
		LOG(logINFO) << "ConservationTracking::perturbedInference: prepare perturbation number " <<iterStep;
		//initialize new model
		SubGmType PertMod2 = SubGmType(model->space());
		
		//store offsets to keep them in memory
		std::vector<marray::Marray<ValueType> > offset_vector;
		
		//prepare offset for each factor
		for(size_t factorId=0; factorId<nOF; ++factorId) {
			const factorType* factor = &(*model)[factorId];

			//add offset on label that was assigned in the last solution
			if (param_.distributionId==DiverseMbest) {
				deterministic_offset[factorId](solution_[factor->variableIndex(0)])+=param_.distributionParam[0];
			}
			offset_vector.push_back(perturbFactor(factor,factorId,&deterministic_offset));
		}

		//iterate over factors in order to add a perturbed view to the new model
		for(size_t factorId=0; factorId<nOF; factorId++) {
			const factorType* factor = &(*model)[factorId];

			if (offset_vector[factorId].size()==0){
				//this is neither a unary nor a dis/app-node
				//TODO: The size-thing is hacky, solve this more elegantly
				ViewFunctionType view(*model,factorId,1.0); // no offset
				const typename SubGmType::FunctionIdentifier funcId = PertMod2.addFunction(view);
				PertMod2.addFactor(funcId,factor->variableIndicesBegin(),factor->variableIndicesEnd());
			} else {
				ViewFunctionType view(*model,factorId,1.0,&(offset_vector[factorId]));
				const typename SubGmType::FunctionIdentifier funcId = PertMod2.addFunction(view);
				PertMod2.addFactor(funcId,factor->variableIndicesBegin(),factor->variableIndicesEnd());
			}
		}
		
		LOG(logDEBUG4) << "information about perturbed model: " << PertMod2.numberOfFactors();
		for (size_t j=0;j<PertMod2.numberOfFactors();j++){
			LOG(logDEBUG4) << "number of Variables " << PertMod2[j].numberOfVariables();
			for (size_t k=0;k<PertMod2[j].numberOfVariables();++k){
				LOG(logDEBUG4) << "potential at variable k " << PertMod2[j](k);
			}
		}
		
		LOG(logINFO) << "ConservationTracking::perturbedInference construct perturbed model";
		optimizer_ = new cplex_optimizer(PertMod2, param);
		
		if (with_constraints_) {
		       add_constraints(*graph);
		}
		LOG(logINFO) << "infer ";
		infer();
		
		LOG(logINFO) << "conclude";
		conclude(*graph);
	}
	printResults(*graph);
}


double ConservationTracking::generateRandomOffset(size_t parameterIndex, marray::Marray<ValueType>* determOffset ,int k) {
	switch (param_.distributionId) {
		case GaussianPertubation: //normal distribution
				//distribution parameter: sigma
				return random_normal_()*param_.distributionParam[parameterIndex];
				
		case PerturbAndMAP: //Gumbel distribution
				//distribution parameter: beta
				return param_.distributionParam[parameterIndex]*log(-log(random_uniform_()));
		case DiverseMbest: //deterministic offset
				if (parameterIndex==0)
					return (*determOffset)(k); //diverse_lambda
				return param_.distributionParam[parameterIndex]*log(-log(random_uniform_()));
		default: //
				return 0;
	}
}

void ConservationTracking::formulate(const HypothesesGraph& hypotheses) {
    LOG(logDEBUG) << "ConservationTracking::formulate: entered";
    reset();
    pgm_ = boost::shared_ptr < pgm::OpengmModelDeprecated > (new pgm::OpengmModelDeprecated());
	
    HypothesesGraph const *graph;
    graph = &hypotheses;
    

    LOG(logDEBUG) << "ConservationTracking::formulate: add_transition_nodes";
    add_transition_nodes(*graph);
    LOG(logDEBUG) << "ConservationTracking::formulate: add_appearance_nodes";
    add_appearance_nodes(*graph);
    LOG(logDEBUG) << "ConservationTracking::formulate: add_disappearance_nodes";
    add_disappearance_nodes(*graph);

    LOG(logDEBUG) << "ConservationTracking::formulate: add_division_nodes";
    if (with_divisions_) {
        add_division_nodes(*graph);
    }

    LOG(logDEBUG) << "ConservationTracking::formulate: add_finite_factors";
    add_finite_factors(*graph);
    LOG(logDEBUG) << "ConservationTracking::formulate: finished add_finite_factors";

    LOG(logINFO) << "number_of_transition_nodes_ = " << number_of_transition_nodes_;
    LOG(logINFO) << "number_of_appearance_nodes_ = " << number_of_appearance_nodes_;
    LOG(logINFO) << "number_of_disappearance_nodes_ = " << number_of_disappearance_nodes_;
    LOG(logINFO) << "number_of_division_nodes_ = " << number_of_division_nodes_;

}

void ConservationTracking::infer() {
	if (!with_constraints_) {
		//opengm::hdf5::save(optimizer_->graphicalModel(), "./conservationTracking.h5", "conservationTracking");
		throw std::runtime_error("GraphicalModel::infer(): inference with soft constraints is not implemented yet. The conservation tracking factor graph has been saved to file");
	}
    opengm::InferenceTermination status = optimizer_->infer();
    if (status != opengm::NORMAL) {
        throw std::runtime_error("GraphicalModel::infer(): optimizer terminated abnormally");
    }
    opengm::InferenceTermination statusExtract = optimizer_->arg(solution_);
    if (statusExtract != opengm::NORMAL) {
        throw runtime_error("GraphicalModel::infer(): solution extraction terminated abnormally");
    }
}

void ConservationTracking::conclude( HypothesesGraph& g) {
    // extract solution from optimizer

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

	int iterStep = active_nodes_count[app_node_map_.begin()->first].size();
	bool isMAP = (iterStep==0);

	if (isMAP){
		//initialize vectors for storing optimizer results
		for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
		active_arcs_count.set(a,std::vector<bool>());
		}
		for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
		it != app_node_map_.end(); ++it) {
			if (with_tracklets_) {
				std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
				for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
						tr_n_it != traxel_nodes.end(); ++tr_n_it) {
					active_nodes_count.set(*tr_n_it,std::vector<long unsigned int>());
				}
			} else {
				active_nodes_count.set(it->first,std::vector<long unsigned int>());
			}
		}
		for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = div_node_map_.begin();
		it != div_node_map_.end(); ++it) {
			if (with_tracklets_) {
				std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
				for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
						tr_n_it != traxel_nodes.end(); ++tr_n_it) {
					active_divisions_count.set(*tr_n_it, std::vector<bool>());
				}
			} else {
				active_divisions_count.set(it->first, std::vector<bool>());
			}
		}
	}
	
	
	//initialize node counts by 0
	for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
		it != app_node_map_.end(); ++it) {
		if (with_tracklets_) {
			std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
            for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
                    tr_n_it != traxel_nodes.end(); ++tr_n_it) {
				HypothesesGraph::Node n = *tr_n_it;
				active_nodes_count.get_value(n).push_back(0);
			}
		}else {
			active_nodes_count.get_value(it->first).push_back(0);
		}
	}

	//initialize division active by "false"
	for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = div_node_map_.begin();
		it != div_node_map_.end(); ++it) {
		if(with_tracklets_) {
			std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
            for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
                    tr_n_it != traxel_nodes.end(); ++tr_n_it) {
				HypothesesGraph::Node n = *tr_n_it;
				active_divisions_count.get_value(n).push_back(0);
			}
		} else {
			active_divisions_count.get_value(it->first).push_back(0);
		}
	}
	
	
	//initialize  arc counts by 0
	for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
		active_arcs.set(a, false);
		active_arcs_count.get_value(a).push_back(0);
	}
	
    // write state after inference into 'active'-property maps
    // the node is also active if its appearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = app_node_map_.begin();
            it != app_node_map_.end(); ++it) {
        if (with_tracklets_) {
            // set state of tracklet nodes
            std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
            for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it = traxel_nodes.begin();
                    tr_n_it != traxel_nodes.end(); ++tr_n_it) {
                HypothesesGraph::Node n = *tr_n_it;
                
				active_nodes.set(n, solution_[it->second]);
				active_nodes_count.get_value(n)[iterStep]=solution_[it->second];
            }
            // set state of tracklet internal arcs
            std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
            for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                    arc_id_it != arc_ids.end(); ++arc_id_it) {
                HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                assert(active_arcs[a] == false);
                if (solution_[it->second] > 0) {
					
					active_arcs.set(a, true);

					active_arcs_count.get_value(a)[iterStep]=true;
                    
                    assert(active_nodes[g.source(a)] == solution_[it->second]
                            && "tracklet internal arcs must have the same flow as their connected nodes");
                    assert(active_nodes[g.target(a)] == solution_[it->second]
                            && "tracklet internal arcs must have the same flow as their connected nodes");
                }
            }
        } else {
			
			active_nodes.set(it->first, solution_[it->second]);
			active_nodes_count.get_value(it->first)[iterStep]=solution_[it->second];
        }
    }

    // the node is also active if its disappearance node is active
    for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = dis_node_map_.begin();
            it != dis_node_map_.end(); ++it) {

        if (solution_[it->second] > 0) {
            if (with_tracklets_) {
                // set state of tracklet nodes
                std::vector<HypothesesGraph::Node> traxel_nodes = tracklet2traxel_node_map_[it->first];
                for (std::vector<HypothesesGraph::Node>::const_iterator tr_n_it =
                        traxel_nodes.begin(); tr_n_it != traxel_nodes.end(); ++tr_n_it) {
                    HypothesesGraph::Node n = *tr_n_it;
                    
                    if (active_nodes[n] == 0) {
							
						active_nodes.set(n, solution_[it->second]);
						active_nodes_count.get_value(n)[iterStep]=solution_[it->second];
                    
                    } else {
                        assert(active_nodes[n] == solution_[it->second]);
                    }
                }
                // set state of tracklet internal arcs
                std::vector<int> arc_ids = tracklet_arc_id_map[it->first];
                for (std::vector<int>::const_iterator arc_id_it = arc_ids.begin();
                        arc_id_it != arc_ids.end(); ++arc_id_it) {
                    HypothesesGraph::Arc a = g.arcFromId(*arc_id_it);
                    if (solution_[it->second] > 0) {
						
						active_arcs.set(a, true);

						active_arcs_count.get_value(a)[iterStep]=true;
                        
                        assert(active_nodes[g.source(a)] == solution_[it->second]
                                && "tracklet internal arcs must have the same flow as their connected nodes");
                        assert(active_nodes[g.target(a)] == solution_[it->second]
                                && "tracklet internal arcs must have the same flow as their connected nodes");
                    }
                }
            } else {
				
				if (active_nodes[it->first] == 0) {
					active_nodes.set(it->first, solution_[it->second]);
					active_nodes_count.get_value(it->first)[iterStep]=solution_[it->second];
					
                } else{
                    assert(active_nodes[it->first] == solution_[it->second]);
                }
            }
        }
    }

    for (std::map<HypothesesGraph::Arc, size_t>::const_iterator it = arc_map_.begin();
            it != arc_map_.end(); ++it) {
        if (solution_[it->second] >= 1) {
            if (with_tracklets_) {
				active_arcs.set(g.arcFromId((traxel_arc_id_map[it->first])), true);

				active_arcs_count.get_value(g.arcFromId((traxel_arc_id_map[it->first]))).push_back(true);
            } else {
				active_arcs.set(it->first, true);
				active_arcs_count.get_value(it->first)[iterStep]=true;
            }
        }
    }
    // write division node map
    if (with_divisions_ && isMAP) {
        for (std::map<HypothesesGraph::Node, size_t>::const_iterator it = div_node_map_.begin();
                it != div_node_map_.end(); ++it) {
            if (solution_[it->second] >= 1) {
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

const std::map<HypothesesGraph::Arc, size_t>& ConservationTracking::get_arc_map() const {
    return arc_map_;
}

void ConservationTracking::reset() {
    if (optimizer_ != NULL) {
        delete optimizer_;
        optimizer_ = NULL;
    }
    arc_map_.clear();
    div_node_map_.clear();
    app_node_map_.clear();
    dis_node_map_.clear();
}

void ConservationTracking::add_appearance_nodes(const HypothesesGraph& g) {
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        pgm_->Model()->addVariable(max_number_objects_ + 1);
        app_node_map_[n] = pgm_->Model()->numberOfVariables() - 1;
        assert(pgm_->Model()->numberOfLabels(app_node_map_[n]) == max_number_objects_ + 1);
        ++count;
    }
    number_of_appearance_nodes_ = count;
}

void ConservationTracking::add_disappearance_nodes(const HypothesesGraph& g) {
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        pgm_->Model()->addVariable(max_number_objects_ + 1);
        dis_node_map_[n] = pgm_->Model()->numberOfVariables() - 1;
        assert(pgm_->Model()->numberOfLabels(dis_node_map_[n]) == max_number_objects_ + 1);
        ++count;
    }
    number_of_disappearance_nodes_ = count;
}

void ConservationTracking::add_transition_nodes(const HypothesesGraph& g) {
    size_t count = 0;
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
        pgm_->Model()->addVariable(max_number_objects_ + 1);
        arc_map_[a] = pgm_->Model()->numberOfVariables() - 1;
        assert(pgm_->Model()->numberOfLabels(arc_map_[a]) == max_number_objects_ + 1);
        ++count;
    }
    number_of_transition_nodes_ = count;
}

void ConservationTracking::add_division_nodes(const HypothesesGraph& g) {
    size_t count = 0;
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        size_t number_of_outarcs = 0;
        for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
            ++number_of_outarcs;
        }
        if (number_of_outarcs > 1) {
            pgm_->Model()->addVariable(2);
            div_node_map_[n] = pgm_->Model()->numberOfVariables() - 1;
            assert(pgm_->Model()->numberOfLabels(div_node_map_[n]) == 2);
            ++count;
        }
    }
    number_of_division_nodes_ = count;
}

namespace {
double get_transition_prob(double distance, size_t state, double alpha) {
    double prob = exp(-distance / alpha);
    if (state == 0) {
        return 1 - prob;
    }
    return prob;
}
}

void ConservationTracking::add_finite_factors(const HypothesesGraph& g) {
    LOG(logDEBUG) << "ConservationTracking::add_finite_factors: entered";
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map =
            g.get(node_tracklet());
    property_map<tracklet_intern_dist, HypothesesGraph::base_graph>::type& tracklet_intern_dist_map =
            g.get(tracklet_intern_dist());

    ////
    //// add detection factors
    ////
    LOG(logDEBUG) << "ConservationTracking::add_finite_factors: add detection factors";
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        size_t num_vars = 0;
        vector<size_t> vi;
        vector<double> cost;

        int node_begin_time = -1;
        int node_end_time = -1;
        if (with_tracklets_) {
            node_begin_time = tracklet_map[n].front().Timestep;
            node_end_time = tracklet_map[n].back().Timestep;
        } else {
            node_begin_time = traxel_map[n].Timestep;
            node_end_time = traxel_map[n].Timestep;
        }

        if (app_node_map_.count(n) > 0) {
            vi.push_back(app_node_map_[n]);
            if (node_begin_time <= g.earliest_timestep()) {  // "<" holds if there are only tracklets in the first frame
                // pay no appearance costs in the first timestep
                cost.push_back(0.);
            } else {
                if (with_tracklets_) {
                    cost.push_back(appearance_cost_(tracklet_map[n].front()));
                    LOG(logDEBUG4) << "App-costs 1: " << appearance_cost_(tracklet_map[n].front()) << ", " << tracklet_map[n].front();
                } else {
                    cost.push_back(appearance_cost_(traxel_map[n]));
                    LOG(logDEBUG4) << "App-costs 2: " << appearance_cost_(traxel_map[n]) << ", " << traxel_map[n];
                }
            }
            ++num_vars;
        }
		
		if (dis_node_map_.count(n) > 0) {
            vi.push_back(dis_node_map_[n]);
            double c = 0;
            if (node_end_time < g.latest_timestep()) { // "<" holds if there are only tracklets in the last frame
               
                if (with_tracklets_) {
					disappearance_cost_(tracklet_map[n].back());
                    LOG(logDEBUG4) << "fail!";
                    c += disappearance_cost_(tracklet_map[n].back());
                    LOG(logDEBUG4) << "Disapp-costs 1: " << disappearance_cost_(tracklet_map[n].back()) << ", " << tracklet_map[n].back();
                } else {
					LOG(logDEBUG4) << traxel_map[n];
                    c += disappearance_cost_(traxel_map[n]);
                    LOG(logDEBUG4) << "Disapp-costs 2: " << disappearance_cost_(traxel_map[n]) << ", " << traxel_map[n];
                }
            }
            cost.push_back(c);
            ++num_vars;
        }
        
        // convert vector to array
        vector<size_t> coords(num_vars, 0); // number of variables
        // ITER first_ogm_idx, ITER last_ogm_idx, VALUE init, size_t states_per_var
        pgm::OpengmExplicitFactor<double> table(vi.begin(), vi.end(), forbidden_cost_, (max_number_objects_ + 1));
        for (size_t state = 0; state <= max_number_objects_; ++state) {
            double energy = 0;
            if (with_tracklets_) {
                // add all detection factors of the internal nodes
                for (std::vector<Traxel>::const_iterator trax_it = tracklet_map[n].begin();
                        trax_it != tracklet_map[n].end(); ++trax_it) {
                    energy += detection_(*trax_it, state);
                }
                // add all transition factors of the internal arcs
                for (std::vector<double>::const_iterator intern_dist_it =
                        tracklet_intern_dist_map[n].begin();
                        intern_dist_it != tracklet_intern_dist_map[n].end(); ++intern_dist_it) {
                    energy += transition_(
                            get_transition_prob(*intern_dist_it, state, transition_parameter_));
                }
            } else {
                energy = detection_(traxel_map[n], state);
            }
            LOG(logDEBUG2) << "ConservationTracking::add_finite_factors: detection[" << state
                    << "] = " << energy;
            for (size_t var_idx = 0; var_idx < num_vars; ++var_idx) {
                coords[var_idx] = state;
                // if only one of the variables is > 0, then it is an appearance in this time frame
                // or a disappearance in the next timeframe. Hence, add the cost of appearance/disappearance
                // to the detection cost
                table.set_value(coords, energy + state * cost[var_idx]);
                coords[var_idx] = 0;
                LOG(logDEBUG4) << "ConservationTracking::add_finite_factors: var_idx "
                        << var_idx << " = " << energy;
            }
            // also this energy if both variables have the same state
            if (num_vars == 2) {
                coords[0] = state;
                coords[1] = state;
                // only pay detection energy if both variables are on
                table.set_value(coords, energy);
                coords[0] = 0;
                coords[1] = 0;

                LOG(logDEBUG4) << "ConservationTracking::add_finite_factors: var_idxs 0 and var_idx 1 = "
                        << energy;
            }
        }

        LOG(logDEBUG3) << "ConservationTracking::add_finite_factors: adding table to pgm";
        table.add_to(*(pgm_->Model()));
    }

    ////
    //// add transition factors
    ////
    LOG(logDEBUG) << "ConservationTracking::add_finite_factors: add transition factors";
    property_map<arc_distance, HypothesesGraph::base_graph>::type& arc_distances = g.get(
            arc_distance());
    for (HypothesesGraph::ArcIt a(g); a != lemon::INVALID; ++a) {
        size_t vi[] = { arc_map_[a] };
        vector<size_t> coords(1, 0); // number of variables
        // ITER first_ogm_idx, ITER last_ogm_idx, VALUE init, size_t states_per_var
        pgm::OpengmExplicitFactor<double> table(vi, vi + 1, forbidden_cost_, (max_number_objects_ + 1));
        for (size_t state = 0; state <= max_number_objects_; ++state) {
            double energy = transition_(get_transition_prob(arc_distances[a], state, transition_parameter_));
            LOG(logDEBUG2) << "ConservationTracking::add_finite_factors: transition[" << state
                    << "] = " << energy;
            coords[0] = state;
            table.set_value(coords, energy);
            coords[0] = 0;
        }
        table.add_to(*(pgm_->Model()));
    }

    ////
    //// add division factors
    ////
    if (with_divisions_) {
        LOG(logDEBUG) << "ConservationTracking::add_finite_factors: add division factors";
        for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
            if (div_node_map_.count(n) == 0) {
                continue;
            }
            size_t vi[] = { div_node_map_[n] };
            vector<size_t> coords(1, 0); // number of variables
            // ITER first_ogm_idx, ITER last_ogm_idx, VALUE init, size_t states_per_var
            pgm::OpengmExplicitFactor<double> table(vi, vi + 1, forbidden_cost_, 2);
            for (size_t state = 0; state <= 1; ++state) {
                double energy = 0;
                if (with_tracklets_) {
                    energy = division_(tracklet_map[n].back(), state);
                } else {
                    energy = division_(traxel_map[n], state);
                }
                LOG(logDEBUG2) << "ConservationTracking::add_finite_factors: division[" << state
                        << "] = " << energy;
                coords[0] = state;
                table.set_value(coords, energy);
                coords[0] = 0;
            }
            table.add_to(*(pgm_->Model()));
        }
    }


    if (!with_constraints_) {
    	for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
			LOG(logDEBUG) << "ConservationTracking::add_finite_factors: add soft-constraints for outgoing";

			// collect and count outgoing arcs
			  std::vector<HypothesesGraph::Arc> arcs;
			  std::vector<size_t> vi;
			  std::vector<size_t> states_vars;
			  states_vars.push_back(max_number_objects_+1);
			  vi.push_back(app_node_map_[n]); // first detection node, remaining will be transition nodes
			  bool has_div_node = false;
			  if (with_divisions_ && div_node_map_.count(n) != 0) {
				  vi.push_back(div_node_map_[n]);
				  has_div_node = true;
				  states_vars.push_back(2);
			  }

			  int count = 0;
			  //int trans_idx = vi.size();
			  for(HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
				  arcs.push_back(a);
				  vi.push_back(arc_map_[a]);
				  states_vars.push_back(max_number_objects_+1);
				  ++count;
			  }

			  // construct factor
			  // build value table
			  if (count != 0) {
				  //size_t table_dim = count + 1 + int(has_div_node); 		// n * transition var + detection var (+ division var)
				  std::vector<size_t> coords;
				  // ITER first_ogm_idx, ITER last_ogm_idx, VALUE init, size_t states_vars
				  pgm::OpengmExplicitFactor<double> table( vi.begin(), vi.end(), 0, states_vars);

				  //assert(table_dim - trans_idx == count);

				  ////
				  //// TODO: set the forbidden configurations to infinity or the allowed to zero
				  ////
              if (has_div_node) {
                    // TODO
              }
              
				  table.add_to(*(pgm_->Model()));
			  }



			  LOG(logDEBUG) << "ConservationTracking::add_finite_factors: add soft-constraints for incomfing";
			  // collect and count incoming arcs
			  arcs.clear();
			  vi.clear();
			  states_vars.clear();
			  states_vars.push_back(max_number_objects_+1);
			  vi.push_back(dis_node_map_[n]); // first detection node, remaining will be transition nodes

			  count = 0;
			  for(HypothesesGraph::InArcIt a(g, n); a != lemon::INVALID; ++a) {
				  arcs.push_back(a);
				  vi.push_back(arc_map_[a]);
				  states_vars.push_back(max_number_objects_+1);
				  ++count;
			  }
			  if (count != 0) {
				  // construct factor
				  // build value table
				  //size_t table_dim = count + 1; 		// n * transition var + detection var
				  std::vector<size_t> coords;
				  // ITER first_ogm_idx, ITER last_ogm_idx, VALUE init, size_t states_vars
				  pgm::OpengmExplicitFactor<double> table( vi.begin(), vi.end(), 0, states_vars);

				  //assert(table_dim - trans_idx == count);

				  ////
				  //// TODO: set the forbidden configurations to infinity or the allowed to zero
				  /////

				  table.add_to(*(pgm_->Model()));
			  }
    	}

    }
}

size_t ConservationTracking::cplex_id(size_t opengm_id, size_t state) {
    return optimizer_->lpNodeVi(opengm_id, state);
}

//set up optimizer from constraints by reading from formulated gm
void ConservationTracking::add_constraints(const HypothesesGraph& g) {
    size_t counter = 0;
    LOG(logDEBUG) << "ConservationTracking::add_constraints: entered";
    //typedef opengm::LPCplex<pgm::OpengmModelDeprecated::ogmGraphicalModel,
    //        pgm::OpengmModelDeprecated::ogmAccumulator> cplex;

    property_map<node_tracklet, HypothesesGraph::base_graph>::type& tracklet_map = g.get(
            node_tracklet());

    std::stringstream constraint_name;

    LOG(logDEBUG) << "ConservationTracking::add_constraints: transitions";
    for (HypothesesGraph::NodeIt n(g); n != lemon::INVALID; ++n) {
        std::stringstream traxel_names_ss;
        for (std::vector<Traxel>::const_iterator trax_it = tracklet_map[n].begin();
                trax_it != tracklet_map[n].end(); ++trax_it) {
            traxel_names_ss << trax_it->Id << "." << trax_it->Timestep << " ";
        }
        std::string traxel_names = traxel_names_ss.str();

        vector<size_t> cplex_idxs, cplex_idxs2;
        vector<int> coeffs, coeffs2;

        ////
        //// outgoing transitions
        ////
        size_t num_outarcs = 0;
        // couple detection and transitions: Y_ij <= App_i
        for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
			
            assert(app_node_map_.count(n) > 0
                    && "this node should be contained in app_node_map_ since it has outgoing arcs");
            
            for (size_t nu = 0; nu < max_number_objects_; ++nu) {
                for (size_t mu = nu + 1; mu <= max_number_objects_; ++mu) {
                    cplex_idxs.clear();
                    coeffs.clear();
                    coeffs.push_back(1);
                    cplex_idxs2.push_back(cplex_id(app_node_map_[n], nu));
                    coeffs.push_back(1);
                    cplex_idxs.push_back(cplex_id(arc_map_[a], mu));
                    // 0 <= App_i[nu] + Y_ij[mu] <= 1  forall mu>nu
                    constraint_name.str(std::string()); // clear the name
                    constraint_name << "outgoing: 0 <= App_i[" << nu << "] + Y_ij[" << mu << "] <= 1; ";
                    constraint_name << "g.id(n) = " << g.id(n) << ", g.id(a) = " << g.id(a) << ", Traxel " << traxel_names;
                    constraint_name << ", cid = " << ++counter;
                    optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(),
                            0, 1, constraint_name.str().c_str());
                    LOG(logDEBUG3) << constraint_name.str();
                }
            }
            ++num_outarcs;
        }

        int div_cplex_id = -1;
        if (with_divisions_ && div_node_map_.count(n) > 0) {
            LOG(logDEBUG3) << "div_node_map_[n] = " << div_node_map_[n];
            LOG(logDEBUG3) << "number_of_transition_nodes_ = " << number_of_transition_nodes_;
            LOG(logDEBUG3) << "number_of_division_nodes_ = " << number_of_division_nodes_;
            div_cplex_id = cplex_id(div_node_map_[n], 1);
        }
        if (num_outarcs > 0) {
            // couple transitions: sum(Y_ij) = D_i + App_i
            cplex_idxs.clear();
            coeffs.clear();
            for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
                for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                    coeffs.push_back(nu);
                    cplex_idxs.push_back(cplex_id(arc_map_[a], nu));
                }
            }
            if (div_cplex_id != -1) {
                cplex_idxs.push_back(div_cplex_id);
                coeffs.push_back(-1);
            }
            for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                coeffs.push_back(-nu);
                cplex_idxs.push_back(cplex_id(app_node_map_[n], nu));
            }

            // 0 <= sum_nu [ sum_j( nu * Y_ij[nu] ) ] - [ sum_nu nu * X_i[nu] + D_i[1] + sum_nu nu * App_i[nu] ]<= 0
            constraint_name.str(std::string()); // clear the name
            constraint_name << "couple transitions: ";
            constraint_name << " sum(Y_ij) = D_i + App_i added for Traxel " << traxel_names << ", "
                    << "n = " << app_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0, 0,
                    constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();

        }

        if (div_cplex_id != -1) {
            // couple detection and division: D_i = 1 => App_i = 1
            assert(app_node_map_.count(n) > 0
                    && "this node should be contained in app_node_map_ since it may divide");
            cplex_idxs.clear();
            coeffs.clear();

            cplex_idxs.push_back(div_cplex_id);
            coeffs.push_back(1);

            cplex_idxs.push_back(cplex_id(app_node_map_[n], 1));
            coeffs.push_back(-1);

            // -1 <= D_i[1] - App_i[1] <= 0
            constraint_name.str(std::string()); // clear the name
            constraint_name << "couple division and detection: ";
            constraint_name << " D_i=1 => App_i =1 added for Traxel " << traxel_names << ", " << "n = "
                    << app_node_map_[n] << ", d = " << div_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), -1, 0,
                    constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();

            // couple divsion and transition: D_1 = 1 => sum_k(Y_ik) = 2
            cplex_idxs2.clear();
            coeffs2.clear(); // -m <= 2 * D_i[1] - sum_j ( Y_ij[1] ) <= 0
            cplex_idxs2.push_back(div_cplex_id);
            coeffs2.push_back(2);

            for (HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
                for (size_t nu = 2; nu <= max_number_objects_; ++nu) {
                    // D_i[1] = 1 => Y_ij[nu] = 0 forall nu > 1
                    cplex_idxs.clear();
                    coeffs.clear();
                    cplex_idxs.push_back(div_cplex_id);
                    coeffs.push_back(1);
                    cplex_idxs.push_back(cplex_id(arc_map_[a], nu));
                    coeffs.push_back(1);

                    // 0 <= D_i[1] + Y_ij[nu] <= 1 forall nu>1
                    constraint_name.str(std::string()); // clear the name
                    constraint_name << "couple division and transition: ";
                    constraint_name << " D_i=1 => Y_i[nu]=0 added for Traxel " << traxel_names << ", "
                            << "d = " << div_node_map_[n] << ", y = " << arc_map_[a] << ", nu = "
                            << nu;
                    constraint_name << ", cid = " << ++counter;

                    optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(),
                            0, 1, constraint_name.str().c_str());
                    LOG(logDEBUG3) << constraint_name.str();

                }

                cplex_idxs2.push_back(cplex_id(arc_map_[a], 1));
                coeffs2.push_back(-1);
            }

            // -m <= 2 * D_i[1] - sum_j (Y_ij[1]) <= 0
            constraint_name.str(std::string()); // clear the name
            constraint_name << "couple division and transitions: ";
            constraint_name  << " D_i = 1 => sum_k(Y_ik) = 2 added for Traxel " << traxel_names << ", "
                    << "d = " << div_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs2.begin(), cplex_idxs2.end(), coeffs2.begin(),
                    -int(max_number_objects_), 0, constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();
        }

        ////
        //// incoming transitions
        ////
        // couple transitions: sum_k(Y_kj) = Dis_j
        cplex_idxs.clear();
        coeffs.clear();

        size_t num_inarcs = 0;
        for (HypothesesGraph::InArcIt a(g, n); a != lemon::INVALID; ++a) {
            for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                cplex_idxs.push_back(cplex_id(arc_map_[a], nu));
                coeffs.push_back(nu);
            }
            ++num_inarcs;
        }

        if (num_inarcs > 0) {
            assert(dis_node_map_.count(n) > 0
                    && "this node should be contained in dis_node_map_ since it has incoming arcs");
            for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                cplex_idxs.push_back(cplex_id(dis_node_map_[n], nu));
                coeffs.push_back(-nu);
            }

            // 0 <= sum_nu [ nu * sum_i (Y_ij[nu] ) ] - sum_nu ( nu * X_j[nu] ) - sum_nu ( nu * Dis_j[nu] ) <= 0
            constraint_name.str(std::string()); // clear the name
            constraint_name << "incoming transitions: ";
            constraint_name << " sum_k(Y_kj) = Dis_j added for Traxel " << traxel_names << ", " << "n = "
                    << dis_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0, 0,
                    constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();
        }

        ////
        //// disappearance/appearance coupling
        ////
        if (app_node_map_.count(n) > 0 && dis_node_map_.count(n) > 0) {
            for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                cplex_idxs.clear();
                coeffs.clear();

                cplex_idxs.push_back(cplex_id(app_node_map_[n], nu));
                coeffs.push_back(1);

                cplex_idxs.push_back(cplex_id(dis_node_map_[n], nu));
                coeffs.push_back(-1);

                cplex_idxs.push_back(cplex_id(dis_node_map_[n], 0));
                coeffs.push_back(-1);

                // A_i[nu] = 1 => V_i[nu] = 1 v V_i[0] = 1
                // -1 <= App_i[nu] - ( Dis_i[nu] + Dis_i[0] ) <= 0 forall nu > 0
                constraint_name.str(std::string()); // clear the name
                constraint_name << "disappearance/appearance coupling: ";
                constraint_name << " A_i[nu] = 1 => V_i[nu] = 1 v V_i[0] = 1 added for Traxel "
                        << traxel_names << ", " << "n = " << app_node_map_[n];
                constraint_name << ", cid = " << ++counter;
                optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), -1,
                        0, constraint_name.str().c_str());
                LOG(logDEBUG3) << constraint_name.str();
            }

            for (size_t nu = 1; nu <= max_number_objects_; ++nu) {
                cplex_idxs.clear();
                coeffs.clear();

                cplex_idxs.push_back(cplex_id(dis_node_map_[n], nu));
                coeffs.push_back(1);

                cplex_idxs.push_back(cplex_id(app_node_map_[n], nu));
                coeffs.push_back(-1);

                cplex_idxs.push_back(cplex_id(app_node_map_[n], 0));
                coeffs.push_back(-1);

                // V_i[nu] = 1 => A_i[nu] = 1 v A_i[0] = 1
                // -1 <= Dis_i[nu] - ( App_i[nu] + App_i[0] ) <= 0 forall nu > 0
                constraint_name.str(std::string()); // clear the name
                constraint_name << "disappearance/appearance coupling: ";
                constraint_name << " V_i[nu] = 1 => A_i[nu] = 1 v A_i[0] = 1 added for Traxel "
                        << traxel_names << ", " << "n = " << app_node_map_[n];
                constraint_name << ", cid = " << ++counter;
                optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), -1,
                        0, constraint_name.str().c_str());
                LOG(logDEBUG3) << constraint_name.str();
            }
        }

        if (!with_misdetections_allowed_) {
            cplex_idxs.clear();
            coeffs.clear();
            if (dis_node_map_.count(n) > 0) {
                cplex_idxs.push_back(cplex_id(dis_node_map_[n], 0));
                coeffs.push_back(1);
            }
            if (app_node_map_.count(n) > 0) {
                cplex_idxs.push_back(cplex_id(app_node_map_[n], 0));
                coeffs.push_back(1);
            }

            // V_i[0] = 0 => 1 <= A_i[0]
            // A_i[0] = 0 => 1 <= V_i[0]
            // V_i <= m, A_i <= m
            // 0 <= Dis_i[0] + App_i[0] <= 0
            constraint_name.str(std::string()); // clear the name
            constraint_name << "disappearance/appearance coupling: ";
            constraint_name << " A_i[0] + V_i[0] = 0 added for Traxel " << traxel_names;
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0, 0,
                    constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();
        }

        if (!with_disappearance_ && (dis_node_map_.count(n) > 0)) {
            cplex_idxs.clear();
            coeffs.clear();
            cplex_idxs.push_back(cplex_id(dis_node_map_[n], 0));
            coeffs.push_back(1);
            // V_i[0] = 0
            // 1 <= V_i <= m
            constraint_name.str(std::string()); // clear the name
            constraint_name << "disappearance/appearance coupling: ";
            constraint_name << " V_i[0] = 0 added for Traxel " << traxel_names << ", " << "n = "
                    << dis_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0,
                    0, constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();
        }

        if (!with_appearance_ && (app_node_map_.count(n) > 0)) {
            cplex_idxs.clear();
            coeffs.clear();
            cplex_idxs.push_back(cplex_id(app_node_map_[n], 0));
            coeffs.push_back(1);
            // A_i[0] = 0
            // 1 <= A_i <= m
            constraint_name.str(std::string()); // clear the name
            constraint_name << "disappearance/appearance coupling: ";
            constraint_name << " A_i[0] = 0 added for Traxel " << traxel_names << ", " << "n = "
                    << app_node_map_[n];
            constraint_name << ", cid = " << ++counter;
            optimizer_->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0,
                    0, constraint_name.str().c_str());
            LOG(logDEBUG3) << constraint_name.str();
        }
    }

}

} /* namespace pgmlink */
