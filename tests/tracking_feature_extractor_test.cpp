#define BOOST_TEST_MODULE tracking_feature_extractor_test

#include <vector>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "pgmlink/reasoner_constracking.h"
#include "pgmlink/tracking.h"
#include "pgmlink/tracking_feature_extractor.h"

#include <boost/python.hpp>

using namespace pgmlink;
using namespace pgmlink::features;

BOOST_AUTO_TEST_CASE( TrackingFeatureExtractor_SimpleMove ) {

    std::cout << "Constructing HypothesesGraph" << std::endl;
    std::cout << std::endl;

    using lemon::INVALID;

    std::cout << "Adding Traxels to TraxelStore" << std::endl;
    std::cout << std::endl;

    //  t=1      2
    //  o ------ o
    TraxelStore ts;
    Traxel n11, n21;
    feature_array com(feature_array::difference_type(3));
    feature_array divProb(feature_array::difference_type(1));
    n11.Id = 1; n11.Timestep = 1; com[0] = 0; com[1] = 0; com[2] = 0; divProb[0] = 0.1;
    n11.features["com"] = com; n11.features["divProb"] = divProb;
    add(ts,n11);
    n21.Id = 10; n21.Timestep = 2; com[0] = 0; com[1] = 0; com[2] = 0; divProb[0] = 0.1;
    n21.features["com"] = com; n21.features["divProb"] = divProb;
    add(ts,n21);

    std::cout << "Initialize Conservation tracking" << std::endl;
    std::cout << std::endl;

    FieldOfView fov(0, 0, 0, 0, 4, 5, 5, 5); // tlow, xlow, ylow, zlow, tup, xup, yup, zup
    ConsTracking tracking = ConsTracking(
                         2, // max_number_objects
                         false, // detection_by_volume
                         double(1.1), // avg_obj_size
                         20, // max_neighbor_distance
                         true, //with_divisions
                         0.3, // division_threshold
                         "none", // random_forest_filename
                         fov
                  );

    std::cout << "Run Conservation tracking" << std::endl;
    std::cout << std::endl;

    std::vector< std::vector<Event> > events = tracking(
                                ts,
                                0, // forbidden_cost
                                0.0, // ep_gap
                                false, // with_tracklets
                                10.0, //division_weight
                                10.0, //transition_weight
                                1500., // disappearance_cost,
                                1500., // appearance_cost
                                false, //with_merger_resolution
                                3, //n_dim
                                5, //transition_parameter
                                0 //border_width for app/disapp costs
                                )[0];

    boost::shared_ptr<HypothesesGraph> hypotheses_graph = tracking.get_hypo_graph();

    std::map<std::string, bool> config;
    config["node_timestep"] = hypotheses_graph->has_property(node_timestep());
    config["node_active"] = hypotheses_graph->has_property(node_active());
    config["node_active2"] = hypotheses_graph->has_property(node_active2());
    config["node_active_count"] = hypotheses_graph->has_property(node_active_count());
    config["node_offered"] = hypotheses_graph->has_property(node_offered());
    config["split_from"] = hypotheses_graph->has_property(split_from());
    config["division_active"] = hypotheses_graph->has_property(division_active());
    config["merger_resolved_to"] = hypotheses_graph->has_property(merger_resolved_to());
    config["node_originated_from"] = hypotheses_graph->has_property(node_originated_from());
    config["node_resolution_candidate"] = hypotheses_graph->has_property(node_resolution_candidate());
    config["arc_distance"] = hypotheses_graph->has_property(arc_distance());
    config["traxel_arc_id"] = hypotheses_graph->has_property(traxel_arc_id());
    config["arc_vol_ratio"] = hypotheses_graph->has_property(arc_vol_ratio());
    config["arc_from_timestep"] = hypotheses_graph->has_property(arc_from_timestep());
    config["arc_to_timestep"] = hypotheses_graph->has_property(arc_to_timestep());
    config["arc_active"] = hypotheses_graph->has_property(arc_active());
    config["arc_resolution_candidate"] = hypotheses_graph->has_property(arc_resolution_candidate());
    config["tracklet_intern_dist"] = hypotheses_graph->has_property(tracklet_intern_dist());
    config["tracklet_intern_arc_ids"] = hypotheses_graph->has_property(tracklet_intern_arc_ids());
    config["arc_active_count"] = hypotheses_graph->has_property(arc_active_count());
    config["node_traxel"] = hypotheses_graph->has_property(node_traxel());

    for(const auto it : config)
    {
        std::cout << it.first << ": " << it.second << std::endl;
    }

    set_solution(*hypotheses_graph, 0);

    std::cout << "Computing higher order features" << std::endl;

    TrackingFeatureExtractor extractor(*hypotheses_graph);
    extractor.compute_features();
    TrackingFeatureExtractor::JointFeatureVector joint_feature_vector;
    extractor.get_feature_vector(joint_feature_vector);

    for(size_t i = 0; i < joint_feature_vector.size(); i++)
    {
        std::cout << "Feature:\t" << extractor.get_feature_description(i) << "\t\tValue:\t" << joint_feature_vector[i] << std::endl;
    }
    std::cout << "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(TrackingFeatureExtractor_CplexMBest)
{
    std::cout << "Constructing HypothesesGraph" << std::endl;
    std::cout << std::endl;

    typedef HypothesesGraph::ArcIt ArcIt2;
    typedef HypothesesGraph::Arc Arc;
    typedef HypothesesGraph::NodeIt NodeIt;
    typedef HypothesesGraph::Node Node;
    using lemon::INVALID;

    std::cout << "Adding Traxels to TraxelStore" << std::endl;
    std::cout << std::endl;

    //  t=1       2       3
    //   o                 o
    //    |               |
    //     ------ o ------
    //    |               |
    //   o                 o
    TraxelStore ts;
    Traxel n11, n12, n21, n31, n32;
    feature_array com(feature_array::difference_type(3));
    feature_array divProb(feature_array::difference_type(1));
    feature_array detProb(feature_array::difference_type(2));
    //detProb[2]=0;
    n11.Id = 11; n11.Timestep = 1; com[0] = 1; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.4;detProb[1]=0.6;
    n11.features["com"] = com; n11.features["divProb"] = divProb; n11.features["detProb"] = detProb;
    add(ts,n11);
    n12.Id = 12; n12.Timestep = 1; com[0] = 3; com[1] = 2; com[2] = 3; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n12.features["com"] = com; n12.features["divProb"] = divProb; n12.features["detProb"] = detProb;
    add(ts,n12);

    n21.Id = 21; n21.Timestep = 2; com[0] = 2; com[1] = 2; com[2] = 3; divProb[0] = 0.5; detProb[0] = 0;detProb[1]=1;
    n21.features["com"] = com; n21.features["divProb"] = divProb; n21.features["detProb"] = detProb;
    add(ts,n21);

    n31.Id = 31; n31.Timestep = 3; com[0] = 2; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n31.features["com"] = com; n31.features["divProb"] = divProb; n31.features["detProb"] = detProb;
    add(ts,n31);
    n32.Id = 32; n32.Timestep = 3; com[0] = 3; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.3;detProb[1]=0.7;
    n32.features["com"] = com; n32.features["divProb"] = divProb; n32.features["detProb"] = detProb;
    add(ts,n32);

    std::cout << "Initialize Conservation tracking" << std::endl;
    std::cout << std::endl;

    vector<double> sigmas(5);
        sigmas[0]=0;
        sigmas[1]=0;
        sigmas[2]=10;
        sigmas[3]=10;
        sigmas[4]=10;

    UncertaintyParameter uparam(3,DiverseMbest,sigmas);//2 iterations, diverse, diverse_lambda=10

    FieldOfView fov(0, 0, 0, 0, 3, 5, 5, 5); // tlow, xlow, ylow, zlow, tup, xup, yup, zup

    ConsTracking tracking = ConsTracking(
                1, // max_number_objects
                false, // detection_by_volume
                double(1.1), // avg_obj_size
                20, // max_neighbor_distance
                true, //with_divisions
                0.3, // division_threshold
                "none", // random forest filename
                fov //field of view
                );

    EventVectorVectorVector events = tracking(ts,
             0, // forbidden_cost
             0.0, // ep_gap
             false, // with_tracklets
             10.0, //division_weight
             10.0, //transition_weight
             10., // disappearance_cost,
             10., // appearance_cost
             false, //with_merger_resolution
             3, //n_dim
             5, //transition_parameter
             0, //border_width for app/disapp costs
             true, //with_constraints
             uparam // uncertainty parameters
             );

    boost::shared_ptr<HypothesesGraph> hypotheses_graph = tracking.get_hypo_graph();

    std::cout << "Features for solution: 0" << std::endl;
    set_solution(*hypotheses_graph, 0);
    TrackingFeatureExtractor extractor(*hypotheses_graph);
    extractor.compute_features();
    TrackingFeatureExtractor::JointFeatureVector joint_feature_vector;
    extractor.get_feature_vector(joint_feature_vector);

    for(size_t i = 0; i < joint_feature_vector.size(); i++)
    {
        std::cout << "Feature:\t" << extractor.get_feature_description(i) << "\tValue:\t" << joint_feature_vector[i] << std::endl;
    }

    for(size_t m = 1; m < events.size(); m++)
    {
        std::cout << "\nFeatures for solution: " << m << std::endl;
        set_solution(*hypotheses_graph, m);
        TrackingFeatureExtractor extractor(*hypotheses_graph);
        extractor.compute_features();
        TrackingFeatureExtractor::JointFeatureVector joint_feature_vector_m;
        extractor.get_feature_vector(joint_feature_vector_m);

        size_t num_different = 0;
        for(size_t i = 0; i < joint_feature_vector_m.size(); i++)
        {
            std::cout << "Feature:\t" << extractor.get_feature_description(i) << "\tValue:\t" << joint_feature_vector_m[i] << std::endl;
            if(joint_feature_vector[i] != joint_feature_vector_m[i])
                num_different++;
        }

        // make sure that we have at least one deviation in features to the previous solution
        BOOST_CHECK(num_different > 0);
    }
    std::cout << "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(TrackingFeatureExtractor_FeatureFile)
{
    std::cout << "Constructing HypothesesGraph" << std::endl;
    std::cout << std::endl;

    typedef HypothesesGraph::ArcIt ArcIt2;
    typedef HypothesesGraph::Arc Arc;
    typedef HypothesesGraph::NodeIt NodeIt;
    typedef HypothesesGraph::Node Node;
    using lemon::INVALID;

    std::cout << "Adding Traxels to TraxelStore" << std::endl;
    std::cout << std::endl;

    //  t=1       2       3
    //   o                 o
    //    |               |
    //     ------ o ------
    //    |               |
    //   o                 o
    TraxelStore ts;
    Traxel n11, n12, n21, n31, n32;
    feature_array com(feature_array::difference_type(3));
    feature_array divProb(feature_array::difference_type(1));
    feature_array detProb(feature_array::difference_type(2));
    //detProb[2]=0;
    n11.Id = 11; n11.Timestep = 1; com[0] = 1; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.4;detProb[1]=0.6;
    n11.features["com"] = com; n11.features["divProb"] = divProb; n11.features["detProb"] = detProb;
    add(ts,n11);
    n12.Id = 12; n12.Timestep = 1; com[0] = 3; com[1] = 2; com[2] = 3; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n12.features["com"] = com; n12.features["divProb"] = divProb; n12.features["detProb"] = detProb;
    add(ts,n12);

    n21.Id = 21; n21.Timestep = 2; com[0] = 2; com[1] = 2; com[2] = 3; divProb[0] = 0.5; detProb[0] = 0;detProb[1]=1;
    n21.features["com"] = com; n21.features["divProb"] = divProb; n21.features["detProb"] = detProb;
    add(ts,n21);

    n31.Id = 31; n31.Timestep = 3; com[0] = 2; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n31.features["com"] = com; n31.features["divProb"] = divProb; n31.features["detProb"] = detProb;
    add(ts,n31);
    n32.Id = 32; n32.Timestep = 3; com[0] = 3; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.3;detProb[1]=0.7;
    n32.features["com"] = com; n32.features["divProb"] = divProb; n32.features["detProb"] = detProb;
    add(ts,n32);

    std::cout << "Initialize Conservation tracking" << std::endl;
    std::cout << std::endl;

    vector<double> sigmas(5);
        sigmas[0]=0;
        sigmas[1]=0;
        sigmas[2]=10;
        sigmas[3]=10;
        sigmas[4]=10;

    UncertaintyParameter uparam(3,DiverseMbest,sigmas);//2 iterations, diverse, diverse_lambda=10

    FieldOfView fov(0, 0, 0, 0, 3, 5, 5, 5); // tlow, xlow, ylow, zlow, tup, xup, yup, zup

    ConsTracking tracking = ConsTracking(
                1, // max_number_objects
                false, // detection_by_volume
                double(1.1), // avg_obj_size
                20, // max_neighbor_distance
                true, //with_divisions
                0.3, // division_threshold
                "none", // random forest filename
                fov //field of view
                );

    EventVectorVectorVector events = tracking(ts,
             0, // forbidden_cost
             0.0, // ep_gap
             false, // with_tracklets
             10.0, //division_weight
             10.0, //transition_weight
             10., // disappearance_cost,
             10., // appearance_cost
             false, //with_merger_resolution
             3, //n_dim
             5, //transition_parameter
             0, //border_width for app/disapp costs
             true, //with_constraints
             uparam // uncertainty parameters
             );

    // remove file before populating
    std::string proposal_feature_filename = "proposal_features.txt";
    boost::filesystem::remove(proposal_feature_filename);
    boost::shared_ptr<HypothesesGraph> hypotheses_graph = tracking.get_hypo_graph();

    size_t feature_vector_length = 0;

    for(size_t m = 0; m < events.size(); m++)
    {
        set_solution(*hypotheses_graph, m);
        TrackingFeatureExtractor extractor(*hypotheses_graph);
        extractor.compute_features();
        TrackingFeatureExtractor::JointFeatureVector joint_feature_vector_m;
        extractor.get_feature_vector(joint_feature_vector_m);
        extractor.append_feature_vector_to_file(proposal_feature_filename);
        feature_vector_length = joint_feature_vector_m.size();
    }

    // read file again
    std::vector< std::vector <double> > proposal_features;
    std::ifstream feature_vector_file(proposal_feature_filename.c_str());
    if(feature_vector_file.good())
    {
        while(!feature_vector_file.eof())
        {
            // read line and remove comments and whitespace
            std::string line;
            std::getline(feature_vector_file, line);
            std::string::size_type comment_start = line.find('#');
            if(comment_start != std::string::npos)
                line = line.substr(comment_start);
            boost::algorithm::trim(line);

            // skip lines without features
            if(line.size() == 0)
                continue;

            // read features
            std::stringstream linestream(line);
            proposal_features.push_back(std::vector<double>());
            while(!linestream.eof())
            {
                double f;
                linestream >> f;
                proposal_features.back().push_back(f);
            }
        }
    }

    BOOST_CHECK(proposal_features.size() == feature_vector_length);
    for(size_t i = 0; i < feature_vector_length; i++)
    {
        BOOST_CHECK(proposal_features[i].size() == events.size());
    }
}

BOOST_AUTO_TEST_CASE(TrackingFeatureExtractor_LabelExport)
{
    std::cout << "Constructing HypothesesGraph" << std::endl;
    std::cout << std::endl;

    typedef HypothesesGraph::ArcIt ArcIt2;
    typedef HypothesesGraph::Arc Arc;
    typedef HypothesesGraph::NodeIt NodeIt;
    typedef HypothesesGraph::Node Node;
    using lemon::INVALID;

    std::cout << "Adding Traxels to TraxelStore" << std::endl;
    std::cout << std::endl;

    //  t=1       2       3
    //   o                 o
    //    |               |
    //     ------ o ------
    //    |               |
    //   o                 o
    TraxelStore ts;
    Traxel n11, n12, n21, n31, n32;
    feature_array com(feature_array::difference_type(3));
    feature_array divProb(feature_array::difference_type(1));
    feature_array detProb(feature_array::difference_type(2));
    //detProb[2]=0;
    n11.Id = 11; n11.Timestep = 1; com[0] = 1; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.4;detProb[1]=0.6;
    n11.features["com"] = com; n11.features["divProb"] = divProb; n11.features["detProb"] = detProb;
    add(ts,n11);
    n12.Id = 12; n12.Timestep = 1; com[0] = 3; com[1] = 2; com[2] = 3; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n12.features["com"] = com; n12.features["divProb"] = divProb; n12.features["detProb"] = detProb;
    add(ts,n12);

    n21.Id = 21; n21.Timestep = 2; com[0] = 2; com[1] = 2; com[2] = 3; divProb[0] = 0.5; detProb[0] = 0;detProb[1]=1;
    n21.features["com"] = com; n21.features["divProb"] = divProb; n21.features["detProb"] = detProb;
    add(ts,n21);

    n31.Id = 31; n31.Timestep = 3; com[0] = 2; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.6;detProb[1]=0.4;
    n31.features["com"] = com; n31.features["divProb"] = divProb; n31.features["detProb"] = detProb;
    add(ts,n31);
    n32.Id = 32; n32.Timestep = 3; com[0] = 3; com[1] = 1; com[2] = 1; divProb[0] = 0; detProb[0] = 0.3;detProb[1]=0.7;
    n32.features["com"] = com; n32.features["divProb"] = divProb; n32.features["detProb"] = detProb;
    add(ts,n32);

    std::cout << "Initialize Conservation tracking" << std::endl;
    std::cout << std::endl;

    vector<double> sigmas(5);
        sigmas[0]=0;
        sigmas[1]=0;
        sigmas[2]=10;
        sigmas[3]=10;
        sigmas[4]=10;

    UncertaintyParameter uparam(3,DiverseMbest,sigmas);//2 iterations, diverse, diverse_lambda=10

    FieldOfView fov(0, 0, 0, 0, 3, 5, 5, 5); // tlow, xlow, ylow, zlow, tup, xup, yup, zup

    ConsTracking tracking = ConsTracking(
                1, // max_number_objects
                false, // detection_by_volume
                double(1.1), // avg_obj_size
                20, // max_neighbor_distance
                true, //with_divisions
                0.3, // division_threshold
                "none", // random forest filename
                fov //field of view
                );

    EventVectorVectorVector events = tracking(ts,
             0, // forbidden_cost
             0.0, // ep_gap
             false, // with_tracklets
             10.0, //division_weight
             10.0, //transition_weight
             10., // disappearance_cost,
             10., // appearance_cost
             false, //with_merger_resolution
             3, //n_dim
             5, //transition_parameter
             0, //border_width for app/disapp costs
             true, //with_constraints
             uparam // uncertainty parameters
             );


    std::string proposal_labels_filename = "proposal_labels.txt";
    tracking.save_ilp_solutions(proposal_labels_filename);

    // read file again
    std::vector< std::vector <size_t> > proposal_labels;
    std::ifstream proposal_labels_file(proposal_labels_filename.c_str());
    if(proposal_labels_file.good())
    {
        while(!proposal_labels_file.eof())
        {
            // read line and remove comments and whitespace
            std::string line;
            std::getline(proposal_labels_file, line);
            std::string::size_type comment_start = line.find('#');
            if(comment_start != std::string::npos)
                line = line.substr(comment_start);
            boost::algorithm::trim(line);

            // skip lines without labels
            if(line.size() == 0)
                continue;

            // read labels
            std::stringstream linestream(line);
            proposal_labels.push_back(std::vector<size_t>());
            while(!linestream.eof())
            {
                size_t l;
                linestream >> l;
                proposal_labels.back().push_back(l);
            }
        }
    }

    BOOST_CHECK(proposal_labels[0].size() == events.size());
}

std::vector<double> execute_ssvm_python(const std::string& ground_truth,
                         const std::string& proposals,
                         const std::string& proposal_features)
{
    using namespace boost::python;

    try
    {
        object main_module = import("__main__");
        object main_namespace = main_module.attr("__dict__"); // get namespace
        exec("import structsvm", main_namespace);
        exec("problem = structsvm.funkey_problem('structsvm/features.txt', 'structsvm/constraints.txt', 'structsvm/labels.txt', [1,1,1,1,1])", main_namespace);
        exec("solver = structsvm.struct_svm_solver_bundle(problem)", main_namespace);
        exec("weights = solver.solve()", main_namespace);
        object python_weights = extract< boost::python::list >(main_namespace["weights"]);
        std::vector<double> weights;
        for(size_t i = 0; i < len(python_weights); i++)
        {
            weights.push_back(extract<double>(python_weights[i]));
        }
        return weights;
    }
    catch (error_already_set const&)
    {
        PyErr_Print();
        return std::vector<double>();
    }
}

BOOST_AUTO_TEST_CASE(TrackingFeatureExtractor_PythonLearner)
{
    Py_Initialize();
    std::vector<double> weights = execute_ssvm_python("bla.txt", "blupp.txt","foobar.txt");
    BOOST_CHECK(weights.size() == 2);
}

// EOF
