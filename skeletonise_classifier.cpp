#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
                              // converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>

#define COMPUTE_COMPACTNESS false

#define LOG() std::cout<<__FILE__<<":"<<__LINE__<< std::endl;

#include <string>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;
typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<unsigned char> UCmap;
typedef Point_set::Property_map<float> Fmap;
namespace Classification = CGAL::Classification;
typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set                                               Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;


#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
namespace AW3 = CGAL::Alpha_wraps_3;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_Alpha;
typedef Kernel_Alpha::Point_3 Point_Alpha;
using Point_container = std::vector<Point_Alpha>;
using Mesh = CGAL::Surface_mesh<Point_Alpha>;



#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>


typedef CGAL::Mean_curvature_flow_skeletonization<Mesh> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                                  Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;

typedef boost::graph_traits<Mesh>::vertex_descriptor    vertex_descriptor;

typedef CGAL::Search_traits_3<Kernel_Alpha> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
typedef CGAL::Kd_tree<Traits> Tree;

;
typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
typedef boost::property_map<Mesh, CGAL::vertex_point_t>::type Vertex_point_pmap;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>                      K_neighbor_search;
typedef K_neighbor_search::Tree                                         Tree;
typedef Tree::Splitter                                                  Splitter;
typedef K_neighbor_search::Distance                                     Distance;


#include "skeletonise.h"

int main(int argc, char** argv)
{

    std::string filename = CGAL::data_file_path("points_3/b9_training.ply");
   // filename = "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/test.ply";
   //filename = "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20220630 - small.ply";
   filename = "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20220630_8trees_new.ply";
   filename = "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20210712_small_new2.ply";
    //filename = argv[1];
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input" << std::endl;
    in >> pts;
    std::cerr << "Input size: " << pts.number_of_points() << std::endl;
    if (pts.number_of_points() == 0) {
        std::cerr << "Error: no vertices loaded." << std::endl;
        return EXIT_FAILURE;
    }

    Imap label_map;
    bool lm_found = false;
    std::tie(label_map, lm_found) = pts.property_map<int>("label"); //scalar_Classification
    for (std::string prop : pts.properties()) { std::cout << prop << std::endl; }
    if (!lm_found)
    {
        Fmap label_fmap;
        std::tie(label_fmap, lm_found) = pts.property_map<float>("scalar_Classification");

        if (lm_found) {
            label_map = pts.add_property_map<int>("label", -1).first;
            for (int i = 0; i < pts.number_of_points(); ++i) {
                label_map[i] = (int)label_fmap[i];
            }
        } else {
            std::cerr << "Error: \"label\" property not found in input file." << std::endl;
            return EXIT_FAILURE;
        }
    }

    Feature_set features;
    std::cerr << "Generating features" << std::endl;
    CGAL::Real_timer t;
    t.start();
    int scales = 5;
    float voxelsize = 0.1; // 0.5 
    Feature_generator generator(pts, pts.point_map(),
        scales, voxelsize);  // using 5 scales

    std::cout << "Neighbourhood radii: " << generator.radius_neighbors(0);
    for (std::size_t i = 1; i < generator.number_of_scales(); ++i)
        std::cout << ", " << generator.radius_neighbors(i);
    std::cout << std::endl;

    features.begin_parallel_additions();
    generator.generate_point_based_features(features);
    features.end_parallel_additions();

    const float radius_size = (argc > 2) ? atof(argv[2]) : 0.1f; // specify radius of neighborhoods (default: 60cm, MAC suggests: 10cm)
    //const float voxel_size = radius_size / 3.f; // re-scale for CGAL's feature generator
    const float voxel_size = (radius_size * 5) / 3.f; // re-scale for CGAL's feature generator (multiply by 5 to not have problems with too fine of a grid)
   
    // wrap surface
    // Compute the alpha and offset values
    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.2f;//0.2;// 10. 
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;//2.f;// 300.;
    std::cout << "relative alpha = " << relative_alpha << " relative offset = " << relative_offset << std::endl;
    double alpha = radius_size / relative_alpha; // bbox / relative_alpha;
    double offset = radius_size / relative_offset; // bbox / relative_offset
    std::cout << "absolute alpha = " << alpha << " absolute offset = " << offset << std::endl;
    
    // convert to a kernel that is more stable for Alpha Wrap
    Point_container points;
    for (auto& point : pts.points()) {
        Point_Alpha pt(point.x(), point.y(), point.z());
        points.push_back(pt);
    }

    typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;


    // construct the wrap
    Mesh wrap;
    if (COMPUTE_COMPACTNESS) {
        CGAL::alpha_wrap_3(points, alpha, offset, wrap);
        std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces, " << std::endl;

        Mesh::Property_map<edge_descriptor, bool> is_constrained_map = wrap.add_property_map<edge_descriptor, bool>("e:is_constrained", false).first;

        CGAL::IO::write_polygon_mesh("wrap.ply", wrap, CGAL::parameters::stream_precision(17));
        std::cout << "Wrap saved" << std::endl;

        t.stop();
        std::cout << "Took " << t.time() << " s" << std::endl;
    }
    LOG();
    auto bbox = CGAL::bounding_box(CGAL::make_transform_iterator_from_property_map(pts.begin(), pts.point_map()),
        CGAL::make_transform_iterator_from_property_map(pts.end(), pts.point_map()));
    using Planimetric_grid = Classification::Planimetric_grid<Kernel, Point_set, Pmap>;
    //using MyFeature = CGAL::Classification::Feature::My_feature<Kernel, Point_set, Pmap>;
    using Myskeleton = CGAL::Classification::Feature::My_skeleton<Kernel, Point_set, Point_set::Point_map>;
    //Myskeleton my_skeleton(pts, pts.point_map(), 1, wrap);
    for (std::size_t i = 0; i < generator.number_of_scales(); ++i) {
        Planimetric_grid grid(pts, pts.point_map(), bbox, generator.grid_resolution(i));
         if (COMPUTE_COMPACTNESS)
             features.add_with_scale_id<Myskeleton>(i, pts, pts.point_map(), generator.radius_neighbors(i), wrap); //désactiver la classification
     }
    LOG();
        t.stop();
        std::cerr << "Done in " << t.time() << " second(s)" << std::endl;
        // Add labels
        Label_set labels;
        Label_handle ground = labels.add("ground");
        Label_handle vegetation = labels.add("vegetation");
        Label_handle roof = labels.add("roof");
        // Check if ground truth is valid for this label set
        if (!labels.is_valid_ground_truth(pts.range(label_map), true)) {
            std::cout << "Ground truths invalid." << std::endl;
            return EXIT_FAILURE;
        }
        LOG();
        std::vector<int> label_indices(pts.size(), -1);
        std::cerr << "Using ETHZ Random Forest Classifier" << std::endl;
        Classification::ETHZ::Random_forest_classifier classifier(labels, features);
        std::cerr << "Training" << std::endl;
        t.reset();
        t.start();
       classifier.train(pts.range(label_map));
        t.stop();
        std::cerr << "Done in " << t.time() << " second(s)" << std::endl;
        t.reset();
        t.start();
        
        Classification::classify_with_graphcut<CGAL::Parallel_if_available_tag>
            (pts, pts.point_map(), labels, classifier,
                generator.neighborhood().k_neighbor_query(12),
                0.2f, 1, label_indices);
        t.stop();
        std::cerr << "Classification with graphcut done in " << t.time() << " second(s)" << std::endl;
        std::cerr << "Precision, recall, F1 scores and IoU:" << std::endl;
        Classification::Evaluation evaluation(labels, pts.range(label_map), label_indices);
        for (Label_handle l : labels)
        {
            std::cerr << " * " << l->name() << ": "
                << evaluation.precision(l) << " ; "
                << evaluation.recall(l) << " ; "
                << evaluation.f1_score(l) << " ; "
                << evaluation.intersection_over_union(l) << std::endl;
        }
        LOG();
        std::cerr << "Accuracy = " << evaluation.accuracy() << std::endl
            << "Mean F1 score = " << evaluation.mean_f1_score() << std::endl
            << "Mean IoU = " << evaluation.mean_intersection_over_union() << std::endl;

        std::ofstream log_file("log_file.txt");
        log_file << "Accuracy = " << evaluation.accuracy() << std::endl
            << "Mean F1 score = " << evaluation.mean_f1_score() << std::endl
            << "Mean IoU = " << evaluation.mean_intersection_over_union() << std::endl;
        log_file.close();

        //Color point set according to class;
        UCmap red = pts.add_property_map<unsigned char>("red", 0).first;
        UCmap green = pts.add_property_map<unsigned char>("green", 0).first;
        UCmap blue = pts.add_property_map<unsigned char>("blue", 0).first;
        //Fmap label_pred = pts.add_property_map<float>("scalar_label_pred", -1).first;
        Imap label_pred = pts.add_property_map<int>("scalar_label_pred", -1).first;
        for (std::size_t i = 0; i < label_indices.size(); ++i)
        {
           // label_pred[i] = (float)label_indices[i]; // update label map with computed classification
            label_pred[i] = label_indices[i]; // update label map with computed classification
            Label_handle label = labels[label_indices[i]];
            const CGAL::IO::Color& color = label->color();
            red[i] = color.red();
            green[i] = color.green();
            blue[i] = color.blue();
        }

        //Stat
        std::vector<std::size_t> count;
        classifier.get_feature_usage(count);
        int total = 0;
        for (auto c : count)
            total += c;
        std::cout << "feature usage: " << std::endl;
        for (int i = 0; i < features.size(); ++i) {
            std::cout << "\t" << features[i]->name() << ": " << count[i] << "/" << total << " (" << 100.f * count[i] / total << "%)" << std::endl;
        }

        // Save configuration for later use
        std::ofstream fconfig("ethz_random_forest.bin", std::ios_base::binary);
        classifier.save_configuration(fconfig);
        // Write result
        std::ofstream f("classification_ethz_random_forest.ply");
        f.precision(18);
        f << pts;
        std::cerr << "All done" << std::endl;
        return EXIT_SUCCESS;
    
}
