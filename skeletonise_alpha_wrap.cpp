#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
// converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>

#define LOG() std::cout<<__FILE__<<":"<<__LINE__<< std::endl;

#include <string>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>
#include <CGAL/Classification/ETHZ/Random_forest_classifier.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef std::vector<Point> Point_container2;
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;
//typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<unsigned char> UCmap;
typedef Point_set::Property_map<float> Fmap;
typedef CGAL::Identity_property_map<Point> Pmap;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set                                               Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;

#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/IO/write_ply_points.h>
#include <utility>

typedef CGAL::Classification::Point_set_neighborhood<Kernel, Point_container2, Pmap>             Neighborhood;
typedef Neighborhood::K_neighbor_query KNeighborQuery;
//typedef CGAL::Search_traits_3<Kernel> TreeTraits;
//typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
//typedef Neighbor_search::Tree Tree2;


#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>


namespace AW3 = CGAL::Alpha_wraps_3;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_Alpha;
typedef Kernel_Alpha::Point_3 Point_Alpha;
using Point_container = std::vector<Point_Alpha>;
using Mesh = CGAL::Surface_mesh<Point_Alpha>;
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
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " filename scale n_scales alpha offset" << std::endl;
    }
  //  const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Skeletonization/elephant.off";
   // const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Skeletonization/Scan_2019_12_17_clean.ply";
   const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20210712_small_new.ply";
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input" << std::endl;
    in >> pts;
    if (pts.number_of_points() == 0) {
        std::cerr << "Error: no vertices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Step 1: Create alpha shape wrapper
   // Alpha_wrap_3 wrap = CGAL::alpha_wrap_3(points);

    std::cout << "No. of vertices: " << pts.number_of_points() << std::endl;

    std::cout << "Properties found:" << std::endl;
    for (auto prop : pts.properties_and_types()) {
        std::cout << "  " << prop.first << std::endl;
    }

    // radius_size = 0.6; relative_alpha = 0.1; relative_offset = 2.0;

    const float radius_size = (argc > 2) ? atof(argv[2]) : 0.05f; // specify radius of neighborhoods (default: 60cm, MAC suggests: 10cm)
    //const float voxel_size = radius_size / 3.f; // re-scale for CGAL's feature generator
    const float voxel_size = (radius_size * 5) / 3.f; // re-scale for CGAL's feature generator (multiply by 5 to not have problems with too fine of a grid)

    // wrap surface
    // Compute the alpha and offset values
    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.15f;//0.2;// 10. 
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;//2.f;// 300.;
    std::cout << "relative alpha = " << relative_alpha << " relative offset = " << relative_offset << std::endl;
    double alpha = radius_size / relative_alpha; // bbox / relative_alpha;
    double offset = radius_size / relative_offset; // bbox / relative_offset
    alpha = 0.666667; offset = 0.05;
    //offset = .10;
    std::cout << "absolute alpha = " << alpha << " absolute offset = " << offset << std::endl;

    CGAL::Real_timer t;
    t.start();

    // convert to a kernel that is more stable for Alpha Wrap
    Point_container points;
    for (auto& point : pts.points()) {
        Point_Alpha pt(point.x(), point.y(), point.z());
        points.push_back(pt);
    }

    typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;

    // construct the wrap
    Mesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces, " << std::endl;

    Mesh::Property_map<edge_descriptor, bool> is_constrained_map = wrap.add_property_map<edge_descriptor, bool>("e:is_constrained", false).first;

    CGAL::IO::write_polygon_mesh("wrap.ply", wrap, CGAL::parameters::stream_precision(17));
    std::cout << "Wrap saved" << std::endl;

    t.stop();
    std::cout << "Took " << t.time() << " s" << std::endl;

    using Myskeleton = CGAL::Classification::Feature::My_skeleton<Kernel, Point_set, Point_set::Point_map>;
    Myskeleton my_skeleton(pts, pts.point_map(), 1, wrap);
    
        // Step 6: Calculate feature descriptor for each point
   /* std::vector<float> features;
    for (const auto& point : points) {
        Point closest_point = tree.query(point);
        float d = std::sqrt((closest_point - point).squared_length());
        features.push_back(d);
    }*/

    // output results
    Fmap skeleton = pts.add_property_map<float>("scalar_Skeletonization", 0).first;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        if (i < 5) std::cout << "skeleton[" << i << "] " << my_skeleton.value(i) << std::endl;
        skeleton[i] = my_skeleton.value(i);
    }
    // Write result
    std::ofstream f("classification_ethz_random_forest.ply");
    f.precision(18);
    f << pts;

    std::cerr << "All done" << std::endl;
    return EXIT_SUCCESS;
}