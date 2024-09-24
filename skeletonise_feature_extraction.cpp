#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>

#define LOG() std::cout << __FILE__ << ":" << __LINE__ << std::endl;

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


#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>

#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/IO/write_ply_points.h>
#include <utility>

namespace AW3 = CGAL::Alpha_wraps_3;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_Alpha;
typedef Kernel_Alpha::Point_3 Point_Alpha;
using Point_container = std::vector<Point_Alpha>;
using Mesh = CGAL::Surface_mesh<Point_Alpha>;
typedef CGAL::Mean_curvature_flow_skeletonization<Mesh> Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;

typedef CGAL::Search_traits_3<Kernel_Alpha> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
typedef CGAL::Kd_tree<Traits> Tree;

typedef CGAL::Classification::Point_set_neighborhood<Kernel, Point_container, Pmap> Neighborhood;
typedef Neighborhood::K_neighbor_query KNeighborQuery;

std::vector<Point> convert_polylines_to_point_cloud(const std::vector<std::vector<Point>>& polylines, int num_samples) {
    std::vector<Point> point_cloud;

    for (const auto& polyline : polylines) {
        int num_points = polyline.size();

        if (num_points < 2) {
            continue;
        }

        // Calculate the step size to sample points evenly along the polyline
        float step_size = static_cast<float>(num_points - 1) / (num_samples + 1);

        // Sample points along the polyline
        for (int i = 1; i <= num_samples; i++) {
            float index = i * step_size;
            int prev_index = static_cast<int>(std::floor(index));
            int next_index = static_cast<int>(std::ceil(index));

            // Interpolate between the two nearest points
            float t = index - prev_index;
            const Point& prev_point = polyline[prev_index];
            const Point& next_point = polyline[next_index];

            Point interpolated_point;
            interpolated_point = Point(  (1 - t) * prev_point.x() + t * next_point.x(),
                                         (1 - t) * prev_point.y() + t * next_point.y(),
                                         (1 - t) * prev_point.z() + t * next_point.z());

            point_cloud.push_back(interpolated_point);
        }
    }

    return point_cloud;
}

// Function to calculate feature distance
float calculateFeatureDistance(const Point& p1, const Point& p2) {
    float dx = p1.x() - p2.x();
    float dy = p1.y() - p2.y();
    float dz = p1.z() - p2.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " filename scale n_scales alpha offset" << std::endl;
    }
    const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20210712_small_new2.ply";
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input" << std::endl;
    in >> pts;
    if (pts.number_of_points() == 0) {
        std::cerr << "Error: no vertices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Step 1: Create alpha shape wrapper
    const float radius_size = (argc > 2) ? atof(argv[2]) : 0.3f;
    const float voxel_size = (radius_size * 5) / 3.f;

    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.15f;
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;
    double alpha = radius_size / relative_alpha;
    double offset = radius_size / relative_offset;

    CGAL::Real_timer t;
    t.start();

    // convert to a kernel that is more stable for Alpha Wrap
    Point_container points;
    for (auto& point : pts.points()) {
        Point_Alpha pt(point.x(), point.y(), point.z());
        points.push_back(pt);
    }

    typedef boost::graph_traits<Mesh>::edge_descriptor edge_descriptor;

    // construct the wrap
    Mesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces, " << std::endl;

    Mesh::template Property_map<edge_descriptor, bool> is_constrained_map = wrap.template add_property_map<edge_descriptor, bool>("e:is_constrained", false).first;

    CGAL::IO::write_polygon_mesh("wrap.ply", wrap, CGAL::parameters::stream_precision(17));
    std::cout << "Wrap saved" << std::endl;

    t.stop();
    std::cout << "Took " << t.time() << " s" << std::endl;

    // Step 2: Extract mean curvature flow skeleton
    Skeleton skeleton;
    Skeletonization mcs(wrap);

    // 1. Contract the mesh by mean curvature flow.
    mcs.contract_geometry();
    // 2. Collapse short edges and split bad triangles.
    mcs.collapse_edges();
    mcs.split_faces();
    // 3. Fix degenerate vertices.
    mcs.detect_degeneracies();
    // Perform the above three steps in one iteration.
    mcs.contract();
    // Iteratively apply step 1 to 3 until convergence.
    mcs.contract_until_convergence();
    // Convert the contracted mesh into a curve skeleton and
    // get the correspondent surface points
    mcs.convert_to_skeleton(skeleton);
    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";

    // Step 3: Split the skeleton into polylines
    std::ofstream output("skel-poly.polylines.txt");
    typedef boost::graph_traits<Skeleton>::edge_descriptor Skeleton_edge;
    for (Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
        const Point& s = skeleton[source(e, skeleton)].point;
        const Point& t = skeleton[target(e, skeleton)].point;
        output << "2 " << s << " " << t << "\n";
    }
    output.close();

    // Step 4: Convert polylines to point cloud
    std::vector<std::vector<Point>> polylines;
    CGAL::split_graph_into_polylines(skeleton, std::back_inserter(polylines));

    std::vector<Point> skeleton_points = convert_polylines_to_point_cloud(polylines, 10);

    // Step 5: Construct a tree for efficient nearest neighbor queries
    Traits traits;
    Tree tree(skeleton_points.begin(), skeleton_points.end(), traits);

    // Step 6: Perform nearest neighbor search and calculate feature descriptors
    const int k = 1; // Number of neighbors to consider
    std::vector<std::vector<Point>> neighbors(skeleton_points.size());

    for (std::size_t r = 0; r < skeleton_points.size(); r++) {
        for (std::size_t s = 0; s < skeleton_points.size(); ++s) {
            Neighbor_search search(tree, skeleton_points[s], k);
            neighbors[s].reserve(k);

            for (const auto& pwd : search) {
                neighbors[s].push_back(pwd.first);
            }
        }
    }

    // Step 7: Calculate feature descriptors
    std::vector<float> feature_descriptors;
    for (std::size_t r = 0; r < skeleton_points.size(); r++) {
        const auto& neighbors_r = neighbors[r];
        float descriptor = 0.0f;

        for (const auto& neighbor : neighbors_r) {
            float distance = calculateFeatureDistance(skeleton_points[r], neighbor);
            descriptor += distance;
        }

        descriptor /= static_cast<float>(neighbors_r.size());
        feature_descriptors.push_back(descriptor);
    }

    // 'feature_descriptors' vector contains the feature descriptors for each point in the skeleton point cloud

    return 0;
}

/*


const unsigned int N = 1;
std::list<Point_d> points;
points.push_back(Point_d(0, 0));
Tree tree(points.begin(), points.end());
// Initialize the search structure, and search all N points
Point_d query(0, 0);
Neighbor_search search(tree, query, N);
// report the N nearest neighbors and their distance
// This should sort all N points by increasing distance from origin
for (Neighbor_search::iterator it = search.begin(); it != search.end(); ++it)
std::cout << it->first << " " << std::sqrt(it->second) << std::endl;
return 0;
}*/

/* #if defined(_MSC_VER) && !defined(_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
// converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
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
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;
//typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<unsigned char> UCmap;
typedef Point_set::Property_map<float> Fmap;
typedef CGAL::Identity_property_map<Point> Pmap; 

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap> Feature_generator;

#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>

#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/IO/write_ply_points.h>
#include <utility>

namespace AW3 = CGAL::Alpha_wraps_3;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_Alpha;
typedef Kernel_Alpha::Point_3 Point_Alpha;
using Point_container = std::vector<Point_Alpha>;
using Mesh = CGAL::Surface_mesh<Point_Alpha>;
typedef CGAL::Mean_curvature_flow_skeletonization<Mesh> Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;

typedef CGAL::Search_traits_3<Kernel_Alpha> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
typedef CGAL::Kd_tree<Traits> Tree;

typedef CGAL::Classification::Point_set_neighborhood<Kernel, Point_container, Pmap> Neighborhood;
typedef Neighborhood::K_neighbor_query KNeighborQuery;

// Function to calculate feature distance
float calculateFeatureDistance(const Point& p1, const Point& p2) {
    float dx = p1.x() - p2.x();
    float dy = p1.y() - p2.y();
    float dz = p1.z() - p2.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Function to convert polylines to point cloud
std::vector<Point> convert_polylines_to_point_cloud(const std::vector<std::vector<Point>>& polylines, int num_samples) {
    std::vector<Point> point_cloud;

    for (const auto& polyline : polylines) {
        int num_points = polyline.size();

        if (num_points < 2) {
            continue;
        }

        // Calculate the step size to sample points evenly along the polyline
        float step_size = static_cast<float>(num_points - 1) / (num_samples + 1);

        // Sample points along the polyline
        for (int i = 1; i <= num_samples; i++) {
            float index = i * step_size;
            int prev_index = static_cast<int>(std::floor(index));
            int next_index = static_cast<int>(std::ceil(index));

            // Interpolate between the two nearest points
            float t = index - prev_index;
            const Point& prev_point = polyline[prev_index];
            const Point& next_point = polyline[next_index];

            Point interpolated_point;
            interpolated_point.x() = (1 - t) * prev_point.x() + t * next_point.x();
            interpolated_point.y() = (1 - t) * prev_point.y() + t * next_point.y();
            interpolated_point.z() = (1 - t) * prev_point.z() + t * next_point.z();

            point_cloud.push_back(interpolated_point);
        }
    }

    return point_cloud;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " filename scale n_scales alpha offset" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Myproject/Chambon_large_displacements/Chambon_Scan_Riegl_20210712_small_new2.ply";
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input" << std::endl;
    in >> pts;
    if (pts.number_of_points() == 0) {
        std::cerr << "Error: no vertices found." << std::endl;
        return EXIT_FAILURE;
    }

    const float radius_size = (argc > 2) ? std::atof(argv[2]) : 0.3f; // specify radius of neighborhoods
    const float voxel_size = (radius_size * 5) / 3.f; // re-scale for CGAL's feature generator

    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.15f;
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;
    double alpha = radius_size / relative_alpha;
    double offset = radius_size / relative_offset;

    CGAL::Real_timer t;
    t.start();

    // Convert to a kernel that is more stable for Alpha Wrap
    Point_container points;
    for (auto& point : pts.points()) {
        Point_Alpha pt(point.x(), point.y(), point.z());
        points.push_back(pt);
    }

    typedef boost::graph_traits<Mesh>::edge_descriptor edge_descriptor;

    // Wrap the surface using alpha wrap
    Mesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces" << std::endl;

    t.stop();
    std::cout << "Wrap time: " << t.time() << " s" << std::endl;

    Skeleton skeleton;
    Skeletonization mcs(wrap);

    // Skeletonize the mesh using mean curvature flow
    mcs.contract_geometry();
    mcs.collapse_edges();
    mcs.split_faces();
    mcs.detect_degeneracies();
    mcs.contract();
    mcs.contract_until_convergence();
    mcs.convert_to_skeleton(skeleton);
    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << std::endl;
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << std::endl;

    // Split the skeleton into polylines
    std::vector<std::vector<Point>> skeleton_polylines;
    CGAL::split_graph_into_polylines(skeleton, std::back_inserter(skeleton_polylines));
    
    int num_samples = 10;
    std::vector<Point> skeleton_points = convert_polylines_to_point_cloud(skeleton_polylines, num_samples);

    // Convert polylines to a point cloud
    std::vector<Point> skeleton_points = convert_polylines_to_point_cloud(skeleton_polylines, num_samples);

    // Construct a tree for efficient nearest neighbor queries
    Tree tree(skeleton_points.begin(), skeleton_points.end());

    // Perform nearest neighbor search for each point
    Neighborhood neighborhood(points, Pmap());
    int n = 1; // number of neighbors
    KNeighborQuery neighbor_query = neighborhood.k_neighbor_query(n);

    
    std::vector<float> features;


    for (std::size_t i = 0; i < points.number_of_points(); ++i) {
        std::vector<std::size_t> neighbors;
        neighbor_query(get(Pmap(), *(points.begin() + i)), std::back_inserter(neighbors));
        Point closest_point = get(Pmap(), *(points.begin() + neighbors[0]));

        float d = calculateFeatureDistance(*(points.begin() + i), closest_point);
        features.push_back(d);
    }

    // 'features' vector contains the feature descriptors for each point in the input point cloud

    return 0;
}*/