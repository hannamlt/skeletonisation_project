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


/* typedef CGAL::Identity_property_map<Point_Alpha> Pmap;
    typedef CGAL::Classification::Point_set_neighborhood<Kernel_Alpha, Point_container, Pmap>
        Neighborhood;
    typedef Neighborhood::K_neighbor_query KNeighborQuery;
*/

//typedef boost::graph_traits<Mesh>::vertex_descriptor Point;
typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
typedef boost::property_map<Mesh, CGAL::vertex_point_t>::type Vertex_point_pmap;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>                      K_neighbor_search;
typedef K_neighbor_search::Tree                                         Tree;
typedef Tree::Splitter                                                  Splitter;
typedef K_neighbor_search::Distance                                     Distance;

std::vector<Point> convert_polylines_to_point_cloud(const std::vector<std::vector<Point>>& polylines, int num_samples) {
    std::vector<Point> point_cloud;
    std::cout << polylines.size() << std::endl;
    for (const auto& polyline : polylines) {
        int num_points = polyline.size();
        //std::cout << polyline.size() << " / " << polylines.size() << std::endl;

        if (num_points < 2) {
            continue;
        }

        // Calculate the step size to sample points evenly along the polyline
        float step_size = static_cast<float>(num_points - 1) / (num_samples + 1);

        // Sample points along the polyline
        for (int i = 0; i <= num_samples; i++) {
            float index = i * step_size;
            int prev_index = static_cast<int>(std::floor(index));
            int next_index = static_cast<int>(std::ceil(index));

            // Interpolate between the two nearest points
            float t = index - prev_index;
            const Point& prev_point = polyline[prev_index];
            const Point& next_point = polyline[next_index];

            Point interpolated_point;
            interpolated_point = Point((1 - t) * prev_point.x() + t * next_point.x(),
                                       (1 - t) * prev_point.y() + t * next_point.y(),
                                       (1 - t) * prev_point.z() + t * next_point.z());

            point_cloud.push_back(interpolated_point);
        }
    }

    return point_cloud;
}


int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " filename scale n_scales alpha offset" << std::endl;
    }
  // const std::string filename = (argc > 1) ? argv[1] : "C:/Users/hmalet/Documents/Skeletonization/elephant.off";
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

    const float radius_size = (argc > 2) ? atof(argv[2]) : 0.3f; // specify radius of neighborhoods (default: 60cm, MAC suggests: 10cm)
    //const float voxel_size = radius_size / 3.f; // re-scale for CGAL's feature generator
    const float voxel_size = (radius_size * 5) / 3.f; // re-scale for CGAL's feature generator (multiply by 5 to not have problems with too fine of a grid)

    // wrap surface
    // Compute the alpha and offset values
    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.15f;//0.2;// 10. 
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;//2.f;// 300.;
    std::cout << "relative alpha = " << relative_alpha << " relative offset = " << relative_offset << std::endl;
    double alpha = radius_size / relative_alpha; // bbox / relative_alpha;
    double offset = radius_size / relative_offset; // bbox / relative_offset
    alpha = 0.5; offset = 0.1;
    //alpha = 0.0666667; offset = 0.05;
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

    // Step 2: Extract mean curvature flow skeleton
   // Skeleton_graph skeleton_graph = CGAL::extract_mean_curvature_flow_skeleton(wrap);

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
    //Skeleton_polylines skeleton_polylines = CGAL::split_graph_into_polylines(skeleton_graph);

    //only needed for the display of the skeleton as maximal polylines
    struct Display_polylines {
        const Skeleton& skeleton;
        std::ofstream& out;
        int polyline_size;
        std::stringstream sstr;
        Display_polylines(const Skeleton& skeleton, std::ofstream& out)
            : skeleton(skeleton), out(out)
        {}
        void start_new_polyline() {
            polyline_size = 0;
            sstr.str("");
            sstr.clear();
        }
        void add_node(Skeleton_vertex v) {
            ++polyline_size;
            sstr << " " << skeleton[v].point;
        }
        void end_polyline()
        {
            out << polyline_size << sstr.str() << "\n";
        }
    };

    CGAL::extract_mean_curvature_flow_skeleton(wrap, skeleton);

    // Output all the edges of the skeleton.
    std::ofstream output("skel-poly.polylines.txt");
    Display_polylines display(skeleton, output);
    CGAL::split_graph_into_polylines(skeleton, display);
    output.close();

    // Output skeleton points and the corresponding surface points
    output.open("correspondance-poly.polylines.txt");
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
        for (vertex_descriptor vd : skeleton[v].vertices)
            output << "2 " << skeleton[v].point << " "
            << get(CGAL::vertex_point, wrap, vd) << "\n";


    // Step 4: Convert polylines to point cloud  
        //adjacent_vertices()
    std::vector<std::vector<Point>> polylines;
    for (Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
        Point_Alpha s = skeleton[source(e, skeleton)].point;
        Point_Alpha t = skeleton[target(e, skeleton)].point;
        std::cout << "2 " << s << " " << t << "\n";
        Point s1(s.x(), s.y(), s.z());
        Point t1(t.x(), t.y(), t.z());
        polylines.push_back(std::vector<Point>({ s1,t1 }));
    }
    int num_samples = 10;
    std::vector<Point> pc = convert_polylines_to_point_cloud(polylines, num_samples);

    // Step 5: Construct a tree for efficient nearest neighbor queries
        // Tree tree = construct_tree(skeleton_points);


   // Nearest neighbor searching

    
    Point_set pts2;
    for (auto p : pc)
        pts2.insert(p);
    std::ofstream f1("poly_points.ply");
    f1.precision(18);
    f1 << pts2;

    /*
    Tree2 tree2(pc.begin(), pc.end());

    Point query(0, 0, 0);
    Neighbor_search search(tree2, query, 10);
    for (Neighbor_search::iterator it = search.begin(); it != search.end(); ++it)
        std::cout << it->first << " " << std::sqrt(it->second) << std::endl;
        */
    std::vector<float> values;
    values.reserve(pts.number_of_points());
    Pmap point_map = Pmap();
    Neighborhood neighborhood(pc, point_map);
    int n = 1; // number of neighbours
   KNeighborQuery neighbor_query = neighborhood.k_neighbor_query(n);
    for (std::size_t i = 0; i < pts.number_of_points(); ++i) {
        std::vector<std::size_t> neighbors;
        const Point& point = get(pts.point_map(), *(pts.begin() + i));
        neighbor_query(point, std::back_inserter(neighbors));
        Point& closest_point = get(point_map, *(pc.begin() + neighbors[0]));

         float d = CGAL::sqrt((closest_point - point).squared_length());
         values.push_back(d);
        }

        // Then perform your feature calculation, etc.
       // float d = ...
         //   feature[i] = d;


     // Perform nearest neighbor search for each point

    // Step 6: Calculate feature descriptor for each point
  
    std::cerr << "All done" << std::endl;
    return EXIT_SUCCESS;
}